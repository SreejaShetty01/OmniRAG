from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, status

from omnirag.api.dependencies import (
    RequestIdentity, get_product_store, get_profile_execution_service, get_request_identity,
    get_retrieval_scope,
)
from omnirag.api.schemas import ChatRequest, ChatTurnResponse
from omnirag.infrastructure.product_store import ProductStore
from omnirag.models.query import Query
from omnirag.profiles.execution import ProfileExecutionService
from omnirag.retrieval.contracts import RetrievalScope
from omnirag.config.profiles import PROFILE_CONFIGS
from omnirag.config.settings import get_settings
from omnirag.experiments.runtime import ExperimentRunStore, ExperimentRunner

router = APIRouter(tags=["chat"])
Identity = Annotated[RequestIdentity, Depends(get_request_identity)]
Store = Annotated[ProductStore, Depends(get_product_store)]
Executor = Annotated[ProfileExecutionService, Depends(get_profile_execution_service)]
Scope = Annotated[RetrievalScope, Depends(get_retrieval_scope)]


@router.post("/chat", response_model=ChatTurnResponse, status_code=status.HTTP_200_OK)
async def chat(payload: ChatRequest, identity: Identity, store: Store,
               executor: Executor, scope: Scope):
    preferences = store.preferences(identity.user_id)
    conversation = None
    conversation_id = payload.conversation_id
    if conversation_id:
        conversation = store.conversation(identity.user_id, conversation_id, identity.tenant_id)
        if conversation is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
    profile = payload.selected_profile or (conversation["selected_profile"] if conversation else
                                           preferences["default_profile"])
    escalation = payload.escalation_preference or preferences["escalation_preference"]
    if payload.collection_id and not store.collection(identity.user_id, payload.collection_id, identity.tenant_id):
        raise HTTPException(status_code=404, detail="Collection not found")
    if conversation_id is None:
        conversation_id = str(uuid4())
    history = store.messages(identity.user_id, conversation_id, identity.tenant_id) if conversation else []
    context = [{"role": item["role"], "content": item["content"]} for item in history[-12:]]
    filters = {"collection_id": payload.collection_id} if payload.collection_id else {}
    query = Query(query_id=str(uuid4()), conversation_id=conversation_id,
                  query_text=payload.query_text, selected_profile=profile,
                  escalation_preference=escalation, filters=filters,
                  conversation_context=context)
    if payload.experiment is None:
        response = await executor.execute(query, scope)
    else:
        runner = ExperimentRunner(ExperimentRunStore(get_settings().data_dir / "experiments" / "runs"))
        policy = executor._retrieval_policy(PROFILE_CONFIGS[profile])
        policy = payload.experiment.apply_to_policy(policy, PROFILE_CONFIGS[profile])
        response = await runner.run(
            profile=profile.value, config=payload.experiment, settings=get_settings(),
            retrieval_config={
                "retrieval_policy": policy.model_dump(mode="json"),
                "profile_policy": executor._policy_summary(PROFILE_CONFIGS[profile], policy),
            },
            resolved_switches=payload.experiment.resolved_switches(PROFILE_CONFIGS[profile]),
            execute=lambda: executor.execute(query, scope, experiment=payload.experiment),
        )
    if conversation is None:
        store.create_conversation(identity.user_id, payload.query_text[:72], profile.value,
                                  conversation_id=conversation_id, tenant_id=identity.tenant_id)
    user_message_id, assistant_message_id = store.save_chat_turn(
        identity.user_id, conversation_id, payload.query_text,
        response.model_dump(mode="json"), profile.value, response.profile_used.value,
        tenant_id=identity.tenant_id)
    return ChatTurnResponse(conversation_id=conversation_id, user_message_id=user_message_id,
                            assistant_message_id=assistant_message_id, response=response)
