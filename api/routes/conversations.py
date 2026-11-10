from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from api.dependencies import RequestIdentity, get_product_store, get_request_identity
from api.schemas import ConversationSummary
from infrastructure.product_store import ProductStore

router = APIRouter(prefix="/conversations", tags=["conversations"])
Identity = Annotated[RequestIdentity, Depends(get_request_identity)]
Store = Annotated[ProductStore, Depends(get_product_store)]


@router.get("", response_model=list[ConversationSummary])
def conversations(identity: Identity, store: Store):
    return store.list_conversations(identity.user_id, identity.tenant_id)


@router.get("/{conversation_id}")
def get_conversation(conversation_id: str, identity: Identity, store: Store):
    conversation = store.conversation(identity.user_id, conversation_id, identity.tenant_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {**conversation, "messages": store.messages(identity.user_id, conversation_id, identity.tenant_id)}


@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_conversation(conversation_id: str, identity: Identity, store: Store):
    if not store.delete_conversation(identity.user_id, conversation_id, identity.tenant_id):
        raise HTTPException(status_code=404, detail="Conversation not found")
