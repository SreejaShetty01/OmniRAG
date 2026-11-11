from typing import Annotated

from fastapi import APIRouter, Depends

from omnirag.api.dependencies import RequestIdentity, get_product_store, get_request_identity
from omnirag.api.schemas import PreferencesResponse, PreferencesUpdate
from omnirag.infrastructure.product_store import ProductStore

router = APIRouter(prefix="/preferences", tags=["preferences"])
Identity = Annotated[RequestIdentity, Depends(get_request_identity)]
Store = Annotated[ProductStore, Depends(get_product_store)]


@router.get("", response_model=PreferencesResponse)
def get_preferences(identity: Identity, store: Store):
    return store.preferences(identity.user_id)


@router.put("", response_model=PreferencesResponse)
def update_preferences(payload: PreferencesUpdate, identity: Identity, store: Store):
    values = payload.model_dump(exclude_unset=True)
    if "default_profile" in values and values["default_profile"] is not None:
        values["default_profile"] = values["default_profile"].value
    if "escalation_preference" in values and values["escalation_preference"] is not None:
        values["escalation_preference"] = values["escalation_preference"].value
    return store.update_preferences(identity.user_id, **values)
