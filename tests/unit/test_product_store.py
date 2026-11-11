from omnirag.infrastructure.product_store import ProductStore


def test_preferences_persist_across_store_instances(tmp_path):
    path = tmp_path / "product.sqlite3"
    store = ProductStore(path)
    store.update_preferences("user-a", preferred_name="Alex", default_profile="research")
    reopened = ProductStore(path)
    assert reopened.preferences("user-a")["preferred_name"] == "Alex"
    assert reopened.preferences("user-a")["default_profile"] == "research"
    assert reopened.update_preferences("user-a", preferred_name=None)["preferred_name"] is None
    assert reopened.preferences("user-b")["preferred_name"] is None


def test_conversations_and_messages_are_scoped_by_owner(tmp_path):
    store = ProductStore(tmp_path / "product.sqlite3")
    conversation_id = store.create_conversation("user-a", "Question", "lite")
    store.save_chat_turn("user-a", conversation_id, "Question", {"answer": "Grounded"}, "lite", "lite")
    assert store.conversation("user-b", conversation_id) is None
    assert store.messages("user-b", conversation_id) == []
    assert store.messages("user-a", conversation_id)[1]["content"] == "Grounded"


def test_product_records_are_isolated_by_tenant_for_same_user(tmp_path):
    store = ProductStore(tmp_path / "product.sqlite3")
    store.upsert_document(document_id="shared-doc", user_id="same-user", tenant_id="tenant-a",
                          source_path="/tmp/a.txt", filename="a.txt", version="v1", collection_id=None,
                          status="ready", source_type="document", document_type="text", source_hash="hash",
                          chunk_count=1, size_bytes=10)
    assert store.document("same-user", "shared-doc", "tenant-a") is not None
    assert store.document("same-user", "shared-doc", "tenant-b") is None
    assert store.list_documents("same-user", tenant_id="tenant-b") == []
    assert not store.delete_document_record("same-user", "shared-doc", "tenant-b")
    assert store.document("same-user", "shared-doc", "tenant-a") is not None

    collection_a = store.create_collection("same-user", "Shared name", "", "tenant-a")
    collection_b = store.create_collection("same-user", "Other name", "", "tenant-b")
    assert store.collection("same-user", collection_a, "tenant-b") is None
    assert [item["collection_id"] for item in store.list_collections("same-user", "tenant-b")] == [collection_b]

    conversation_a = store.create_conversation("same-user", "Tenant A", "lite", tenant_id="tenant-a")
    store.save_chat_turn("same-user", conversation_a, "private question", {"answer": "private"},
                         "lite", "lite", tenant_id="tenant-a")
    assert store.conversation("same-user", conversation_a, "tenant-b") is None
    assert store.messages("same-user", conversation_a, "tenant-b") == []
    assert not store.delete_conversation("same-user", conversation_a, "tenant-b")
    assert store.conversation("same-user", conversation_a, "tenant-a") is not None


def test_unattributed_legacy_product_records_are_invisible_to_tenants(tmp_path):
    store = ProductStore(tmp_path / "product.sqlite3")
    collection_id = store.create_collection("owner", "legacy", "", tenant_id="")
    conversation_id = store.create_conversation("owner", "legacy", "lite", tenant_id="")
    assert store.collection("owner", collection_id, "tenant-a") is None
    assert store.collection("owner", collection_id, "tenant-b") is None
    assert store.conversation("owner", conversation_id, "tenant-a") is None
    assert store.conversation("owner", conversation_id, "tenant-b") is None
