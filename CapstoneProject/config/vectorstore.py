"""
ChromaDB setup for storing feedback embeddings.
"""

import chromadb
from config.settings import CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME


def get_chroma_client():
    """Return a persistent ChromaDB client."""
    CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))


def get_collection():
    """Get or create the feedback embeddings collection."""
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"description": "User feedback embeddings for similarity search"},
    )


def upsert_feedback(collection, ids: list[str], documents: list[str], metadatas: list[dict]):
    """Upsert feedback documents into ChromaDB."""
    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)


def query_similar(collection, query_text: str, n_results: int = 5):
    """Query ChromaDB for similar feedback items."""
    return collection.query(query_texts=[query_text], n_results=n_results)
