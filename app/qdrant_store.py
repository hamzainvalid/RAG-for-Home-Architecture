from qdrant_client import QdrantClient

from langchain_qdrant import QdrantVectorStore

from config import (
    QDRANT_URL,
    QDRANT_API_KEY,
    COLLECTION_NAME
)

from embeddings import get_embedding_model


client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)


vector_store = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=get_embedding_model()
)