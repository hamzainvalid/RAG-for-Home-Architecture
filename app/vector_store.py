# vector_store.py

import os
from uuid import uuid4

from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader
)

from langchain_community.embeddings import (
    HuggingFaceEmbeddings
)

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from langchain_qdrant import QdrantVectorStore


# =========================================================
# LOAD ENV VARIABLES
# =========================================================

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

COLLECTION_NAME = "architecture_ai"

DOCUMENTS_PATH = "../documents"


# =========================================================
# EMBEDDING MODEL
# =========================================================

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# =========================================================
# LOAD DOCUMENTS
# =========================================================

print("Loading PDF documents...")

loader = DirectoryLoader(
    DOCUMENTS_PATH,
    glob="**/*.pdf",
    loader_cls=PyPDFLoader
)

documents = loader.load()

print(f"Loaded {len(documents)} pages")


# =========================================================
# SPLIT DOCUMENTS
# =========================================================

print("Splitting documents into chunks...")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks")


# =========================================================
# ADD METADATA
# =========================================================

for chunk in chunks:

    source = chunk.metadata.get("source", "").lower()

    chunk.metadata["category"] = "general"

    if "living" in source:
        chunk.metadata["room_type"] = "living_room"

    if "kitchen" in source:
        chunk.metadata["room_type"] = "kitchen"

    if "sofa" in source:
        chunk.metadata["object_type"] = "sofa"

    if "lighting" in source:
        chunk.metadata["object_type"] = "lighting"


# =========================================================
# CONNECT TO QDRANT
# =========================================================

print("Connecting to Qdrant...")

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)


# =========================================================
# CREATE COLLECTION IF NOT EXISTS
# =========================================================

collections = client.get_collections().collections
collection_names = [c.name for c in collections]

if COLLECTION_NAME not in collection_names:

    print("Creating collection...")

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=384,
            distance=Distance.COSINE
        )
    )

    print("Collection created")

else:
    print("Collection already exists")


# =========================================================
# CREATE VECTOR STORE
# =========================================================

print("Uploading embeddings to Qdrant...")

vector_store = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embedding_model,
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    collection_name=COLLECTION_NAME,
    ids=[str(uuid4()) for _ in chunks]
)

print("Upload complete")
print("Vector database ready")