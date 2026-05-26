import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

COLLECTION_NAME = "architecture_ai"

MODEL_NAME = "gpt-4o-mini"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"