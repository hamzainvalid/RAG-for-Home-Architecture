import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "gpt-4.1-mini"
CHROMA_DB_PATH = "../chroma_db"
DOCUMENTS_PATH = "../documents"