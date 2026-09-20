import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import DashScopeEmbeddings

load_dotenv()

BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
API_KEY = os.getenv("DASHSCOPE_API_KEY")

LLM_MODEL = "qwen3.7-plus"
EMBEDDING_MODEL = "text-embedding-v4"
VL_MODEL ="qwen-vl-plus"

VECTOR_DB_PATH = "./chroma_db"
PDF_FOLDER = "./data/pdfs"

def get_llm(temperature=0.1):
    return ChatOpenAI(
        model=LLM_MODEL,
        api_key=API_KEY,
        base_url=BASE_URL,
        temperature=temperature,
        max_retries=3,
        timeout=60,
    )

def get_embedding():
    return  DashScopeEmbeddings(
    model=EMBEDDING_MODEL,
    dashscope_api_key=API_KEY,
    max_retries=3

)

