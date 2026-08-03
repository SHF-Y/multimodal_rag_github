import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import DashScopeEmbeddings

load_dotenv()

# 全局配置
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
API_KEY = os.getenv("DASHSCOPE_API_KEY")
LLM_MODEL = "qwen-turbo"
VL_MODEL = "qwen-vl-plus"
EMBEDDING_MODEL = "text-embedding-v3"
VECTOR_DB_PATH = "./chroma_db"
PDF_FOLDER = "./data/pdfs"

# 初始化模型
def get_llm():
    return ChatOpenAI(
        model=LLM_MODEL,
        api_key=API_KEY,
        base_url=BASE_URL,
        temperature=0.1,
        max_retries=3,        
        timeout=60,           
    )

def get_embedding():
    return  DashScopeEmbeddings(
    model=EMBEDDING_MODEL,
    dashscope_api_key=API_KEY,
    max_retries=3        

)

