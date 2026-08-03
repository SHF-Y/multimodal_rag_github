from .config import get_embedding, VECTOR_DB_PATH, PDF_FOLDER
from .vectorstore_manager import  load_or_create_vectorstore
def init_vector_store():#初始化向量
    embedding=get_embedding()
    return load_or_create_vectorstore(VECTOR_DB_PATH, PDF_FOLDER,embedding)

def get_retriever(k=3):
    vectorstore =init_vector_store()
    return vectorstore.as_retriever(search_kwargs={"k": k})
def format_docs(docs):
    return "\n\n".join([f"[文档{i+1}]: {doc.page_content}" for i, doc in enumerate(docs)])