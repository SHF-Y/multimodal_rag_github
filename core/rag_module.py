from .config import get_embedding, VECTOR_DB_PATH, PDF_FOLDER
from .vectorstore_manager import load_or_create_vectorstore
from .hybrid_retriever import HybridRetriever

_all_docs = None

_hybrid_retriever = None

def init_vector_store():
    embedding=get_embedding()
    return load_or_create_vectorstore(VECTOR_DB_PATH, PDF_FOLDER,embedding)

def _get_all_docs():
    """获取所有文档（用于BM25索引构建）"""
    global _all_docs
    if _all_docs is None:
        vectorstore = init_vector_store()

        results = vectorstore._collection.get(include=["documents", "metadatas"])
        from langchain_core.documents import Document
        _all_docs = [
            Document(page_content=doc, metadata=meta or {})
            for doc, meta in zip(results["documents"], results["metadatas"])
        ]
    return _all_docs

def get_retriever(k=5):
    """获取混合检索器（单例缓存，避免每次请求重建BM25索引）"""
    global _hybrid_retriever
    if _hybrid_retriever is None:
        vectorstore = init_vector_store()
        vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
        all_docs = _get_all_docs()
        _hybrid_retriever = HybridRetriever(
            vector_retriever=vector_retriever,
            docs=all_docs,
            bm25_top_k=10,
            vector_top_k=10,
            final_top_k=k
        )
    else:

        _hybrid_retriever.final_top_k = k
    return _hybrid_retriever

def format_docs(docs):
    return "\n\n".join([f"[文档{i+1}]: {doc.page_content}" for i, doc in enumerate(docs)])