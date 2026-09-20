'''
判断向量库是否可用（目录存在、非空、能成功加载且有数据）。
构建向量库：从 PDF 目录读取所有 PDF，分割后写入 Chroma。
'''
import os
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

def build_vectorstore(pdf_dir: str, persist_dir: str, embedding):
    """从 PDF 目录构建向量库，并持久化到 persist_dir。
    """
    print(f"正在从 {pdf_dir} 加载 PDF 文件...")
    loader = DirectoryLoader(
        pdf_dir,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )
    docs = loader.load()
    print(f"共加载 {len(docs)} 个文档页面。")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        separators=["\n\n", "\n", "。", "；", "，", " ", ""]

        )
    

    splits = text_splitter.split_documents(docs)
    print(f"分割后得到 {len(splits)} 个文本块。")

    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embedding,
        persist_directory=persist_dir
    )

    print("向量库构建完成并已持久化。")
    return vectordb

def load_or_create_vectorstore(persist_dir: str, pdf_dir: str, embedding):
    """
    加载已有向量库；若不存在或为空，则从 PDF创建。
    返回一个 Chroma 向量库对象。
    """

    if os.path.exists(persist_dir) and os.listdir(persist_dir):

        try:
            vectordb = Chroma(
                persist_directory=persist_dir,
                embedding_function=embedding

            )

            if vectordb._collection.count() > 0:
                print("向量库已存在，直接加载。")
                return vectordb
            else:
                print("向量库为空，将重新构建。")
        except Exception as e:
            print(f"加载向量库失败: {e}，将重新构建。")

    print("向量库不存在或无效，开始自动构建...")
    return build_vectorstore(pdf_dir, persist_dir, embedding)

from langchain_text_splitters import RecursiveCharacterTextSplitter
import uuid

def build_vectorstore_parent_child(pdf_dir: str, persist_dir: str, embedding,
                                     parent_chunk_size: int = 1200,
                                     parent_overlap: int = 200,
                                     child_chunk_size: int = 300,
                                     child_overlap: int = 50):
    """父子分块构建向量库"""

    from langchain_core.documents import Document

    loader = DirectoryLoader(pdf_dir, glob="**/*.pdf", loader_cls=PyPDFLoader)

    docs = loader.load()

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_chunk_size,
        chunk_overlap=parent_overlap,
        separators=["\n\n", "\n", "。", "；", "，", " ", ""]
    )
    parent_docs = parent_splitter.split_documents(docs)

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_chunk_size,
        chunk_overlap=child_overlap,
        separators=["\n\n", "\n", "。", "；", "，", " ", ""]
    )
    child_docs = []
    for parent_doc in parent_docs:
        parent_id = str(uuid.uuid4())

        parent_doc.metadata["doc_id"] = parent_id
        parent_doc.metadata["doc_type"] = "parent"

        children = child_splitter.split_documents([parent_doc])
        for child in children:
            child.metadata["parent_id"] = parent_id
            child.metadata["doc_type"] = "child"
            child_docs.append(child)

    print(f"父块数量: {len(parent_docs)}, 子块数量: {len(child_docs)}")

    vectordb = Chroma.from_documents(
        documents=child_docs,
        embedding=embedding,
        persist_directory=persist_dir
    )

    vectordb.add_documents(parent_docs)

    return vectordb
def search_with_parent_child(vectordb, query: str, k: int = 5) -> list:
    """
    父子分块检索：检索子块，返回对应的父块（去重）
    """

    child_results = vectordb.similarity_search(
        query, k=k,
        filter={"doc_type": "child"}
    )

    parent_ids = []
    seen = set()
    for doc in child_results:
        pid = doc.metadata.get("parent_id")
        if pid and pid not in seen:
            seen.add(pid)
            parent_ids.append(pid)

    parent_results = []
    for pid in parent_ids:
        parents = vectordb.get(
            where={"doc_id": pid},
            include=["documents", "metadatas"]
        )
        if parents["documents"]:
            parent_results.append(Document(
                page_content=parents["documents"][0],
                metadata=parents["metadatas"][0]
            ))

    return parent_results