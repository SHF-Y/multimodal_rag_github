
import os
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

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

    # 文本分割
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        separators=["\n\n", "\n", "。", "；", "，", " ", ""]
        
        )
    

    splits = text_splitter.split_documents(docs)
    print(f"分割后得到 {len(splits)} 个文本块。")

    # 创建并持久化向量库
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
    # 检查向量数据库目录是否存在且非空
    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        
        try:
            vectordb = Chroma(
                persist_directory=persist_dir,
                embedding_function=embedding
                
            )
            # 进一步检查集合中是否有数据
            if vectordb._collection.count() > 0:
                print("向量库已存在，直接加载。")
                return vectordb
            else:
                print("向量库为空，将重新构建。")
        except Exception as e:
            print(f"加载向量库失败: {e}，将重新构建。")

    # 目录不存在/为空/加载失败 → 重建
    print("向量库不存在或无效，开始自动构建...")
    return build_vectorstore(pdf_dir, persist_dir, embedding)