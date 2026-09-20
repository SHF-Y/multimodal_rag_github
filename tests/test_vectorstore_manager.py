from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

import core.vectorstore_manager as vsm

class TestBuildVectorstore:
    """build_vectorstore 函数测试
    """

    def test_builds_and_persists(self):
        """正常流程：验证"加载 → 分割 → 持久化"整条链路。"""

        mock_loader = MagicMock()
        mock_loader.load.return_value = [
            Document(page_content="第一页内容，介绍锈蚀缺陷的分级标准。"),
            Document(page_content="第二页内容，介绍渗漏油的检测方法。"),
        ]

        mock_vectordb = MagicMock()
        mock_chroma = MagicMock()
        mock_chroma.from_documents.return_value = mock_vectordb

        mock_embedding = MagicMock()

        with patch("core.vectorstore_manager.DirectoryLoader", return_value=mock_loader) as mock_loader_cls, \
             patch("core.vectorstore_manager.Chroma", mock_chroma):

            result = vsm.build_vectorstore(
                pdf_dir="/fake/pdfs",
                persist_dir="/fake/chroma_db",
                embedding=mock_embedding,
            )

        mock_loader_cls.assert_called_once_with(
            "/fake/pdfs", glob="**/*.pdf", loader_cls=vsm.PyPDFLoader
        )

        call_kwargs = mock_chroma.from_documents.call_args.kwargs
        assert len(call_kwargs["documents"]) > 0
        assert call_kwargs["embedding"] is mock_embedding
        assert call_kwargs["persist_directory"] == "/fake/chroma_db"

        assert result is mock_vectordb

class TestLoadOrCreateVectorstore:
    """load_or_create_vectorstore 函数测试
    """

    def test_loads_existing_nonempty_store(self):
        """目录存在且非空、集合 count>0 → 直接返回加载的库，不重建。"""

        mock_db = MagicMock()
        mock_db._collection.count.return_value = 5

        mock_chroma = MagicMock(return_value=mock_db)
        mock_embedding = MagicMock()

        with patch("core.vectorstore_manager.os.path.exists", return_value=True), \
             patch("core.vectorstore_manager.os.listdir", return_value=["chroma.sqlite3"]), \
             patch("core.vectorstore_manager.Chroma", mock_chroma), \
             patch("core.vectorstore_manager.build_vectorstore") as mock_build:
            result = vsm.load_or_create_vectorstore("/fake/db", "/fake/pdfs", mock_embedding)

        mock_chroma.assert_called_once_with(
            persist_directory="/fake/db", embedding_function=mock_embedding
        )

        assert result is mock_db

        mock_build.assert_not_called()

    def test_rebuilds_when_collection_empty(self):
        """目录有内容但集合 count==0 → 判定为空库，走重建分支。"""
        mock_db = MagicMock()
        mock_db._collection.count.return_value = 0
        mock_chroma = MagicMock(return_value=mock_db)
        mock_embedding = MagicMock()

        with patch("core.vectorstore_manager.os.path.exists", return_value=True), \
             patch("core.vectorstore_manager.os.listdir", return_value=["chroma.sqlite3"]), \
             patch("core.vectorstore_manager.Chroma", mock_chroma), \
             patch("core.vectorstore_manager.build_vectorstore") as mock_build:
            result = vsm.load_or_create_vectorstore("/fake/db", "/fake/pdfs", mock_embedding)

        mock_build.assert_called_once_with("/fake/pdfs", "/fake/db", mock_embedding)
        assert result is mock_build.return_value

    def test_rebuilds_when_dir_empty(self):
        """目录存在但 listdir 返回空列表 → 目录"空"也重建。"""
        with patch("core.vectorstore_manager.os.path.exists", return_value=True), \
             patch("core.vectorstore_manager.os.listdir", return_value=[]), \
             patch("core.vectorstore_manager.Chroma") as mock_chroma, \
             patch("core.vectorstore_manager.build_vectorstore") as mock_build:
            result = vsm.load_or_create_vectorstore("/fake/db", "/fake/pdfs", MagicMock())

        mock_chroma.assert_not_called()
        mock_build.assert_called_once()
        assert result is mock_build.return_value

    def test_rebuilds_when_dir_missing(self):
        """目录不存在（exists 返回 False）→ 跳过加载，直接重建。"""
        with patch("core.vectorstore_manager.os.path.exists", return_value=False), \
             patch("core.vectorstore_manager.os.listdir") as mock_listdir, \
             patch("core.vectorstore_manager.build_vectorstore") as mock_build:
            result = vsm.load_or_create_vectorstore("/fake/db", "/fake/pdfs", MagicMock())

        mock_listdir.assert_not_called()
        mock_build.assert_called_once()
        assert result is mock_build.return_value

    def test_rebuilds_when_load_raises(self):
        """加载过程抛异常 → 函数内部捕获后重建。"""
        mock_chroma = MagicMock(side_effect=RuntimeError("corrupted index"))

        with patch("core.vectorstore_manager.os.path.exists", return_value=True), \
             patch("core.vectorstore_manager.os.listdir", return_value=["chroma.sqlite3"]), \
             patch("core.vectorstore_manager.Chroma", mock_chroma), \
             patch("core.vectorstore_manager.build_vectorstore") as mock_build:

            result = vsm.load_or_create_vectorstore("/fake/db", "/fake/pdfs", MagicMock())

        mock_build.assert_called_once()
        assert result is mock_build.return_value

class TestBuildVectorstoreParentChild:
    """build_vectorstore_parent_child 函数测试
    验证："父子关系"是否被正确写进了 metadata。
    """

    def test_parent_child_metadata_and_separate_storage(self):
        """验证两条：
        1) 写入向量库的每个子块都带 parent_id 和 doc_type=child
        2) 父块通过 add_documents 单独入库，带 doc_id 和 doc_type=parent"""
        mock_loader = MagicMock()

        mock_loader.load.return_value = [
            Document(page_content="这是一段用于父子分块测试的工业缺陷知识文本。" * 30)
        ]
        mock_vectordb = MagicMock()
        mock_chroma = MagicMock()
        mock_chroma.from_documents.return_value = mock_vectordb

        with patch("core.vectorstore_manager.DirectoryLoader", return_value=mock_loader), \
             patch("core.vectorstore_manager.Chroma", mock_chroma):
            result = vsm.build_vectorstore_parent_child(
                pdf_dir="/fake/pdfs",
                persist_dir="/fake/db",
                embedding=MagicMock(),

                parent_chunk_size=200,
                parent_overlap=0,
                child_chunk_size=60,
                child_overlap=0,
            )

        assert result is mock_vectordb

        child_docs = mock_chroma.from_documents.call_args.kwargs["documents"]
        assert len(child_docs) > 0
        assert all("parent_id" in d.metadata for d in child_docs)
        assert all(d.metadata["doc_type"] == "child" for d in child_docs)

        parent_docs = mock_vectordb.add_documents.call_args.args[0]
        assert len(parent_docs) > 0
        assert all("doc_id" in d.metadata for d in parent_docs)
        assert all(d.metadata["doc_type"] == "parent" for d in parent_docs)

        child_parent_ids = {d.metadata["parent_id"] for d in child_docs}
        parent_doc_ids = {d.metadata["doc_id"] for d in parent_docs}
        assert child_parent_ids == parent_doc_ids

class TestSearchWithParentChild:
    """search_with_parent_child 函数测试
    """

    def test_returns_unique_parent_docs(self):
        """三个命中的子块中，前两个属于同一个父块 p1 → 结果必须去重为 2 个父块，顺序按子块命中的先后保持。"""

        mock_vs = MagicMock()
        mock_vs.similarity_search.return_value = [
            Document(page_content="子块1", metadata={"parent_id": "p1", "doc_type": "child"}),
            Document(page_content="子块2", metadata={"parent_id": "p1", "doc_type": "child"}),
            Document(page_content="子块3", metadata={"parent_id": "p2", "doc_type": "child"}),
        ]

        parent_pool = {
            "p1": (["父块一：锈蚀分级标准全文"], [{"doc_id": "p1", "doc_type": "parent"}]),
            "p2": (["父块二：渗漏油检测方法全文"], [{"doc_id": "p2", "doc_type": "parent"}]),
        }

        def fake_get(where, include):
            return {

                "documents": parent_pool[where["doc_id"]][0],
                "metadatas": parent_pool[where["doc_id"]][1],
            }

        mock_vs.get.side_effect = fake_get

        result = vsm.search_with_parent_child(mock_vs, "锈蚀", k=5)

        mock_vs.similarity_search.assert_called_once_with(
            "锈蚀", k=5, filter={"doc_type": "child"}
        )

        assert [d.metadata["doc_id"] for d in result] == ["p1", "p2"]

        assert result[0].page_content == "父块一：锈蚀分级标准全文"
        assert result[1].page_content == "父块二：渗漏油检测方法全文"

    def test_skips_child_without_parent_id(self):
        """子块缺失 parent_id 时直接跳过，不参与去重与父块取回。"""
        mock_vs = MagicMock()
        mock_vs.similarity_search.return_value = [
            Document(page_content="孤儿子块", metadata={}),
            Document(page_content="正常子块", metadata={"parent_id": "p9", "doc_type": "child"}),
        ]

        mock_vs.get.return_value = {
            "documents": ["父块九内容"],
            "metadatas": [{"doc_id": "p9", "doc_type": "parent"}],
        }

        result = vsm.search_with_parent_child(mock_vs, "查询", k=3)

        assert len(result) == 1
        assert result[0].page_content == "父块九内容"

        mock_vs.get.assert_called_once_with(
            where={"doc_id": "p9"}, include=["documents", "metadatas"]
        )
