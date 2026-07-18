from typing import List

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_ollama import OllamaEmbeddings

from document_loader.base_loader import Document
from preprocessor.prompt_sanitizer import PromptSecurityFilter
from preprocessor.text_splitter import DocumentSplitter
from retriever.reranker import FreightReranker
from utils.base_config import app_config
from utils.logger import get_logger

logger = get_logger("hybrid_retriever")


class LogisticsHybridRetriever:
    """BM25 keyword + Chroma vector + rerank three-stage industrial retriever"""

    def __init__(self, chunked_docs: List[Document]):
        self.security_filter = PromptSecurityFilter()
        self._embedding_model = OllamaEmbeddings(
            model=app_config.EMBEDDING_MODEL, base_url=app_config.OLLAMA_BASE_URL
        )
        self._persist_path = app_config.CHROMA_STORAGE_PATH

        # Convert custom Document to LangChain Document for underlying library
        lc_docs = self._convert_to_langchain_docs(chunked_docs)

        # Build vector store
        self._chroma_db = Chroma.from_documents(
            documents=lc_docs,
            embedding=self._embedding_model,
            persist_directory=self._persist_path,
        )
        self.vector_retriever = self._chroma_db.as_retriever(search_kwargs={"k": 4})

        # Build BM25 keyword retriever
        self.bm25_retriever = BM25Retriever.from_documents(lc_docs)
        self.bm25_retriever.k = 4

        self.reranker = FreightReranker()

    def _convert_to_langchain_docs(self, custom_docs):
        from langchain_core.documents import Document as LangChainDoc

        lc_docs = []
        for d in custom_docs:
            full_meta = d.metadata.copy()
            lc_doc = LangChainDoc(page_content=d.content, metadata=full_meta)
            lc_docs.append(lc_doc)
        return lc_docs

    def _merge_and_deduplicate(self, vector_hits, bm25_hits):
        all_docs = vector_hits + bm25_hits
        unique_map = {}
        for res in all_docs:
            source_file = res.metadata.get(
                "source_file", res.metadata.get("source", "unknown")
            )
            chunk_idx = res.metadata.get("chunk_index", 0)
            meta_key = f"{source_file}-{chunk_idx}"
            if meta_key not in unique_map:
                unique_map[meta_key] = res
        return list(unique_map.values())

    def retrieve_raw(self, user_query: str):
        """Two-stage retrieval without rerank: vector + keyword BM25 fusion"""
        # Security sanitize first
        safe_input = self.security_filter.full_pipeline_clean(user_query)
        if safe_input is None:
            logger.warning("Block retrieval due to prompt injection risk")
            return []

        vector_hits = self.vector_retriever.invoke(safe_input)
        bm25_hits = self.bm25_retriever.invoke(safe_input)
        combined = self._merge_and_deduplicate(vector_hits, bm25_hits)

        logger.info(
            f"Vector hits: {len(vector_hits)}, BM25 hits: {len(bm25_hits)}, unique merged chunks: {len(combined)}"
        )

        # Rerank to filter irrelevant chunks
        final_docs = self.reranker.rerank_docs(user_query, combined)
        logger.info(f"After rerank, reserved top {len(final_docs)} relevant chunks")

        return final_docs
