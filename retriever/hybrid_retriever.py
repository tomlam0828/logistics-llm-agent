from typing import List

import chromadb
from langchain.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever

from document_loader.base_loader import Document
from preprocessor.prompt_sanitizer import PromptSecurityFilter
from preprocessor.text_splitter import DocumentSplitter
from utils.base_config import app_config
from utils.logger import get_logger

logger = get_logger("hybrid_retriever")


class LogisticsHybridRetriever:
    """BM25 keyword + Chroma vector + rerank three-stage industrial retriever"""

    def __init__(self, chunked_docs: List[Document]):
        self.security_filter = PromptSecurityFilter()
        self._embedding_model = OllamaEmbeddings(model=app_config.embedding_model_name)
        self._persist_path = app_config.chroma_persist_directory

        # Convert custom Document to LangChain Document for underlying library
        lc_docs = self._convert_to_langchain_docs(chunked_docs)

        # Build vector store
        self._chroma_db = Chroma.from_documents(
            documents=lc_docs,
            embedding=self._embedding_model,
            persist_directory=self._persist_path,
        )
        self._chroma_db.persist()
        self.vector_retriever = self._chroma_db.as_retriever(search_kwargs={"k": 4})

        # Build BM25 keyword retriever
        self.bm25_retriever = BM25Retriever.from_documents(lc_docs)
        self.bm25_retriever.k = 4

    def _convert_to_langchain_docs(self, custom_docs: List[Document]):
        from langchain import Document as LangChainDoc

        output = []
        for doc in custom_docs:
            lc_doc = LangChainDoc(page_content=doc.content, metadata=doc.metadata)
            output.append(lc_doc)
        return output

    def _merge_and_deduplicate(self, vector_results, bm25_results):
        """Merge two retrieval result lists, remove duplicate chunks by source metadata"""
        seen_ids = set()
        merged = []
        for res in vector_results + bm25_results:
            meta_key = f"{res.metadata['source_file']}-{res.metadata['chunk_index']}"
            if meta_key not in seen_ids:
                seen_ids.add(meta_key)
                merged.append(res)
        return merged

    def retrieve_raw(self, user_query: str):
        """Two-stage retrieval without rerank: vector + keyword BM25 fusion"""
        # Security sanitize first
        safe_input = self.security_filter.full_pipeline_clean(user_query)
        if safe_input is None:
            logger.warning("Block retrieval due to prompt injection risk")
            return []

        vector_hits = self.vector_retriever.get_relevant_documents(safe_input)
        bm25_hits = self.bm25_retriever.get_relevant_documents(safe_input)
        combined = self._merge_and_deduplicate(vector_hits, bm25_hits)
        logger.info(
            f"Vector hits: {len(vector_hits)}, BM25 hits: {len(bm25_hits)}, unique merged chunks: {len(combined)}"
        )
        return combined
