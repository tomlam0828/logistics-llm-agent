from typing import List

from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_core.documents import Document as LangChainDoc
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from utils.base_config import app_config
from utils.logger import get_logger

logger = get_logger("reranker")


class FreightReranker:
    def __init__(self):
        self.cross_encoder = HuggingFaceCrossEncoder(model_name=app_config.RERANK_MODEL)
        self.compressor = CrossEncoderReranker(
            model=self.cross_encoder, top_n=app_config.RERANK_TOP_N
        )

    def rerank_docs(self, query: str, docs: List[LangChainDoc]) -> List[LangChainDoc]:
        """
        Rerank retrieved documents by relevance score
        :param query: user input question
        :param docs: raw mixed vector + bm25 documents
        :return: sorted high-relevance document list
        """
        if not docs:
            logger.warning("Empty document list, skip rerank")
            return []
        try:
            ranked_docs = self.compressor.compress_documents(docs, query)
            logger.info(
                f"CrossEncoder rerank completed: original {len(docs)} candidates, reserved top {len(ranked_docs)} relevant chunks"
            )
            return ranked_docs
        except Exception as e:
            logger.error(f"CrossEncoder rerank runtime error: {str(e)}")
            return docs
