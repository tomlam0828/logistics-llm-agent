from typing import List
from langchain_ollama import OllamaRerank
from langchain_core.documents import Document as LangChainDoc
from utils.base_config import app_config
from utils.logger import get_logger

logger = get_logger("reranker")

class FreightReranker:
    def __init__(self):
        self.rerank_model = OllamaRerank(
            model=app_config.RERANK_MODEL,
            base_url=app_config.OLLAMA_BASE_URL,
            top_n=app_config.RERANK_TOP_N
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
            ranked_docs = self.rerank_model.rerank(query, docs)
            logger.info(f"Rerank finished, original {len(docs)} docs -> top {len(ranked_docs)}")
            return ranked_docs
        except Exception as e:
            logger.error(f"Rerank model failed: {str(e)}, return original docs")
            return docs