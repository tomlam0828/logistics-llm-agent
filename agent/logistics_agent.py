from typing import List

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from document_loader.base_loader import Document
from preprocessor.prompt_sanitizer import PromptSecurityFilter
from retriever.hybrid_retriever import LogisticsHybridRetriever
from utils.base_config import app_config
from utils.logger import get_logger
from utils.prompt_loader import load_prompt_template

logger = get_logger("logistics_agent")


class LogisticsRAGAgent:
    """End-to-end RAG agent for freight inquiry"""

    def __init__(self, chunked_docs: List[Document]):
        self.security_filter = PromptSecurityFilter()
        self.retriever = LogisticsHybridRetriever(chunked_docs)
        self.llm = ChatOllama(
            model=app_config.DEBUG_LLM_MODEL, temperature=app_config.LLM_TEMPERATURE
        )
        self.output_parser = StrOutputParser()

        # Load prompt from external file instead of hardcode
        system_prompt = load_prompt_template("system_rag.txt")
        self.no_data_msg = load_prompt_template("no_data_response.txt")

        # Custom system prompt for freight business
        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{user_query}"),
            ]
        )

        # Build RAG chain: retrieve -> fill prompt -> llm generate -> parse text
        self.rag_chain = prompt_template | self.llm | self.output_parser

    def run_chat(self, raw_user_input: str):
        """Full pipeline entrance: security -> retrieve -> llm generate"""
        # Step 1 security check
        safe_text = self.security_filter.full_pipeline_clean(raw_user_input)
        if safe_text is None:
            warn_msg = "Malicious prompt injection detected, request blocked"
            logger.warning(warn_msg)
            return warn_msg

        # Step 2 hybrid retrieval get relevant chunks
        relevant_docs = self.retriever.retrieve_raw(safe_text)
        if not relevant_docs:
            return self.no_data_msg

        # Combine all retrieved chunks into one context string
        context_text = "\n\n".join([doc.page_content for doc in relevant_docs])

        # Step 3 invoke LLM chain
        response = self.rag_chain.invoke(
            {"context": context_text, "user_query": safe_text}
        )
        logger.info("RAG response generated successfully")
        return response
