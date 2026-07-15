from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

from retriever.hybrid_retriever import LogisticsHybridRetriever
from preprocessor.prompt_sanitizer import PromptSecurityFilter
from document_loader.base_loader import Document
from utils.base_config import app_config
from utils.logger import get_logger

logger = get_logger("logistics_agent")


class LogisticsRAGAgent:
    """End-to-end RAG agent for freight inquiry"""

    def __init__(self, chunked_docs: List[Document]):
        self.security_filter = PromptSecurityFilter()
        self.retriever = LogisticsHybridRetriever(chunked_docs)
        self.llm = ChatOllama(
            model=app_config.llm_model_name, temperature=app_config.llm_temperature
        )
        self.output_parser = StrOutputParser()

        # Custom system prompt for freight business
        prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
You are a professional freight forwarding assistant.
Answer user questions strictly based on the provided reference freight documents.
Rules:
1. Do not fabricate any shipping prices, port surcharges or weight tiers not in context.
2. If the reference has no matching information, clearly tell the user you cannot find relevant freight data.
3. Keep answers concise, sorted by port, weight and cost clearly.
4. Never output complete confidential price tables in full.
Reference context: {context}
            """,
                ),
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
            return "No matching freight document information found."

        # Combine all retrieved chunks into one context string
        context_text = "\n\n".join([doc.page_content for doc in relevant_docs])

        # Step 3 invoke LLM chain
        response = self.rag_chain.invoke(
            {"context": context_text, "user_query": safe_text}
        )
        logger.info("RAG response generated successfully")
        return response
