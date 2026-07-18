from langchain.tools import tool

from agent.logistics_agent import LogisticsRAGAgent
from document_loader.base_loader import BaseDocumentLoader, Document
from retriever.hybrid_retriever import LogisticsHybridRetriever
from utils.logger import get_logger

logger = get_logger("main")

global_rag_agent = None


@tool
def freight_document_search(query: str) -> str:
    """
    Search freight price, port, customs policy from local shipping documents.
    Only use this tool when user asks about shipping cost, port surcharge, weight tier, logistics rules.
    Args:
        query: User's shipping related question in English
    Returns:
        Relevant freight reference text
    """
    global global_rag_agent
    docs = global_rag_agent.retriever.retrieve_raw(query)
    if not docs:
        return "No matching freight information found."
    return "\n\n".join([doc.page_content for doc in docs])


def init_rag_system():
    global global_rag_agent
    if global_rag_agent is not None:
        return global_rag_agent

    loader = BaseDocumentLoader()
    raw_docs: list[Document] = loader.batch_load_folder("./data")
    logger.info(f"Loaded total {len(raw_docs)} document chunks")

    global_rag_agent = LogisticsRAGAgent(raw_docs)
    logger.info("RAG system initialization complete")
    return global_rag_agent


def chat_loop():
    agent = init_rag_system()
    print("===== Freight RAG Chat System =====")
    print("Input your shipping inquiry, type exit to quit\n")

    while True:
        user_input = input("You: ")
        if user_input.strip().lower() == "exit":
            print("System exit")
            break
        resp = agent.run_chat(user_input)
        print(f"Assistant: {resp}\n")


if __name__ == "__main__":
    chat_loop()
