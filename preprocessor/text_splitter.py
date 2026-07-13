from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from document_loader.base_loader import Document
from utils.logger import get_logger

logger = get_logger("text_preprocessor")


class DocumentSplitter:
    """Industrial text splitter optimized for logistics freight documents"""

    def __init__(self):
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,
            chunk_overlap=180,
            separators=["\n\nTABLE_DATA:\n", "\n\n", "\n", " | ", ". ", ", "],
        )
        
    def split_single_document(self, doc: Document) -> List[Document]:
        """Split single raw Document object into small chunks for vector retrieval"""
        chunk_text_list = self._splitter.split_text(doc.content)
        chunk_result = []
        
        for chunk_idx, chunk_text in enumerate(chunk_text_list):
            new_meta = doc.metadata.copy()
            new_meta["chunk_index"] = chunk_idx
            new_meta["chunk_length"] = len(chunk_text)
            
            chunk_doc = Document(
                content=chunk_text,
                source_file=new_meta["source_file"],
                file_type=new_meta["file_type"],
                page_num=new_meta.get("page_num"),
                extra_meta=new_meta
            )
            
            chunk_result.append(chunk_doc)
        
        logger.info(
            f"Split file {doc.metadata['source_file']}, generated {len(chunk_result)} text chunks"
        )
        
        return chunk_result
    
    def batch_split(self, raw_document_list: List[Document]) -> List[Document]:
        """Batch split all loaded raw documents"""
        all_chunks = []
        for raw_doc in raw_document_list:
            chunks = self.split_single_document(raw_doc)
            all_chunks.extend(chunks)

        logger.info(f"Batch split finished, total chunk count: {len(all_chunks)}")
        return all_chunks