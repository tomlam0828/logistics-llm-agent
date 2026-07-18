from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from document_loader.base_loader import Document
from utils.logger import get_logger

logger = get_logger("text_preprocessor")


class DocumentSplitter:
    """Industrial text splitter optimized for logistics freight business documents"""

    def __init__(self):
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,
            chunk_overlap=180,
            separators=["\n\nTABLE_DATA:\n", "\n\n", "\n", " | ", ". ", ", "],
        )

    def split_single_document(self, doc: Document) -> List[Document]:
        """Split one raw custom Document object into small retrieval chunks"""
        chunk_text_list = self._splitter.split_text(doc.content)
        chunk_result = []

        for chunk_idx, chunk_text in enumerate(chunk_text_list):
            # Only attach chunk-specific metadata to avoid duplicated meta nesting
            chunk_extra_meta = {
                "chunk_index": chunk_idx,
                "chunk_length": len(chunk_text),
            }

            chunk_doc = Document(
                content=chunk_text,
                source_file=doc.metadata["source_file"],
                file_type=doc.metadata["file_type"],
                page_num=doc.metadata.get("page_num"),
                extra_meta=chunk_extra_meta,
            )
            chunk_result.append(chunk_doc)

        logger.info(
            f"Split source file {doc.metadata['source_file']}, total generated chunks: {len(chunk_result)}"
        )
        return chunk_result

    def batch_split(self, raw_document_list: List[Document]) -> List[Document]:
        """Batch split all loaded raw documents into segmented chunks"""
        all_chunk_list = []
        for raw_doc in raw_document_list:
            split_chunks = self.split_single_document(raw_doc)
            all_chunk_list.extend(split_chunks)
        logger.info(
            f"Batch document split finished, total chunk amount: {len(all_chunk_list)}"
        )
        return all_chunk_list
