import os
from pathlib import Path
from typing import Any, Dict, List

import pytesseract
from openpyxl import load_workbook
from pdfplumber import open as pdfplumber_open
from PIL import Image
from pypdf import PdfReader

from utils.base_config import app_config
from utils.logger import get_logger

logger = get_logger("document_loader")


class Document:
    """Standard document data structure to store raw content and metadata"""

    def __init__(
        self,
        content: str,
        source_file: str,
        file_type: str,
        page_num: int = None,
        extra_meta: Dict[str, Any] = None,
    ):
        self.content = content
        self.metadata = {
            "source_file": source_file,
            "file_type": file_type,
            "page_num": page_num,
            **(extra_meta or {}),
        }


class BaseDocumentLoader:
    """Base loader for multi-format freight business documents"""

    def __init__(self):
        self.test_doc_dir = Path(app_config.TEST_DOCUMENT_FOLDER)
        os.makedirs(self.test_doc_dir, exist_ok=True)

    def load_pdf(self, file_path: str) -> List[Document]:
        """Parse PDF file, support text extraction and table capture"""
        docs = []
        try:
            with pdfplumber_open(file_path) as pdf:
                for page_idx, page in enumerate(pdf.pages, start=1):
                    page_text = page.extract_text() or ""
                    page_tables = page.extract_tables()
                    table_text = ""

                    for table in page_tables:
                        for row in table:
                            table_text += (
                                " | ".join([str(cell) for cell in row if cell]) + "\n"
                            )

                    full_content = page_text + "\nTABLE_DATA:\n" + table_text

                    doc = Document(
                        content=full_content,
                        source_file=os.path.basename(file_path),
                        file_type="pdf",
                        page_num=page_idx,
                        extra_meta={"file_path": file_path},
                    )
                    docs.append(doc)
                    logger.info(
                        f"Successfully loaded PDF: {file_path}, total pages: {len(docs)}"
                    )
        except Exception as e:
            logger.error(f"Failed to load PDF {file_path}: {str(e)}")
        return docs

    def load_image_ocr(self, file_path: str) -> List[Document]:
        """Extract text from shipment image files via OCR"""
        try:
            img = Image.open(file_path)
            ocr_text = pytesseract.image_to_string(img)
            doc = Document(
                content=ocr_text,
                source_file=os.path.basename(file_path),
                file_type="image",
                extra_meta={"file_path": file_path},
            )
            logger.info(f"Successfully OCR image: {file_path}")
            return [doc]
        except Exception as e:
            logger.error(f"Failed to process image OCR {file_path}: {str(e)}")
            return []

    def load_excel(self, file_path: str) -> List[Document]:
        """Parse freight price excel sheets"""
        docs = []
        try:
            workbook = load_workbook(file_path, read_only=True)
            for sheet_name in workbook.sheetnames:
                sheet = workbook[sheet_name]
                sheet_content = f"SHEET_NAME: {sheet_name}\n"
                for row in sheet.iter_rows(values_only=True):
                    row_text = " | ".join(
                        [str(cell) for cell in row if cell is not None]
                    )
                    sheet_content += row_text + "\n"

                doc = Document(
                    content=sheet_content,
                    source_file=os.path.basename(file_path),
                    file_type="excel",
                    extra_meta={"sheet_name": sheet_name, "file_path": file_path},
                )
                docs.append(doc)
            logger.info(f"Successfully loaded Excel: {file_path}, sheets: {len(docs)}")
        except Exception as e:
            logger.error(f"Failed to load Excel {file_path}: {str(e)}")
        return docs

    def load_file(self, file_path: str) -> List[Document]:
        """Dispatch loader based on file extension"""
        ext = Path(file_path).suffix.lower()

        if ext == ".pdf":
            return self.load_pdf(file_path)
        elif ext in [".png", ".jpg", ".jpeg", ".webp"]:
            return self.load_image_ocr(file_path)
        elif ext in [".xlsx", ".xls"]:
            return self.load_excel(file_path)
        else:
            logger.warning(f"Unsupported file type: {ext}, skip {file_path}")
            return []

    def batch_load_folder(self, folder_path: str = None) -> List[Document]:
        """Batch load all supported documents in target folder"""
        target_dir = Path(folder_path) if folder_path else self.test_doc_dir
        all_docs = []

        if not target_dir.exists():
            logger.warning(f"Folder {target_dir} not found, return empty list")
            return all_docs

        for file in target_dir.iterdir():
            if file.is_file():
                file_docs = self.load_file(str(file))
                all_docs.extend(file_docs)
        logger.info(f"Batch load complete, total raw documents: {len(all_docs)}")
        return all_docs
