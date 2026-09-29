import re
from pathlib import Path
from typing import List, Dict, Any
from pypdf import PdfReader
from app.config import settings

class PDFProcessor:
    """
    Handles PDF text extraction and document chunking with metadata.
    """

    @staticmethod
    def extract_text_from_pdf(pdf_path: Path) -> List[Dict[str, Any]]:
        """
        Extract text from each page of a PDF file.
        Returns a list of dicts: [{'page_number': int, 'text': str}, ...]
        """
        reader = PdfReader(str(pdf_path))
        pages_content = []

        for idx, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            # Clean excessive linebreaks and whitespace
            clean_text = re.sub(r"[ \t]+", " ", text)
            clean_text = re.sub(r"\n\s*\n+", "\n\n", clean_text).strip()
            
            if clean_text:
                pages_content.append({
                    "page_number": idx + 1,
                    "text": clean_text
                })

        return pages_content

    @classmethod
    def chunk_document(
        cls,
        pdf_path: Path,
        chunk_size: int = None,
        chunk_overlap: int = None
    ) -> List[Dict[str, Any]]:
        """
        Splits extracted page text into overlapping chunks with metadata.
        Each chunk contains:
          - chunk_id: unique identifier
          - document_name: original filename
          - page_number: page where text originated
          - text: the chunk content
        """
        chunk_size = chunk_size or settings.CHUNK_SIZE
        chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

        pages_data = cls.extract_text_from_pdf(pdf_path)
        chunks: List[Dict[str, Any]] = []
        chunk_counter = 0

        stride = max(1, chunk_size - chunk_overlap)

        for page in pages_data:
            page_num = page["page_number"]
            page_text = page["text"]
            text_length = len(page_text)

            for start in range(0, text_length, stride):
                end = min(start + chunk_size, text_length)
                
                # Expand slightly to sentence end or whitespace if within range
                if end < text_length:
                    next_space = page_text.find(" ", end)
                    if next_space != -1 and (next_space - end) < 50:
                        end = next_space

                chunk_str = page_text[start:end].strip()
                if len(chunk_str) > 20:  # Ignore tiny stray fragments
                    chunk_counter += 1
                    chunks.append({
                        "chunk_id": f"{pdf_path.name}_p{page_num}_c{chunk_counter}",
                        "document_name": pdf_path.name,
                        "page_number": page_num,
                        "text": chunk_str
                    })

        return chunks
