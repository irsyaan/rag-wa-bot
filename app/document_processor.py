"""
Document processor for handling PDF uploads and chunking.
"""

import os
from PyPDF2 import PdfReader
from loguru import logger

from app.ollama_client import ollama_client
from app.qdrant_store import qdrant_store
from app.mysql_store import mysql_store
from app.config import settings

class DocumentProcessor:
    def __init__(self):
        self.chunk_size = 1000
        self.chunk_overlap = 100
        self.collection = settings.qdrant_knowledge_collection

    def _chunk_text(self, text: str) -> list[str]:
        """Split text into overlapping chunks."""
        words = text.split()
        chunks = []
        i = 0
        while i < len(words):
            chunk = " ".join(words[i:i + self.chunk_size])
            if chunk.strip():
                chunks.append(chunk)
            i += self.chunk_size - self.chunk_overlap
        return chunks

    def process_pdf(self, file_path: str, filename: str, uploader_number: str) -> tuple[bool, str]:
        """Process a PDF and ingest it into Qdrant."""
        try:
            user = mysql_store.get_user(uploader_number)
            uploader_id = user["id"] if user else None

            # Add document record to MySQL
            doc_id = mysql_store.add_document(
                filename=filename,
                file_path=file_path,
                file_type="pdf",
                uploaded_by=uploader_id,
            )
            
            if not doc_id:
                return False, "Failed to register document in database."

            # Update status
            mysql_store.update_document_status(doc_id, "processing")

            # Read PDF
            reader = PdfReader(file_path)
            full_text = ""
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n"
            
            if not full_text.strip():
                mysql_store.update_document_status(doc_id, "error")
                return False, "PDF appears to be empty or unreadable."

            # Chunk text
            chunks = self._chunk_text(full_text)
            
            logger.info(f"Chunked {filename} into {len(chunks)} parts.")

            # Embed and save
            for i, chunk_text in enumerate(chunks):
                vector = ollama_client.embed(chunk_text)
                if not vector:
                    continue
                
                payload = {
                    "document_id": doc_id,
                    "filename": filename,
                    "text": chunk_text,
                    "source": filename
                }

                # Save to Qdrant
                point_id = qdrant_store.add_point(self.collection, vector, payload)
                
                # Save chunk to MySQL
                mysql_store.add_chunk(
                    document_id=doc_id,
                    chunk_index=i,
                    chunk_text=chunk_text,
                    qdrant_point_id=point_id,
                    collection_name=self.collection
                )

            # Mark as done
            mysql_store.update_document_status(doc_id, "done", len(chunks))
            
            # Clean up uploaded file
            try:
                os.remove(file_path)
            except OSError as e:
                logger.warning(f"Failed to remove temp file {file_path}: {e}")

            return True, f"Successfully processed '{filename}' into {len(chunks)} knowledge chunks!"

        except Exception as e:
            logger.exception(f"Error processing PDF {filename}: {e}")
            if 'doc_id' in locals() and doc_id:
                mysql_store.update_document_status(doc_id, "error")
            return False, f"Failed to process PDF: {str(e)}"

# Singleton
document_processor = DocumentProcessor()
