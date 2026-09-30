import os
import shutil
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.config import settings
from app.pdf_processor import PDFProcessor
from app.vector_store import VectorStore
from app.rag_service import RAGService

# FastAPI Application metadata
app = FastAPI()

# Enable CORS for frontend flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global services
vector_store = VectorStore()
rag_service = RAGService(vector_store=vector_store)


# ----------------- Pydantic Models -----------------
class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        description="The question you want to ask about the uploaded PDF document",
        example="What is the late submission policy and grading criteria?"
    )
    document_name: Optional[str] = Field(
        default=None,
        description="Optional: Filter query to a specific document filename. Leave null or omit to search all uploaded documents.",
        example=None
    )
    top_k: Optional[int] = Field(
        None,
        description="Number of relevant chunks to retrieve (default: 3)",
        ge=1,
        le=10,
        example=3
    )

class SourceSnippet(BaseModel):
    document_name: str
    page_number: int
    similarity_score: float
    snippet: str

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceSnippet]
    model_used: str

class UploadResponse(BaseModel):
    message: str
    filename: str
    file_size_bytes: int
    total_pages_extracted: int
    chunks_indexed: int
    document_status: str

class DocumentSummary(BaseModel):
    document_name: str
    total_chunks: int
    total_pages: int


# ----------------- API Endpoints -----------------

@app.get("/", tags=["General"])
async def root():
    """Root endpoint showing system status and active configuration."""
    api_key_configured = bool(settings.GEMINI_API_KEY and settings.GEMINI_API_KEY != "your_gemini_api_key_here")
    return {
        "status": "online",
        "system": "University Course PDF RAG System",
        "api_key_configured": api_key_configured,
        "models": {
            "generation_model": settings.GEMINI_MODEL,
            "embedding_model": settings.EMBEDDING_MODEL
        },
        "docs_url": "/docs",
        "instructions": "Go to /docs to test PDF upload and question answering." if api_key_configured else "Please configure GEMINI_API_KEY in the .env file."
    }


@app.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["RAG Operations"]
)
async def upload_pdf(
    file: UploadFile = File(..., description="PDF document to upload (e.g. course outline or syllabus)")
):
    """
    Upload a course PDF document.
    - Saves the file to disk.
    - Extracts text page by page.
    - Chunks text into overlapping segments with page metadata.
    - Embeds and indexes the chunks into the vector store.
    """
    if not settings.GEMINI_API_KEY or settings.GEMINI_API_KEY == "your_gemini_api_key_here":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="GEMINI_API_KEY is not configured. Please set your Gemini API key in the .env file."
        )

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format. Only PDF files are supported."
        )

    save_path = settings.UPLOADS_DIR / file.filename

    # Save uploaded file to disk
    try:
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save uploaded file: {str(e)}"
        )
    finally:
        await file.close()

    # Process and chunk PDF
    try:
        chunks = PDFProcessor.chunk_document(save_path)
        if not chunks:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="No readable text could be extracted from the PDF file."
            )

        # Ingest chunks into vector store
        chunks_indexed = vector_store.add_chunks(chunks)

        # Unique pages extracted
        pages_extracted = len(set(c["page_number"] for c in chunks))

        return UploadResponse(
            message="PDF successfully processed, embedded, and indexed!",
            filename=file.filename,
            file_size_bytes=save_path.stat().st_size,
            total_pages_extracted=pages_extracted,
            chunks_indexed=chunks_indexed,
            document_status="Indexed and ready for queries"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error indexing document: {str(e)}"
        )


@app.post(
    "/query",
    response_model=QueryResponse,
    tags=["RAG Operations"]
)
async def query_rag(request: QueryRequest):
    """
    Ask a question about the uploaded PDF document(s).
    - Retrieves top-k most relevant chunks using cosine similarity.
    - Assembles the prompt with retrieved context and page citations.
    - Generates a grounded response using Gemini LLM.
    """
    # Always ensure in-memory store has latest persisted data
    if not vector_store.documents:
        vector_store.load()

    if not vector_store.documents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No documents have been indexed yet. Please upload a PDF file via POST /upload first."
        )

    try:
        response = rag_service.answer_query(
            question=request.question,
            top_k=request.top_k,
            document_filter=request.document_name
        )
        return response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate answer: {str(e)}"
        )


@app.get(
    "/documents",
    response_model=List[DocumentSummary],
    tags=["Document Management"]
)
async def list_documents():
    """List all currently indexed documents in the vector store."""
    return vector_store.list_documents()


@app.delete(
    "/documents",
    tags=["Document Management"]
)
async def clear_documents():
    """Clears all indexed vectors and documents from the store."""
    vector_store.clear()
    return {"message": "Vector store successfully cleared."}
