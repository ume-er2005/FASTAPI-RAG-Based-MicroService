from typing import List, Dict, Any, Optional
from google import genai
from app.config import settings
from app.vector_store import VectorStore

class RAGService:
    """
    RAG Orchestration Service:
    1. Retrieves relevant document chunks.
    2. Builds an augmented prompt with context & page citations.
    3. Queries Gemini to generate grounded, accurate answers.
    """

    def __init__(self, vector_store: Optional[VectorStore] = None):
        self.vector_store = vector_store or VectorStore()

    def _get_client(self) -> genai.Client:
        """Initialize Google GenAI client."""
        if not settings.GEMINI_API_KEY:
            raise ValueError(
                "GEMINI_API_KEY is not set. Please add your Gemini API key in the .env file."
            )
        return genai.Client(api_key=settings.GEMINI_API_KEY)

    def answer_query(
        self,
        question: str,
        top_k: Optional[int] = None,
        document_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes full RAG workflow for a given question.
        """
        # Step 1: Retrieval
        relevant_chunks = self.vector_store.search(
            query=question,
            top_k=top_k,
            document_filter=document_filter
        )

        if not relevant_chunks:
            return {
                "question": question,
                "answer": "No relevant documents or information found. Please ensure you have uploaded a PDF document first.",
                "sources": [],
                "model_used": settings.GEMINI_MODEL
            }

        # Step 2: Context Construction
        context_parts = []
        for i, chunk in enumerate(relevant_chunks, 1):
            doc = chunk["document_name"]
            page = chunk["page_number"]
            score = chunk["similarity_score"]
            text = chunk["text"]
            context_parts.append(
                f"[Source {i}] Document: {doc} | Page: {page} | Relevance: {score:.2f}\n{text}"
            )

        context_str = "\n\n---\n\n".join(context_parts)

        # Step 3: Prompt Engineering
        prompt = f"""You are a helpful and precise university academic assistant. Answer the user's question based strictly on the provided course document context below.

Rules:
1. Ground your answer ONLY in the provided context snippets.
2. If the answer cannot be determined from the context, clearly state that the document does not contain this information. Do not hallucinate or make up details.
3. Cite the document name and page number(s) whenever you reference specific facts or policies (e.g., "[Document: syllabus.pdf, Page: 2]").
4. Keep the explanation clear, professional, and well-structured.

--- PROVIDED CONTEXT ---
{context_str}
--- END OF CONTEXT ---

User Question: {question}

Answer:"""

        # Step 4: Generation via Gemini with graceful fallback
        client = self._get_client()
        candidate_models = [settings.GEMINI_MODEL, "gemini-3.5-flash", "gemini-3.1-flash-lite"]
        # Ensure unique models while preserving order
        models_to_try = list(dict.fromkeys(candidate_models))

        answer_text = None
        last_error = None
        model_used = settings.GEMINI_MODEL

        for model_name in models_to_try:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )
                if response and response.text:
                    answer_text = response.text
                    model_used = model_name
                    break
            except Exception as e:
                last_error = e
                continue

        if answer_text is None:
            answer_text = f"Error during generation: {str(last_error)}"

        return {
            "question": question,
            "answer": answer_text,
            "sources": [
                {
                    "document_name": chunk["document_name"],
                    "page_number": chunk["page_number"],
                    "similarity_score": round(chunk["similarity_score"], 4),
                    "snippet": chunk["text"][:250] + ("..." if len(chunk["text"]) > 250 else "")
                }
                for chunk in relevant_chunks
            ],
            "model_used": model_used
        }
