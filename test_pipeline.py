from pathlib import Path
from app.pdf_processor import PDFProcessor
from app.vector_store import VectorStore
from app.rag_service import RAGService

def main():
    pdf_path = Path("CS402_Artificial_Intelligence_Syllabus.pdf")
    print("1. Chunking PDF...")
    chunks = PDFProcessor.chunk_document(pdf_path)
    print(f"   Extracted {len(chunks)} chunks.")

    print("2. Indexing into VectorStore...")
    store = VectorStore()
    store.clear()
    count = store.add_chunks(chunks)
    print(f"   Indexed {count} chunks. Summary: {store.list_documents()}")

    print("3. Querying RAG Pipeline...")
    rag = RAGService(vector_store=store)
    question = "What is the late submission policy and what are slip days?"
    response = rag.answer_query(question)

    print("\n" + "="*50)
    print(f"QUESTION: {response['question']}")
    print("="*50)
    print(f"ANSWER:\n{response['answer']}")
    print("="*50)
    print("SOURCES:")
    for s in response["sources"]:
        print(f"• {s['document_name']} [Page {s['page_number']}] (Score: {s['similarity_score']})")

if __name__ == "__main__":
    main()
