"""
Top-level entrypoint for the FastAPI RAG application.
Allows running with either:
  uvicorn main:app --reload
or:
  python main.py
"""

from app.main import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
