from fastapi.testclient import TestClient
from app.main import app

def test_queries():
    client = TestClient(app)

    print("--- Test 1: Standard query without document_name filter ---")
    r1 = client.post("/query", json={"question": "What is the grading breakdown?"})
    print("Status:", r1.status_code)
    data1 = r1.json()
    print("Answer:\n", data1.get("answer"))
    print("Sources:", len(data1.get("sources", [])))

    print("\n--- Test 2: Query with Swagger placeholder ('string') ---")
    r2 = client.post("/query", json={"question": "What is the late submission policy?", "document_name": "string"})
    print("Status:", r2.status_code)
    data2 = r2.json()
    print("Answer:\n", data2.get("answer"))
    print("Sources:", len(data2.get("sources", [])))

if __name__ == "__main__":
    test_queries()
