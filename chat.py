import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load University Data
def load_university_data():
    with open("./university_data .json", "r", encoding="utf-8") as file:
        data = json.load(file)
    return data

# Convert Data into Embeddings
def create_vector_store(data):
    questions = [entry["question"] for entry in data]
    answers = [entry["answer"] for entry in data]
    question_embeddings = embedding_model.encode(questions)
    
    index = faiss.IndexFlatL2(question_embeddings.shape[1])
    index.add(np.array(question_embeddings))
    return index, questions, answers

# Find Best Answer
def get_best_answer(query, index, questions, answers):
    query_embedding = embedding_model.encode([query])
    _, idx = index.search(np.array(query_embedding), 1)
    return answers[idx[0][0]]

# Initialize FastAPI
app = FastAPI()

data = load_university_data()
index, questions, answers = create_vector_store(data)

# Request Model
class QueryModel(BaseModel):
    query: str

@app.post("/chat")
def chat(request: QueryModel):
    response = get_best_answer(request.query, index, questions, answers)
    return {"response": response}

# Run FastAPI Server
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Get Render's assigned port
    uvicorn.run(app, host="0.0.0.0", port=port)
