# Install required libraries
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os

# Load a lightweight embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Function to Load University Data
def load_university_data():
    file_path = "./university_data .json"
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: {file_path} not found! Please ensure the file exists.")
    
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    
    return data

# Convert Data into Embeddings
def create_vector_store(data):
    if not data:
        raise ValueError("University data is empty or invalid!")

    questions = [entry["question"] for entry in data]
    answers = [entry["answer"] for entry in data]

    # Convert questions to embeddings
    question_embeddings = embedding_model.encode(questions)
    
    # Create FAISS index
    index = faiss.IndexFlatL2(question_embeddings.shape[1])
    index.add(np.array(question_embeddings))
    
    return index, questions, answers

# Find Best Answer
def get_best_answer(query, index, questions, answers):
    query_embedding = embedding_model.encode([query])
    _, idx = index.search(np.array(query_embedding), 1)
    
    best_index = idx[0][0]
    if best_index < 0 or best_index >= len(answers):
        return "Sorry, I couldn't find an answer to your question."
    
    return answers[best_index]

# Initialize FastAPI
app = FastAPI()

# Load data and create vector store
try:
    data = load_university_data()
    index, questions, answers = create_vector_store(data)
except Exception as e:
    print(f"Error initializing chatbot: {e}")
    data, index, questions, answers = None, None, None, None

# Define Request Model
class QueryModel(BaseModel):
    query: str

@app.post("/chat")
def chat(request: QueryModel):
    if not data or not index:
        raise HTTPException(status_code=500, detail="Chatbot data is not properly initialized.")
    
    response = get_best_answer(request.query, index, questions, answers)
    return {"response": response}
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import os

# Load a lightweight embedding model
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
    port = int(os.environ.get("PORT", 8000))  # Use environment variable for port
    uvicorn.run(app, host="0.0.0.0", port=port)

# Run FastAPI Server
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)

