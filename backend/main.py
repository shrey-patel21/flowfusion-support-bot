from fastapi import FastAPI
from pydantic import BaseModel
from retrieval_qa import qa_chain  
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS middleware — update in production to restrict access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # e.g., ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request body structure
class QueryRequest(BaseModel):
    question: str

@app.get("/")
async def root():
    return {"message": "FlowFusion Support Bot API is live 🚀"}

# Main query endpoint
@app.post("/query")
async def query_api(request: QueryRequest):
    try:
        result = qa_chain.invoke({"query": request.question})
        return {"answer": result["result"]}
    except Exception as e:
        return {"error": str(e)}


