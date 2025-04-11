# FlowFusion Support Bot (Backend)

This is the backend for FlowFusion’s AI-powered customer support bot. It uses a retrieval-augmented generation (RAG) architecture to answer user queries from a markdown-based knowledge base using Hugging Face models and LangChain.

# Features

- Query answering via LLM using LangChain + HuggingFace
- Vector store backed by FAISS
- Markdown-based knowledge base ingestion
- FastAPI backend with a /query endpoint
- CORS enabled for easy frontend integration
- Easy to run, extend, and deploy

##  Setup Instructions

## ⚙️ Setup Instructions

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/flowfusion-support-bot.git
    cd flowfusion-support-bot
    ```

2. Create a virtual environment and install dependencies:

    ```bash
    python -m venv .venv
    # Activate the virtual environment:
    # On macOS/Linux:
    source .venv/bin/activate
    # On Windows:
    .venv\Scripts\activate

    pip install -r requirements.txt
    ```

3. Add your Hugging Face API key in a `.env` file in the root directory:

    ```env
    HUGGINGFACEHUB_API_TOKEN=your_token_here
    ```

4. Run the FastAPI server:

    ```bash
    cd backend
    uvicorn main:app --reload
    ```

5. Open your browser and go to:  
    [http://localhost:8000/docs](http://localhost:8000/docs) to interact with the API.

## System Architecture
    FastAPI backend
    LangChain + HuggingFace for Retrieval QA
    FAISS vector store
    Sentence Transformers for document embeddings
    LLM via HuggingFace Inference API (e.g., LLaMA-3 or similar)


Prompt Engineering Approach
    I used LangChain’s RetrievalQA chain with a custom prompt:
    Use the following context to answer the user's question. If you don’t know, say you don’t know.
    Context: [retrieved knowledge chunks]
    Question: [user query]


