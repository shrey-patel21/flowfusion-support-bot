import os
from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.vectorstores import FAISS

# Load environment variables
load_dotenv()
hf_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

if not hf_token:
    raise ValueError("Missing Hugging Face API token in .env.")

# Load all markdown files from the ../kb directory
docs = []
kb_dir = "../kb"
for filename in os.listdir(kb_dir):
    if filename.endswith(".md"):
        path = os.path.join(kb_dir, filename)
        loader = TextLoader(path, encoding='utf-8')
        docs.extend(loader.load())

# Split the documents to chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# Embed using Hugging Face Inference API
embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=hf_token,
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Build and save FAISS vectorstore
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("faiss_index")

print(f"Saved {len(chunks)} chunks to FAISS index using Hugging Face embeddings.")
