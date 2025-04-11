import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA


# Load environment variables
load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Load vectorstore
vectorstore = FAISS.load_local(
    "faiss_index",
    HuggingFaceInferenceAPIEmbeddings(
        api_key=hf_token,
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    ),
    allow_dangerous_deserialization=True
)
from langchain_community.llms import HuggingFaceEndpoint

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.1",  
    task="text-generation",
    huggingfacehub_api_token=hf_token,
    temperature=0.5,
    max_new_tokens=512
)


# Setup Retrieval QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

# Sample query
query = "How can I integrate FlowFusion with Slack?"
result = qa_chain({"query": query})
print(result["result"])
