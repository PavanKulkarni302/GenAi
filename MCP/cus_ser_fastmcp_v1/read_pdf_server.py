from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# -------------------------------------------------------
#  MCP Server Setup
# -------------------------------------------------------
mcp = FastMCP(name="rag_mcp_server")

print("Loading environment variables...")
load_dotenv()

PDF_PATH = "home_depot_policy.pdf"     # << CHANGE PATH HERE
PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "rag_collection"
# -------------------------------------------------------
#  Load PDF and Build Vector DB ONCE (Fast)
# -------------------------------------------------------
def initialize_vector_db():

    # If database already exists — LOAD it
    if os.path.exists(PERSIST_DIR):
        print("Loading existing Chroma DB...")
        embeddings = HuggingFaceEmbeddings()
        return Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME
        )

    # Otherwise create new DB
    print("Creating new Chroma DB...")

    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"PDF file not found at: {PDF_PATH}")

    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()
    print(f"Loaded {len(docs)} pages from PDF")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} text chunks")

    embeddings = HuggingFaceEmbeddings()

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
        collection_name=COLLECTION_NAME
    )

    print("Vector DB created & persisted.")
    return vectordb


# Build vector DB at startup
vectorstore = initialize_vector_db()


# -------------------------------------------------------
#  MCP Tool: query_pdf
# -------------------------------------------------------
@mcp.tool()
def query_pdf(query: str) -> list:
    """
    Answer the user’s question based solely on the available internal knowledge. 
    Do not mention PDFs, documents, pages, chunks, retrieval processes,
    or any technical details about where the information comes from. Provide clear,
    helpful answers only. If information is not available, respond politely without referencing missing documents or sources.
    if you don't know what to answer then just say "I'm sorry, I don't have that information available." or "I don't have access to that information at the moment.
    or "I'm unable to provide that information right now. or ask me anything related to home depot policies."
    """
    results = vectorstore.similarity_search(query, k=3)
    return [doc.page_content for doc in results]


# -------------------------------------------------------
#  Run MCP Server
# -------------------------------------------------------
if __name__ == "__main__":
    mcp.run(transport="stdio")
