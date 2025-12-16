from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
import os
import requests
from bs4 import BeautifulSoup
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# -------------------------------------------------------
#  MCP Server Setup
# -------------------------------------------------------
mcp = FastMCP(name="website_rag_server")

load_dotenv()

WEBSITE_URLS = [
    "https://www.homedepot.com/c/customer_service",
    "https://www.homedepot.com/c/Return_Policy",
    "https://www.homedepot.com/c/refund-policy",
    "https://www.homedepot.com/c/tool-and-equipment-rental"
]  # Add all FAQ / policy pages here

PERSIST_DIR = "./chroma_web_db"
COLLECTION_NAME = "website_rag"


# -------------------------------------------------------
#  Function: Fetch website & extract text
# -------------------------------------------------------
def fetch_and_clean(url):
    print(f"Fetching: {url}")
    r = requests.get(url, timeout=10)
    soup = BeautifulSoup(r.text, "html.parser")

    # Remove scripts, styles, headers, navs
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.extract()

    # Clean text
    text = soup.get_text(separator="\n")
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    return text


# -------------------------------------------------------
#  Build Vector DB (only once)
# -------------------------------------------------------
def initialize_vector_db():

    if os.path.exists(PERSIST_DIR):
        print("Loading existing website Chroma...")
        embeddings = HuggingFaceEmbeddings()
        return Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME
        )

    print("Scraping website...")
    documents = []

    for url in WEBSITE_URLS:
        text = fetch_and_clean(url)
        documents.append(text)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    # Convert into document objects
    docs = splitter.create_documents(documents)

    print(f"Website produced {len(docs)} chunks")

    embeddings = HuggingFaceEmbeddings()

    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
        collection_name=COLLECTION_NAME
    )

    print("Website vector DB created.")
    return vectordb


vectorstore = initialize_vector_db()
print("Website RAG Vector Store is ready.")
exit(0)
# -------------------------------------------------------
#  MCP Tool: Website Query
# -------------------------------------------------------
@mcp.tool()
def query_website(query: str) -> list:
    """
    Answer the user's question using only internal policy knowledge.
    Do NOT mention websites, URLs, crawling, scraping, pages, or technical sources.
    Provide clear, helpful answers.try to drill down into more deeper if you need to go into another links and gets answer,
    and dont ask for further clarification or assistance and end the conversation.
    
    If information is unavailable, respond politely with:
    - "I'm sorry, I don't have that information available."
    - "I don't have access to that information right now."
    - "I'm unable to provide that information at the moment."
    """

    results = vectorstore.similarity_search(query, k=3)
    return [doc.page_content for doc in results]


# -------------------------------------------------------
#  Run MCP Server
# -------------------------------------------------------
if __name__ == "__main__":
    mcp.run(transport="stdio")
