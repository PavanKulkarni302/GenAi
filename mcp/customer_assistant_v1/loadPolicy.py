from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# -------------------------------------------------------
#  MCP Server Setup
# -------------------------------------------------------
mcp = FastMCP(name="website_rag_server")

load_dotenv()

PERSIST_DIR = "./chroma_db"          # MUST already exist
COLLECTION_NAME = "rag_collection"      # MUST match your DB


# -------------------------------------------------------
#  Load Existing Vector DB ONLY
# -------------------------------------------------------
def load_vector_db():
    if not os.path.exists(PERSIST_DIR):
        raise FileNotFoundError(
            f"ERROR: No existing Chroma DB found at {PERSIST_DIR}. "
            "Run your scraper/builder script first."
        )

    # print("Loading existing website Chroma DB...")

    embeddings = HuggingFaceEmbeddings()

    vectordb = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )

    return vectordb


# Load DB at startup
vectorstore = load_vector_db()


# -------------------------------------------------------
#  MCP Tool: Query Website Policies
# -------------------------------------------------------
@mcp.tool()
def query_website(query: str) -> list:
    """
    Answer the user's question using only internal policy knowledge.

    Do NOT mention:
    - websites
    - links
    - URLs
    - crawling
    - scraping
    - pages
    - documents
    - chunks

    Provide clear, helpful, direct answers. 

    Always attempt to answer the user's question using the provided internal policy knowledge. 
    Even if the search results are partial or incomplete, create the best possible answer 
    based on the available text without asking the user for more details.
    Only use fallback messages if the text absolutely contains zero relevant info.
    only use the database to answer questions about company policies, dont try to hit any website or external source.

    If information is unavailable, respond with:
    - "I'm sorry, I don't have that information available."
    - "I don't have access to that information right now."
    - "I'm unable to provide that information at the moment."

    """

    results = vectorstore.similarity_search(query, k=8)
    return [doc.page_content for doc in results]


# -------------------------------------------------------
#  Run MCP Server
# -------------------------------------------------------
if __name__ == "__main__":
    mcp.run(transport="stdio")
