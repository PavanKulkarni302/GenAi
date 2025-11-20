from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
import os
import snowflake.connector

# Create MCP server instance
mcp = FastMCP(name="snowflake_mcp_server")

print("Loading environment variables...")
load_dotenv()

# Read environment variables
ACCOUNT = os.getenv("SNOWFLAKE_ACCOUNT")
USER = os.getenv("SNOWFLAKE_USER")
PASSWORD = os.getenv("SNOWFLAKE_PASSWORD")
DATABASE = os.getenv("SNOWFLAKE_DATABASE")
SCHEMA = os.getenv("SNOWFLAKE_SCHEMA")
WAREHOUSE = os.getenv("SNOWFLAKE_WAREHOUSE")

print("Environment variables loaded.")

# -----------------------------
#   HELPER: SNOWFLAKE CONNECT
# -----------------------------
def get_connection():
    return snowflake.connector.connect(
        account=ACCOUNT,
        user=USER,
        password=PASSWORD,
        warehouse=WAREHOUSE,
        database=DATABASE,
        schema=SCHEMA
    )

# -----------------------------
#   MCP TOOL: RUN SQL QUERY
# -----------------------------
@mcp.tool()
def query_snowflake(sql: str) -> list:
    """
    Execute SQL in Snowflake and return rows as list of objects.
    Retrieve the required information from internal systems and present the results clearly.
    Do not mention SQL, queries, tables, Snowflake, databases, or any technical execution details.
    Return only the final answer in a customer-friendly manner.
    """
    print("Executing SQL:", sql)

    ctx = get_connection()
    cursor = ctx.cursor()

    try:
        cursor.execute(sql)
        rows = cursor.fetchall()
        columns = [c[0] for c in cursor.description]

        results = [dict(zip(columns, row)) for row in rows]
        return results

    finally:
        cursor.close()
        ctx.close()

if __name__=="__main__":
    mcp.run(transport="stdio")