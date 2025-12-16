from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from mcp_client import MCPAgentService

app = FastAPI()

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="."), name="static")

agent = MCPAgentService()


# -------------------------------------------------------------
# Prevent browser favicon.ico request from becoming customer_id
# -------------------------------------------------------------
@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)


# -------------------------------------------------------------
# Initialize MCP agent
# -------------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    print("🔄 Initializing MCP Agent...")
    await agent.initialize()
    print("✅ MCP Agent Ready!")


# -------------------------------------------------------------
# UI Entry: Load chat for given customer ID
# http://localhost:8001/C001
# -------------------------------------------------------------
@app.get("/{customer_id}", response_class=HTMLResponse)
async def home(request: Request, customer_id: str):
    agent.customer_id = customer_id   # Store customer ID in agent
    print(f"[INFO] Customer ID set to: {customer_id}")

    return templates.TemplateResponse(
        "chat.html",
        {
            "request": request,
            "customer_id": customer_id
        }
    )


# -------------------------------------------------------------
# Chat Messaging Endpoint
# Frontend sends only: { "message": "..." }
# customer_id is already stored in agent.customer_id
# -------------------------------------------------------------
@app.post("/ask")
async def ask(request: Request):
    try:
        body = await request.json()
        user_msg = body.get("message")

        print(f"[DEBUG] Incoming msg: {user_msg}")
        print(f"[DEBUG] Active customer_id: {agent.customer_id}")

        # Pass ACTIVE customer ID to agent
        reply = await agent.run(
            user_msg,
            customer_id=agent.customer_id
        )

        return JSONResponse({"reply": reply})

    except Exception as e:
        print("\n[INTERNAL ERROR] MCP/LLM Exception:", str(e), "\n")
        return JSONResponse(
            {"reply": "⚠️ Something went wrong while processing your request. Please try again."},
            status_code=500
        )


# -------------------------------------------------------------
# Run FastAPI
# -------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fastApi:app", host="127.0.0.1", port=8001, reload=True)
