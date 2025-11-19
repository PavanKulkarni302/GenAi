import streamlit as st
import asyncio
from strmlit_client import MCPAgentService

st.set_page_config(page_title="MCP Multi-Agent UI", layout="centered")
st.title("🔗 MCP Multi-Server Agent")

# Persistent agent instance
if 'agent_service' not in st.session_state:
    st.session_state.agent_service = MCPAgentService()
    asyncio.run(st.session_state.agent_service.initialize())

query = st.text_input("Enter your question:", placeholder="Ask anything…")

if st.button("Run Query"):
    if query.strip():
        with st.spinner("Thinking…"):
            result = asyncio.run(st.session_state.agent_service.run_query(query))
        st.success("Response:")
        st.write(result)
    else:
        st.error("Please enter a prompt first.")
