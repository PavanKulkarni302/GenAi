import streamlit as st
import asyncio
import nest_asyncio

nest_asyncio.apply()

from strmlit_client import MCPAgentService


st.set_page_config(page_title="MCP Multi-Server Agent", layout="wide")
st.title("💬 Order & Policy Assistant")


# -----------------------------------------------------
# Initialize MCP Agent Only Once
# -----------------------------------------------------
async def initialize_agent():
    if "agent_initialized" not in st.session_state:
        st.session_state.agent_service = MCPAgentService()
        await st.session_state.agent_service.initialize()
        st.session_state.agent_initialized = True


# Run async initialization safely
asyncio.get_event_loop().run_until_complete(initialize_agent())


# -----------------------------------------------------
# Setup Chat History
# -----------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []


# -----------------------------------------------------
# Display Chat History
# -----------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# -----------------------------------------------------
# User Input
# -----------------------------------------------------
prompt = st.chat_input("Ask anything…")

if prompt:
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            async def get_reply():
                return await st.session_state.agent_service.run_query(prompt)

            reply = asyncio.get_event_loop().run_until_complete(get_reply())
            st.markdown(reply)

    # Save assistant response
    st.session_state.messages.append({"role": "assistant", "content": reply})
