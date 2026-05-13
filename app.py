import streamlit as st
# Import your existing function and setup
from pdf_loader import rag_simple, rag_retriever, llm

st.title("📚 My RAG Assistant")

# Initialize chat history in the browser's session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get user input from the chat box
if prompt := st.chat_input("Ask me anything (e.g., Types of solar cells)"):
    
    # 1. Display user message in UI
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Get the AI response using your exact function!
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Here is your engine running!
            answer = rag_simple(prompt, rag_retriever, llm)
            st.markdown(answer)
            
    # 3. Save the response to chat history
    st.session_state.messages.append({"role": "assistant", "content": answer})