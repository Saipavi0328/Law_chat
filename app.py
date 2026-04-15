import os
import tempfile
import streamlit as st
from rag import (
    load_pdfs,
    create_vector_store,
    load_vector_store,
    add_to_vector_store,
    ask,
)

st.set_page_config(page_title="Law Chatbot", page_icon="⚖️", layout="wide")
st.title("⚖️ Law Chatbot")

# --- Sidebar: PDF Upload ---
with st.sidebar:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF files", type=["pdf"], accept_multiple_files=True
    )

    if st.button("Process Documents", disabled=not uploaded_files):
        with st.spinner("Processing PDFs..."):
            temp_dir = tempfile.mkdtemp()
            saved_paths = []
            for f in uploaded_files:
                path = os.path.join(temp_dir, f.name)
                with open(path, "wb") as out:
                    out.write(f.getbuffer())
                saved_paths.append(path)

            chunks = load_pdfs(saved_paths)

            store = load_vector_store()
            if store:
                add_to_vector_store(store, chunks)
            else:
                store = create_vector_store(chunks)

            st.session_state["vector_store"] = store
            st.session_state["doc_names"] = st.session_state.get("doc_names", []) + [
                f.name for f in uploaded_files
            ]
        st.success(f"Processed {len(uploaded_files)} PDF(s)!")

    # Load existing index on first run
    if "vector_store" not in st.session_state:
        store = load_vector_store()
        if store:
            st.session_state["vector_store"] = store

    if st.session_state.get("doc_names"):
        st.markdown("**Loaded documents:**")
        for name in st.session_state["doc_names"]:
            st.markdown(f"- {name}")

# --- Main: Chat ---
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if query := st.chat_input("Ask a legal question..."):
    st.session_state["messages"].append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    store = st.session_state.get("vector_store")
    if not store:
        response = "Please upload and process PDF documents first."
        st.session_state["messages"].append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)
    else:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, context = ask(store, query, st.session_state["messages"])
            st.markdown(answer)
            with st.expander("View Retrieved Context"):
                st.text(context)
        st.session_state["messages"].append({"role": "assistant", "content": answer})
