import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

FAISS_INDEX_PATH = "faiss_index"

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
)


def load_pdf(file_path: str):
    """Load a single PDF and split into chunks."""
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_documents(docs)


def load_pdfs(file_paths: list[str]):
    """Load multiple PDFs and split into chunks."""
    all_chunks = []
    for path in file_paths:
        all_chunks.extend(load_pdf(path))
    return all_chunks


def create_vector_store(chunks):
    """Create a new FAISS vector store from document chunks."""
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(FAISS_INDEX_PATH)
    return vector_store


def load_vector_store():
    """Load an existing FAISS index from disk."""
    if os.path.exists(FAISS_INDEX_PATH):
        return FAISS.load_local(
            FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True
        )
    return None


def add_to_vector_store(vector_store, chunks):
    """Add new document chunks to an existing vector store."""
    new_store = FAISS.from_documents(chunks, embeddings)
    vector_store.merge_from(new_store)
    vector_store.save_local(FAISS_INDEX_PATH)
    return vector_store


def search(vector_store, query: str, k: int = 4):
    """Similarity search on the vector store."""
    return vector_store.similarity_search(query, k=k)


def ask(vector_store, query: str, history: list[dict] | None = None):
    """Retrieve relevant context and generate an answer using Google Gemini LLM."""
    results = search(vector_store, query)
    context = "\n\n".join([doc.page_content for doc in results])

    messages = [
        SystemMessage(
            content="""You are a helpful legal assistant. Provide answers based on the given context and query," as a friendly assistant, greet the user and provide a concise answer
        dont provide any information that is not present in the context. If you don't know the answer, say you don't know. Always be concise and to the point."""
        )
    ]

    #  chat history
    for msg in (history or []):
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))

    messages.append(HumanMessage(content=f"Context:\n{context}\n\nQuestion:\n{query}"))

    llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", google_api_key=gemini_api_key)
    response = llm.invoke(messages)
    content = response.content
    if isinstance(content, list):
        content = "".join(
            block["text"] for block in content if block.get("type") == "text"
        )
    return content, context
