
import streamlit as st
from pathlib import Path
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="College RAG Assistant")
st.title("College Knowledge Assistant")
st.write("Ask questions based on the provided college documents.")

DOCS_DIR = Path(__file__).parent / "documents"

@st.cache_resource
def load_vectorstore():
    documents = []

    for file_path in DOCS_DIR.iterdir():
        if file_path.suffix.lower() == ".txt":
            documents.extend(
                TextLoader(str(file_path), encoding="utf-8").load()
            )
        elif file_path.suffix.lower() == ".pdf":
            documents.extend(PyPDFLoader(str(file_path)).load())

    if not documents:
        raise ValueError("No PDF or TXT documents found in documents folder.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.from_documents(chunks, embeddings)

db = load_vectorstore()
llm = ChatOllama(model="gemma2:2b", temperature=0)

prompt = ChatPromptTemplate.from_template("""
You are a college information assistant.

Answer using ONLY the provided document excerpts.
Do not invent facts or use outside knowledge.

If the excerpts do not contain enough information, say:
"I could not find this information in the provided documents."

If the documents conflict, clearly explain the conflicting statements
and identify the source of each statement. Do not choose one as official
unless the documents establish which one is authoritative.

DOCUMENT EXCERPTS:
{context}

QUESTION:
{question}

ANSWER:
""")

query = st.text_input("Enter your question:")

if query:
    with st.spinner("Searching documents and generating answer..."):
        results = db.similarity_search(query, k=4)
        context = "\n\n".join(
            f"Source: {doc.metadata.get('source', 'Unknown document')}\n"
            f"Page: {doc.metadata.get('page', 'N/A')}\n"
            f"Content: {doc.page_content}"
            for doc in results
        )

        response = (prompt | llm).invoke({
            "context": context,
            "question": query
        })

    st.subheader("Answer")
    st.write(response.content)

    st.subheader("Retrieved Sources")
    for i, doc in enumerate(results, start=1):
        source = Path(doc.metadata.get("source", "Unknown")).name
        page = doc.metadata.get("page")
        st.markdown(f"**{i}. {source}**")
        if page is not None:
            st.caption(f"Page {page + 1}")
        st.write(doc.page_content)
