
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama


# ==========================================
# 1. CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).parent
TEXT_FILE = BASE_DIR / "cpp.txt"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = "gemma2:2b"


# ==========================================
# 2. LOAD THE TEXT FILE
# ==========================================

print("\n[1/5] Loading C++ notes...")

if not TEXT_FILE.exists():
    raise FileNotFoundError(
        f"Could not find {TEXT_FILE}\n"
        "Please create cpp.txt in the same folder as rag.py."
    )

loader = TextLoader(str(TEXT_FILE), encoding="utf-8")
documents = loader.load()

if not any(doc.page_content.strip() for doc in documents):
    raise ValueError("cpp.txt is empty. Add some C++ notes first.")

print("Text file loaded successfully!")


# ==========================================
# 3. SPLIT TEXT INTO CHUNKS
# ==========================================

print("\n[2/5] Splitting text into chunks...")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20
)

chunks = text_splitter.split_documents(documents)

print(f"Total chunks created: {len(chunks)}")


# ==========================================
# 4. CREATE EMBEDDINGS AND FAISS DATABASE
# ==========================================

print("\n[3/5] Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)

print("Embeddings model loaded!")

print("\n[4/5] Creating FAISS vector database...")

db = FAISS.from_documents(chunks, embeddings)

print("FAISS vector database created successfully!")


# ==========================================
# 5. CONNECT TO GEMMA 2 USING OLLAMA
# ==========================================

print("\n[5/5] Connecting to Gemma 2...")

llm = ChatOllama(
    model=OLLAMA_MODEL,
    temperature=0
)

print("Gemma 2 is ready!")
print("\n" + "=" * 50)
print("       C++ RAG CHATBOT")
print("=" * 50)
print("Ask questions about your cpp.txt notes.")
print("Type 'exit' to close the chatbot.")


# ==========================================
# 6. CHAT LOOP: RETRIEVAL + ANSWER GENERATION
# ==========================================

while True:
    try:
        query = input("\nYou: ").strip()

        if not query:
            print("Please enter a question.")
            continue

        if query.lower() in ("exit", "quit", "bye"):
            print("Goodbye!")
            break

        # Retrieve the four most relevant passages
        results = db.similarity_search(query, k=4)

        context = "\n\n".join(
            doc.page_content for doc in results
        )

        # Ask Gemma to answer using retrieved context
        prompt = f"""
You are a helpful C++ programming tutor.

Answer the user's question using the provided context.
Explain the concept clearly and simply.

If the context does not contain enough information,
say that the notes do not provide enough information.
Do not pretend that missing information is present.

RETRIEVED CONTEXT:
{context}

USER QUESTION:
{query}

ANSWER:
"""

        print("\nGemma is generating an answer...")

        response = llm.invoke(prompt)

        print("\nGemma:")
        print(response.content)

        # Display sources used by the RAG pipeline
        print("\n--- Retrieved passages ---")

        for index, doc in enumerate(results, start=1):
            print(f"\nPassage {index}:")
            print(doc.page_content)

        print("\n" + "-" * 50)

    except KeyboardInterrupt:
        print("\n\nChatbot stopped.")
        break

    except Exception as error:
        print(f"\nAn error occurred: {error}")
        print(
            "Check that Ollama is running and "
            "gemma2:2b is installed."
        )
