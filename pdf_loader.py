import os
import re
import json
from typing import List

# LangChain Core & Documents
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Retrieval & Vector Store
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever

# LLM
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# ==========================================
# CUSTOM ENSEMBLE RETRIEVER (No dependency)
# ==========================================
class EnsembleRetriever(BaseRetriever):
    retrievers: list
    weights: list

    def _get_relevant_documents(self, query: str) -> List[Document]:
        all_docs = {}
        for retriever, weight in zip(self.retrievers, self.weights):
            docs = retriever.invoke(query)
            for rank, doc in enumerate(docs):
                key = doc.page_content[:100]
                score = weight * (1 / (rank + 1))  # Reciprocal Rank Fusion
                if key in all_docs:
                    all_docs[key][1] += score
                else:
                    all_docs[key] = [doc, score]
        sorted_docs = sorted(all_docs.values(), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in sorted_docs]


# ==========================================
# 1. DATA INGESTION (JSON-to-Chunks)
# ==========================================
def load_and_structure_gita(json_path="gita_structured.json"):
    """Loads the parsed JSON and creates metadata-rich chunks."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Could not find {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        gita_data = json.load(f)

    all_docs = []
    purport_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )

    for item in gita_data:
        base_meta = {
            "chapter": item["chapter"],
            "verse": item["verse"],
            "source": "Bhagavad-gita As It Is"
        }

        # 1. Sanskrit Shloka
        if item.get("sanskrit"):
            all_docs.append(Document(
                page_content=item["sanskrit"],
                metadata={**base_meta, "type": "sanskrit"}
            ))

        # 2. English Translation
        if item.get("translation"):
            all_docs.append(Document(
                page_content=item["translation"],
                metadata={**base_meta, "type": "translation"}
            ))

        # 3. Purport (Chunked)
        if item.get("purport"):
            p_chunks = purport_splitter.split_text(item["purport"])
            for i, chunk in enumerate(p_chunks):
                all_docs.append(Document(
                    page_content=chunk,
                    metadata={**base_meta, "type": "purport", "part": i + 1}
                ))

    print(f"Total structured chunks created: {len(all_docs)}")
    return all_docs


# ==========================================
# 2. EMBEDDING & VECTOR STORE (Chroma)
# ==========================================
chunks = load_and_structure_gita("gita_structured.json")

embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
persist_dir = "./gita_chroma_db"

if os.path.exists(persist_dir) and len(os.listdir(persist_dir)) > 0:
    print("Loading existing ChromaDB from disk...")
    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings_model,
        collection_name="gita_hybrid_v1"
    )
else:
    print("Building new ChromaDB...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings_model,
        persist_directory=persist_dir,
        collection_name="gita_hybrid_v1"
    )


# ==========================================
# 3. HYBRID RETRIEVER PIPELINE
# ==========================================
print("Building Hybrid Search (BM25 + Vector)...")

keyword_retriever = BM25Retriever.from_documents(chunks, k=2)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

hybrid_retriever = EnsembleRetriever(
    retrievers=[keyword_retriever, vector_retriever],
    weights=[0.4, 0.6]
)
print("Hybrid Retriever Ready.")

# Alias for app.py import
rag_retriever = hybrid_retriever


# ==========================================
# 4. LLM CONFIGURATION
# ==========================================

# 1. REMOVE the hardcoded line: os.environ["OPENROUTER_API_KEY"] = "sk-..."
# 2. INSTEAD, use getenv which will read the key from the Cloud Secrets
api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    # This helps you debug if the key isn't loading correctly in the cloud
    print("⚠️ Warning: OPENROUTER_API_KEY not found in environment variables.")

llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key, # Use the variable we just defined
    model="ling-2.6-flash",
    temperature=0.1,
    max_tokens=1024
)
print("LLM Initialized.")


# ==========================================
# 5. SMART VERSE DETECTOR
# ==========================================
def detect_verse_query(query: str):
    q = query.lower().strip()

    # Last verse of entire Gita
    if "last verse" in q or "final verse" in q or "last shloka" in q:
        if "chapter" not in q and "ch" not in q:
            return 18, 78

    # First verse of entire Gita
    if ("first verse" in q or "first shloka" in q) and "chapter" not in q:
        return 1, 1

    # First verse of a specific chapter
    first_ch_match = re.search(r'first\s*(?:verse|text|shloka).*?chapter\s*(\d+)', q)
    if not first_ch_match:
        first_ch_match = re.search(r'chapter\s*(\d+).*?first\s*(?:verse|text|shloka)', q)
    if first_ch_match:
        ch = int(first_ch_match.group(1))
        return ch, 1

    # Specific chapter/verse patterns
    patterns = [
        r'chapter\s*(\d+)\s*(?:verse|text|shloka|sloka|v)\s*(\d+)',
        r'ch\s*(\d+)\s*(?:verse|text|shloka|sloka|v)\s*(\d+)',
        r'\b(\d+)\.(\d+)\b',
    ]
    for pattern in patterns:
        match = re.search(pattern, q)
        if match:
            return int(match.group(1)), int(match.group(2))

    return None, None

def detect_doc_type(query: str):
    """Detects if user wants a specific content type."""
    q = query.lower()
    if "translation" in q:
        return "translation"
    elif "purport" in q:
        return "purport"
    elif "sanskrit" in q or "shloka" in q or "sloka" in q:
        return "sanskrit"
    return None  # Return all types if not specified


# ==========================================
# 6. DIRECT METADATA LOOKUP
# ==========================================
def direct_verse_lookup(chapter: int, verse: int, doc_type: str = None) -> List[Document]:
    """
    Fetches documents directly from ChromaDB using metadata filters.
    Much more accurate than semantic search for specific verse queries.
    """
    where_filter = {
        "$and": [
            {"chapter": {"$eq": chapter}},
            {"verse":   {"$eq": verse}},
        ]
    }
    if doc_type:
        where_filter["$and"].append({"type": {"$eq": doc_type}})

    results = vectorstore.get(
        where=where_filter,
        include=["documents", "metadatas"]
    )

    docs = []
    for content, meta in zip(results["documents"], results["metadatas"]):
        docs.append(Document(page_content=content, metadata=meta))

    return docs


# ==========================================
# 7. SMART RAG PIPELINE
# ==========================================
def rag_simple(query: str, retriever, llm, top_k: int = 3) -> str:
    """
    Smart RAG with two routing paths:
      - Path 1: Direct metadata lookup  → for specific chapter/verse queries
      - Path 2: Hybrid semantic search  → for conceptual/thematic queries
    """

    # --- Detect query intent ---
    chapter, verse = detect_verse_query(query)
    doc_type = detect_doc_type(query)

    # ---- PATH 1: Direct Metadata Lookup ----
    if chapter and verse:
        print(f"[Route: Direct Lookup] Chapter {chapter}, Verse {verse}, Type: {doc_type or 'all'}")
        docs = direct_verse_lookup(chapter, verse, doc_type)

        if docs:
            # Sort: sanskrit → translation → purport for clean presentation
            type_order = {"sanskrit": 0, "translation": 1, "purport": 2}
            docs.sort(key=lambda d: type_order.get(d.metadata.get("type", "purport"), 3))

            context = "\n\n".join([
                f"[Ch {d.metadata.get('chapter')}, Vs {d.metadata.get('verse')} | {d.metadata.get('type').upper()}]\n{d.page_content}"
                for d in docs
            ])

            prompt = f"""You are a knowledgeable guide on the Bhagavad Gita As It Is by Srila Prabhupada.
The user has asked for a specific verse. Present the content clearly and completely.
Do NOT add information beyond what is provided below.

Content:
{context}

Question: {query}

Answer:"""
            try:
                response = llm.invoke(prompt)
                return response.content
            except Exception as e:
                return f"Error connecting to LLM: {e}"
        else:
            return f"Sorry, Chapter {chapter}, Verse {verse} was not found in the database."

    # ---- PATH 2: Hybrid Semantic Search ----
    print(f"[Route: Semantic Search] Query: {query}")
    results = retriever.invoke(query)

    if not results:
        return "No relevant context found to answer the question."

    context_list = []
    for doc in results:
        source_info = f"[Ch {doc.metadata.get('chapter')}, Vs {doc.metadata.get('verse')} | {doc.metadata.get('type', '').upper()}]"
        context_list.append(f"{source_info}\n{doc.page_content}")

    context = "\n\n".join(context_list)

    prompt = f"""You are a spiritual guide based on the Bhagavad Gita As It Is by Srila Prabhupada.
Answer the following question accurately using ONLY the provided context.
Always mention the Chapter and Verse numbers when referencing specific teachings.

Context:
{context}

Question: {query}

Answer:"""

    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"Error connecting to LLM: {e}"


# ==========================================
# QUICK TEST (runs only when executed directly)
# ==========================================
if __name__ == "__main__":
    test_queries = [
        "chapter 1 text 1 translation",
        "chapter 2 verse 47 purport",
        "What does Krishna say about the soul?",
        "Who is Arjuna talking to in Chapter 1?",
        "2.13",
    ]
    for q in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {q}")
        print(f"{'='*60}")
        answer = rag_simple(q, hybrid_retriever, llm)
        print(answer)