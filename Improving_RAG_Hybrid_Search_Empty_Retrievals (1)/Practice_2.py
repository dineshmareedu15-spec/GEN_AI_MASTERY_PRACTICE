from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi

documents = {
    "ai_overview.txt": """Artificial Intelligence: An Overview

Artificial Intelligence, or AI, is the field of computer science focused on building systems
that can perform tasks which normally require human intelligence.

Machine Learning is a subset of AI where systems learn patterns from data instead of following
hardcoded rules.

Deep Learning takes Machine Learning further by using neural networks with many layers to
automatically discover complex patterns in data.

Retrieval-Augmented Generation, or RAG, combines a search system with a language model so answers
are grounded in real retrieved documents instead of only the model's memorized training data.
""",
    "embeddings_in_practice.txt": """Embeddings and Vector Search in Practice

Modern search engines increasingly rely on embeddings rather than simple keyword matching, allowing
systems to find results that match the meaning of a query, not just its exact words.

Vector databases such as ChromaDB store these embeddings and allow fast similarity search across
thousands or millions of documents at once.
""",
    "product_catalog.txt": """Product Catalog Notes

The wireless noise-cancelling headphones are listed under product code SKU-4521 and are currently
our best-selling audio accessory.

The standing desk converter is listed under product code SKU-7788 and ships within three business
days.

Return requests for any product must reference the exact SKU code so our warehouse team can locate
the correct item quickly.
"""
}

splitter = RecursiveCharacterTextSplitter(
    chunk_size=220, chunk_overlap=30, separators=["\n\n", "\n", ". ", " ", ""]
)

client = chromadb.Client()
hf_embedder = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = client.create_collection(name="rag_hybrid_demo", embedding_function=hf_embedder)

all_chunks = []
all_chunk_ids = []
all_chunk_meta = []

for source, text in documents.items():
    doc_chunks = splitter.split_text(text)
    doc_ids = [f"{source}_chunk_{i}" for i in range(len(doc_chunks))]
    doc_meta = [{"source": source, "chunk_index": i} for i in range(len(doc_chunks))]
    collection.add(documents=doc_chunks, metadatas=doc_meta, ids=doc_ids)
    all_chunks.extend(doc_chunks)
    all_chunk_ids.extend(doc_ids)
    all_chunk_meta.extend(doc_meta)


def semantic_search(query, n_results=3):
    n_results = min(n_results, collection.count())
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )
    output = []
    for doc, meta, distance in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        output.append({"text": doc, "source": meta["source"], "similarity": round(1 - distance, 4)})
    return output


def tokenize(text):
    return text.lower().replace("-", " ").split()


bm25 = BM25Okapi([tokenize(c) for c in all_chunks])


def bm25_search(query, n_results=3):
    scores = bm25.get_scores(tokenize(query))
    ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n_results]
    output = []
    for idx in ranked_idx:
        if scores[idx] > 0:
            output.append({
                "text": all_chunks[idx],
                "source": all_chunk_meta[idx]["source"],
                "bm25_score": round(float(scores[idx]), 4)
            })
    return output


def hybrid_search(query, n_results=3, k=60):
    semantic_results = semantic_search(query, n_results=10)
    keyword_results = bm25_search(query, n_results=10)

    rrf_scores = {}
    chunk_lookup = {}

    for rank, r in enumerate(semantic_results):
        key = r["text"]
        rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (k + rank + 1)
        chunk_lookup[key] = r

    for rank, r in enumerate(keyword_results):
        key = r["text"]
        rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (k + rank + 1)
        chunk_lookup.setdefault(key, r)

    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:n_results]
    return [{"text": t, "source": chunk_lookup[t]["source"], "rrf_score": round(s, 5)} for t, s in ranked]


def retrieve_with_threshold(query, n_results=5, similarity_threshold=0.3):
    n_results = min(n_results, collection.count())
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )
    filtered = []
    for doc, meta, distance in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        similarity = 1 - distance
        if similarity >= similarity_threshold:
            filtered.append({"text": doc, "source": meta["source"], "similarity": round(similarity, 4)})
    return filtered


KNOWLEDGE_BASE_TOPICS = sorted({m["source"] for m in all_chunk_meta})


def smart_retrieve(query, n_results=4, similarity_threshold=0.3):
    primary = retrieve_with_threshold(query, n_results=n_results, similarity_threshold=similarity_threshold)
    if primary:
        return {"status": "ok", "method": "semantic", "chunks": primary}

    hybrid_results = hybrid_search(query, n_results=n_results)
    if hybrid_results:
        return {"status": "ok", "method": "hybrid_fallback", "chunks": hybrid_results}

    return {
        "status": "empty",
        "method": None,
        "chunks": [],
        "message": (
            "I don't know based on the provided documents. "
            f"This knowledge base currently covers: {', '.join(KNOWLEDGE_BASE_TOPICS)}."
        )
    }


new_source = "warranty_info.txt"
new_text = """Warranty Information

The ergonomic office chair is listed under product code SKU-9034 and comes with a two year warranty.
Warranty claims must include the exact SKU code and the original purchase receipt.
"""

new_chunks = splitter.split_text(new_text)
new_ids = [f"{new_source}_chunk_{i}" for i in range(len(new_chunks))]
new_meta = [{"source": new_source, "chunk_index": i} for i in range(len(new_chunks))]
collection.add(documents=new_chunks, metadatas=new_meta, ids=new_ids)

all_chunks.extend(new_chunks)
all_chunk_ids.extend(new_ids)
all_chunk_meta.extend(new_meta)

bm25 = BM25Okapi([tokenize(c) for c in all_chunks])

KNOWLEDGE_BASE_TOPICS = sorted({m["source"] for m in all_chunk_meta})

result = smart_retrieve("Tell me about SKU-9034", similarity_threshold=0.9)
print(result["status"], "|", result["method"])
for c in result["chunks"]:
    print(" -", c["source"], "|", c["text"][:70])
