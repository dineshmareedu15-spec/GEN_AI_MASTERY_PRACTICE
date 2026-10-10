import re

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "what", "whats", "which", "who",
    "how", "do", "does", "did", "of", "to", "in", "on", "for", "and", "or", "me",
    "tell", "about", "can", "you", "please", "s", "i", "it", "this", "that", "with"
}

def remove_stopwords(query):
    tokens = re.findall(r"[a-z0-9\-]+", query.lower())
    return " ".join(t for t in tokens if t not in STOPWORDS)

def strict_hybrid_search(query, n_results=4, k=60, similarity_threshold=0.3):
    semantic = [r for r in semantic_search(query, n_results=10)
                if r["similarity"] >= similarity_threshold]
    keyword = bm25_search(query, n_results=10)

    rrf, lookup = {}, {}
    for rank, r in enumerate(semantic):
        rrf[r["text"]] = rrf.get(r["text"], 0) + 1 / (k + rank + 1)
        lookup[r["text"]] = r
    for rank, r in enumerate(keyword):
        rrf[r["text"]] = rrf.get(r["text"], 0) + 1 / (k + rank + 1)
        lookup.setdefault(r["text"], r)

    ranked = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:n_results]
    return [{"text": t, "source": lookup[t]["source"], "rrf_score": round(s, 5)}
            for t, s in ranked]

def smart_retrieve(query, n_results=4, similarity_threshold=0.3):
    primary = retrieve_with_threshold(query, n_results, similarity_threshold)
    if primary:
        return {"status": "ok", "method": "semantic", "chunks": primary}

    hybrid = strict_hybrid_search(query, n_results, similarity_threshold=similarity_threshold)
    if hybrid:
        return {"status": "ok", "method": "hybrid_fallback", "chunks": hybrid}

    cleaned = remove_stopwords(query)
    if cleaned and cleaned != query.lower():
        retry = strict_hybrid_search(cleaned, n_results, similarity_threshold=similarity_threshold)
        if retry:
            return {"status": "ok", "method": "stopword_fallback", "chunks": retry}

    return {
        "status": "empty", "method": None, "chunks": [],
        "message": ("I don't know based on the provided documents. "
                    f"This knowledge base currently covers: {', '.join(KNOWLEDGE_BASE_TOPICS)}.")
    }

for q in ["What is deep learning?", "Tell me about SKU-4521", "What's the best recipe for chocolate cake?"]:
    r = smart_retrieve(q)
    print(q, "->", r["status"], "|", r["method"])