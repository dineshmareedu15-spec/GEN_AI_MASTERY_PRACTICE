test_query = "What is SKU-4521?"

for k in (60, 10):
    print(f"\n--- k = {k} ---")
    for r in hybrid_search(test_query, n_results=3, k=k):
        print(f"[{r['rrf_score']}] ({r['source']}) {r['text'][:70]}...")
