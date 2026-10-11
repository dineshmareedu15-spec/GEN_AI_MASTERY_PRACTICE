# Add a fourth document at run time, then check that in-scope questions are answered and the off-topic one is refused
from Practice_1 import collection, store_document, rag_answer, NO_ANSWER

lunar_text = """Lunar Exploration

The Apollo programme landed twelve astronauts on the Moon between 1969 and 1972. Samples they brought back are still studied today.

The Artemis programme aims to return astronauts to the Moon and to build a base near the lunar south pole, where ice may be found in permanently shadowed craters.
"""

added = store_document("lunar_exploration.txt", lunar_text)
print(f"Added {added} chunks. Total now: {collection.count()}")

tests = [
    ("Who walked on the Moon during the Apollo programme?", True),
    ("Where might the Artemis programme find ice?", True),
    ("What is the capital city of Australia?", False),
]

for question, should_answer in tests:
    answer = rag_answer(question, show_sources=False)
    refused = NO_ANSWER.lower().rstrip(".") in answer.lower()
    ok = (not refused) if should_answer else refused
    print(f"\n[{'PASS' if ok else 'FAIL'}] {question}")
    print("   ->", answer.replace("\n", " ")[:160])
