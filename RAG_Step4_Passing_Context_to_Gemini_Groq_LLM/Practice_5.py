# Does retrieval help? Same questions with no context, 1 chunk and 4 chunks - does the answer contain the expected fact?
from Practice_1 import retrieve_context, build_user_prompt, ask_llm

checks = [
    ("How many hexagonal segments make up the main mirror of Webb?", "18"),
    ("In which crater did the Curiosity rover land?", "Gale"),
    ("How often does the ISS go around Earth?", "90"),
]

PLAIN_PROMPT = "Answer in one short sentence."

print(f"{'mode':<14}{'chunks':<8}{'has fact':<10}answer")
for question, fact in checks:
    print(f"\nQ: {question}   (expected: {fact})")

    answer = ask_llm(f"Question: {question}", system_prompt=PLAIN_PROMPT, temperature=0.0)
    print(f"{'no context':<14}{0:<8}{('yes' if fact in answer else 'no'):<10}{answer.replace(chr(10), ' ')[:80]}")

    for n in (1, 4):
        chunks = retrieve_context(question, n_results=n)
        if not chunks:
            print(f"{f'{n} chunk(s)':<14}{0:<8}{'no':<10}(nothing passed the similarity cut-off)")
            continue
        answer = ask_llm(build_user_prompt(question, chunks), temperature=0.0)
        print(f"{f'{n} chunk(s)':<14}{len(chunks):<8}{('yes' if fact in answer else 'no'):<10}{answer.replace(chr(10), ' ')[:80]}")
