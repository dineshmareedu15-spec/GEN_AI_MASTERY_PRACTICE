# Structured output: make the model reply as JSON, parse it, validate it, and retry once if it is invalid
import json
from Practice_1 import collection, retrieve_context, build_user_prompt, ask_llm, NO_ANSWER

JSON_PROMPT = (
    "You answer questions using ONLY the context you are given. "
    "Reply with one JSON object and nothing else, in exactly this shape: "
    '{"answer": "<one or two sentences>", "sources": ["<file name>"], "found": true}. '
    f'If the context does not contain the answer, use "answer": "{NO_ANSWER}", "sources": [] and "found": false.'
)

known_files = {m["source"] for m in collection.get(include=["metadatas"])["metadatas"]}


def parse_json_reply(text):
    text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def ask_json(question):
    chunks = retrieve_context(question)
    if not chunks:
        return {"answer": NO_ANSWER, "sources": [], "found": False}
    prompt = build_user_prompt(question, chunks)
    for attempt in (1, 2):
        data = parse_json_reply(ask_llm(prompt, system_prompt=JSON_PROMPT, temperature=0.0))
        if data is not None:
            return data
        print(f"   (attempt {attempt} was not valid JSON)")
    return None


def validate(data):
    if data is None:
        return {"valid JSON": False}
    return {
        "valid JSON": True,
        "has answer / sources / found": {"answer", "sources", "found"} <= set(data),
        "answer is text": isinstance(data.get("answer"), str) and data["answer"] != "",
        "found is true/false": isinstance(data.get("found"), bool),
        "sources are real files": isinstance(data.get("sources"), list)
                                  and all(s in known_files for s in data["sources"]),
    }


for q in ["How many segments does the Webb mirror have?",
          "How often do supply ships reach the space station?",
          "Which company makes the best electric cars?"]:
    data = ask_json(q)
    print(f"\nQ: {q}\n{json.dumps(data, indent=2)}")
    for name, ok in validate(data).items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
