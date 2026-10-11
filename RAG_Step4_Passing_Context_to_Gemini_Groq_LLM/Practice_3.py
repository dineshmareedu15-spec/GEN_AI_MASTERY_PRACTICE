# Temperature test: ask the same question three times at temperature 0.0 and 0.9 and see whether the answers repeat
from Practice_1 import retrieve_context, build_user_prompt, ask_llm

question = "Why can rovers not be driven with a joystick?"
prompt = build_user_prompt(question, retrieve_context(question))

for temp in (0.0, 0.9):
    answers = [ask_llm(prompt, temperature=temp) for _ in range(3)]
    print(f"\n--- temperature {temp} ---")
    for i, a in enumerate(answers, 1):
        print(f"  run {i}: {a.replace(chr(10), ' ')[:110]}")
    print(f"  distinct answers: {len(set(answers))} of 3 | identical: {len(set(answers)) == 1}")
