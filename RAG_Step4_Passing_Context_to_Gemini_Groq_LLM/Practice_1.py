# Shared setup for Step 4: space documents -> ChromaDB (cosine) -> retrieve -> prompt -> Groq or Gemini
# Practice_2 to Practice_5 import from this file. Put GROQ_API_KEY or GEMINI_API_KEY in a .env file next to it.
import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils import embedding_functions

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

if not GROQ_API_KEY and not GEMINI_API_KEY:
    raise SystemExit("No API key found. Add GROQ_API_KEY or GEMINI_API_KEY to your .env file and run again.")

PROVIDER = "groq" if GROQ_API_KEY else "gemini"
print("Using provider:", PROVIDER)

documents = {
    "mars_rovers.txt": """Mars Rovers: A Short Guide

A rover is a wheeled robot that drives across the surface of Mars to study rocks and soil. Rovers carry cameras, drills and small laboratories so scientists on Earth can learn about the planet's past.

Perseverance landed in Jezero Crater in 2021 and collects rock samples sealed in tubes for a possible future return to Earth.

Curiosity landed in Gale Crater in 2012 and found evidence that the area once held fresh water and the chemical ingredients for life.

Because radio signals take several minutes to cross space, rovers cannot be driven with a joystick. Engineers send a plan once per day and the rover carries it out by itself.
""",
    "space_telescopes.txt": """Space Telescopes

Telescopes in space avoid the blurring effect of Earth's atmosphere, so they can take much sharper pictures than telescopes on the ground.

The Hubble Space Telescope was launched in 1990 and orbits close to Earth. It mainly observes visible and ultraviolet light.

The James Webb Space Telescope was launched in 2021 and sits about 1.5 million kilometres from Earth. It observes infrared light, which lets it see very distant galaxies and look through dust clouds.

The main mirror of Webb is made of 18 gold-coated hexagonal segments that work together as one large mirror.
""",
    "space_station.txt": """The International Space Station

The International Space Station, or ISS, is a research laboratory that circles Earth about every 90 minutes at a height of roughly 400 kilometres.

Astronauts on board run experiments in biology, medicine and materials science that cannot be done under normal gravity. Crews usually stay for about six months.

Air, water and electricity are recycled or generated on board. Large solar arrays supply power, and water from sweat and breath moisture is cleaned and reused.

Supply ships bring food, tools and new experiments from Earth every few months.
""",
}

splitter = RecursiveCharacterTextSplitter(
    chunk_size=240, chunk_overlap=35, separators=["\n\n", "\n", ". ", " ", ""]
)
client = chromadb.Client()
hf_embedder = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = client.create_collection(
    name="space_rag", embedding_function=hf_embedder, metadata={"hnsw:space": "cosine"}
)


def store_document(source, text):
    pieces = splitter.split_text(text)
    collection.add(
        documents=pieces,
        metadatas=[{"source": source, "chunk_index": i} for i in range(len(pieces))],
        ids=[f"{source}_chunk_{i}" for i in range(len(pieces))],
    )
    return len(pieces)


for source, text in documents.items():
    store_document(source, text)
print("Total chunks stored:", collection.count())


def retrieve_context(query, n_results=3, min_similarity=0.3):
    n_results = min(n_results, collection.count())
    res = collection.query(
        query_texts=[query], n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    chunks = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        similarity = 1 - dist
        if similarity >= min_similarity:
            chunks.append({"text": doc, "source": meta["source"],
                           "chunk_index": meta["chunk_index"], "similarity": round(similarity, 4)})
    return chunks


NO_ANSWER = "I don't know based on the provided documents."

SYSTEM_PROMPT = (
    "You answer questions using ONLY the context you are given. "
    f'If the answer is not in the context, reply exactly: "{NO_ANSWER}" '
    "Keep the answer to two sentences at most."
)


def build_user_prompt(question, chunks):
    context = "\n\n".join(f"[Source: {c['source']}, chunk {c['chunk_index']}]\n{c['text']}" for c in chunks)
    return f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"


GROQ_MODELS = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
GEMINI_MODELS = ["gemini-2.5-flash", "gemini-2.5-flash-lite"]


def ask_groq(user_prompt, system_prompt=SYSTEM_PROMPT, temperature=0.2):
    from groq import Groq
    groq_client = Groq(api_key=GROQ_API_KEY)
    last_error = None
    for model_name in GROQ_MODELS:
        try:
            response = groq_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "system", "content": system_prompt},
                          {"role": "user", "content": user_prompt}],
                temperature=temperature,
                max_tokens=500,
            )
            if response.choices[0].message.content:
                return response.choices[0].message.content.strip()
        except Exception as e:
            last_error = e
    return f"[Groq error: {last_error}]"


def ask_gemini(user_prompt, system_prompt=SYSTEM_PROMPT, temperature=0.2):
    from google import genai
    from google.genai import types
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    last_error = None
    for model_name in GEMINI_MODELS:
        try:
            response = gemini_client.models.generate_content(
                model=model_name,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=temperature,
                    max_output_tokens=1000,
                ),
            )
            if response.text:
                return response.text.strip()
        except Exception as e:
            last_error = e
    return f"[Gemini error: {last_error}]"


def ask_llm(user_prompt, system_prompt=SYSTEM_PROMPT, temperature=0.2):
    if PROVIDER == "groq":
        return ask_groq(user_prompt, system_prompt, temperature)
    return ask_gemini(user_prompt, system_prompt, temperature)


def rag_answer(question, n_results=3, system_prompt=SYSTEM_PROMPT, temperature=0.2, show_sources=True):
    chunks = retrieve_context(question, n_results=n_results)
    if not chunks:
        return NO_ANSWER + " (nothing relevant was retrieved)"
    answer = ask_llm(build_user_prompt(question, chunks), system_prompt, temperature)
    if show_sources:
        answer += "\n   sources: " + ", ".join(sorted({f"{c['source']}#{c['chunk_index']}" for c in chunks}))
    return answer
