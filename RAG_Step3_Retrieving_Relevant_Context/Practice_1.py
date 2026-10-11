# Similarity bars: see how fast cosine similarity falls off and which chunks survive a 0.3 cut-off
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter

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
    chunk_size=220, chunk_overlap=30, separators=["\n\n", "\n", ". ", " ", ""]
)
client = chromadb.Client()
hf_embedder = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = client.create_collection(
    name="space_retrieval", embedding_function=hf_embedder, metadata={"hnsw:space": "cosine"}
)

for source, text in documents.items():
    pieces = splitter.split_text(text)
    collection.add(
        documents=pieces,
        metadatas=[{"source": source, "chunk_index": i} for i in range(len(pieces))],
        ids=[f"{source}_chunk_{i}" for i in range(len(pieces))],
    )
print("Total chunks stored:", collection.count())


def retrieve(query, n_results=4, where=None):
    n_results = min(n_results, collection.count())
    kwargs = {"query_texts": [query], "n_results": n_results,
              "include": ["documents", "metadatas", "distances"]}
    if where:
        kwargs["where"] = where
    res = collection.query(**kwargs)
    return [
        {"text": d, "source": m["source"], "chunk_index": m["chunk_index"], "similarity": round(1 - dist, 4)}
        for d, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]


CUTOFF = 0.3

for q in ["How do rovers talk to Earth?",
          "Why put a telescope in space?",
          "How do astronauts get clean water?",
          "Who won the football world cup?"]:
    print(f"\nQuery: {q}")
    for r in retrieve(q, n_results=4):
        bar = "#" * max(int(r["similarity"] * 40), 0)
        verdict = "keep" if r["similarity"] >= CUTOFF else "drop"
        print(f"  {r['similarity']:>6.3f} |{bar:<40}| {verdict} | {r['source']} #{r['chunk_index']}")
