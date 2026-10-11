# Threshold calibration: grid of chunks kept (n_results x threshold), then a cut-off suggested from in-scope vs off-topic questions
import json
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


in_scope = [
    "Which rover collects rock samples?",
    "How big is the Webb mirror?",
    "How long do astronauts stay on the station?",
]
off_topic = [
    "How do I bake sourdough bread?",
    "What is the best programming language for beginners?",
    "Who won the football world cup?",
]

probe = in_scope[0]
thresholds = [0.2, 0.3, 0.4, 0.5]
print(f"\nChunks kept for: {probe!r}")
print(f"{'n_results':<11}" + "".join(f"t>={t:<6}" for t in thresholds))
for n in (1, 3, 5):
    hits = retrieve(probe, n_results=n)
    print(f"{n:<11}" + "".join(f"{sum(h['similarity'] >= t for h in hits):<8}" for t in thresholds))

in_top = [retrieve(q, n_results=1)[0]["similarity"] for q in in_scope]
off_top = [retrieve(q, n_results=1)[0]["similarity"] for q in off_topic]

summary = {
    "lowest_in_scope_top_similarity": round(min(in_top), 3),
    "highest_off_topic_top_similarity": round(max(off_top), 3),
}
if min(in_top) > max(off_top):
    summary["suggested_threshold"] = round((min(in_top) + max(off_top)) / 2, 3)
else:
    summary["suggested_threshold"] = None
    summary["note"] = "scores overlap - a single cut-off cannot separate them, add keyword search or a fallback"
print("\n" + json.dumps(summary, indent=2))
