# Store three space documents with metadata in a cosine collection, then verify it with PASS/FAIL checks
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
    chunk_size=240, chunk_overlap=35, separators=["\n\n", "\n", ". ", " ", ""]
)
hf_embedder = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

client = chromadb.Client()
collection = client.create_collection(
    name="space_chunks", embedding_function=hf_embedder, metadata={"hnsw:space": "cosine"}
)

expected = 0
for source, text in documents.items():
    pieces = splitter.split_text(text)
    collection.add(
        documents=pieces,
        metadatas=[{"source": source, "chunk_index": i, "word_count": len(p.split())}
                   for i, p in enumerate(pieces)],
        ids=[f"{source}_chunk_{i}" for i in range(len(pieces))],
    )
    expected += len(pieces)
    print(f"{source:<24}-> {len(pieces)} chunks")

stored = collection.get(include=["metadatas"])
per_source = {}
for m in stored["metadatas"]:
    per_source[m["source"]] = per_source.get(m["source"], 0) + 1
print("\nChunks per source:", per_source)

checks = {
    "collection count equals chunks created": collection.count() == expected,
    "all ids are unique": len(set(stored["ids"])) == len(stored["ids"]),
    "every chunk has word_count > 0": all(m["word_count"] > 0 for m in stored["metadatas"]),
    "every source got at least one chunk": set(per_source) == set(documents),
}
for name, ok in checks.items():
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")

res = collection.query(
    query_texts=["Where do astronauts get their electricity?"],
    n_results=2,
    include=["documents", "metadatas", "distances"],
)
print("\nQuery: Where do astronauts get their electricity?")
for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
    print(f"  sim={1 - dist:.3f} | {meta['source']} #{meta['chunk_index']} | {doc[:60]!r}")
