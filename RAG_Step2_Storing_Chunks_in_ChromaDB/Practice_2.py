# Safe re-indexing: upsert twice (count must not grow), edit one chunk in place, delete a whole source by metadata
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
    chunk_size=240, chunk_overlap=35, separators=["\n\n", "\n", ". ", " ", ""]
)
hf_embedder = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

client = chromadb.Client()
collection = client.create_collection(
    name="space_reindex", embedding_function=hf_embedder, metadata={"hnsw:space": "cosine"}
)


def index_document(source, text):
    pieces = splitter.split_text(text)
    collection.upsert(
        documents=pieces,
        metadatas=[{"source": source, "chunk_index": i} for i in range(len(pieces))],
        ids=[f"{source}_chunk_{i}" for i in range(len(pieces))],
    )
    return len(pieces)


report = {}

counts = []
for run in (1, 2):
    for source, text in documents.items():
        index_document(source, text)
    counts.append(collection.count())
report["count_after_run_1"], report["count_after_run_2"] = counts
report["upsert_is_idempotent"] = counts[0] == counts[1]

target = "mars_rovers.txt_chunk_0"
old = collection.get(ids=[target], include=["documents", "metadatas"])
collection.upsert(
    documents=[old["documents"][0] + " [reviewed]"], metadatas=old["metadatas"], ids=[target]
)
new = collection.get(ids=[target], include=["documents"])
report["chunk_edited_in_place"] = new["documents"][0].endswith("[reviewed]")
report["count_after_edit"] = collection.count()

collection.delete(where={"source": "space_station.txt"})
left = {m["source"] for m in collection.get(include=["metadatas"])["metadatas"]}
report["sources_left_after_delete"] = sorted(left)
report["deleted_source_is_gone"] = "space_station.txt" not in left

print(json.dumps(report, indent=2))
