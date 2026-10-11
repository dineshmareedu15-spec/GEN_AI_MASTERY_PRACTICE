# Persist chunks on disk with PersistentClient, then reopen the folder with a fresh client and query it
# (the chroma_db folder is created next to this file and is listed in .gitignore)
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

DB_PATH = "chroma_db"
NAME = "space_saved"


def build_and_save():
    writer = chromadb.PersistentClient(path=DB_PATH)
    try:
        writer.delete_collection(NAME)      # start clean so the script can be re-run
    except Exception:
        pass
    col = writer.create_collection(
        name=NAME, embedding_function=hf_embedder, metadata={"hnsw:space": "cosine"}
    )
    for source, text in documents.items():
        pieces = splitter.split_text(text)
        col.add(
            documents=pieces,
            metadatas=[{"source": source, "chunk_index": i} for i in range(len(pieces))],
            ids=[f"{source}_chunk_{i}" for i in range(len(pieces))],
        )
    return col.count()


saved_count = build_and_save()

# a brand-new client object: nothing is added here, the data must come from disk
reader = chromadb.PersistentClient(path=DB_PATH)
reopened = reader.get_collection(name=NAME, embedding_function=hf_embedder)

res = reopened.query(
    query_texts=["Which telescope looks at infrared light?"],
    n_results=1,
    include=["documents", "metadatas", "distances"],
)
top_source = res["metadatas"][0][0]["source"]

print(json.dumps({
    "saved_count": saved_count,
    "reopened_count": reopened.count(),
    "counts_match": saved_count == reopened.count(),
    "top_source_for_infrared_question": top_source,
    "top_source_is_telescopes": top_source == "space_telescopes.txt",
}, indent=2))
