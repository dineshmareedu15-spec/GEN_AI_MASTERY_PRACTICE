# Three chunkers on the same text (fixed / recursive / sentence-based): which one stops mid-sentence the least?
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter

space_text = """Mars Rovers: A Short Guide

A rover is a wheeled robot that drives across the surface of Mars to study rocks and soil. Rovers carry cameras, drills and small laboratories so scientists on Earth can learn about the planet's past.

Perseverance landed in Jezero Crater in 2021 and collects rock samples sealed in tubes for a possible future return to Earth.

Curiosity landed in Gale Crater in 2012 and found evidence that the area once held fresh water and the chemical ingredients for life.

Because radio signals take several minutes to cross space, rovers cannot be driven with a joystick. Engineers send a plan once per day and the rover carries it out by itself.

Space Telescopes

Telescopes in space avoid the blurring effect of Earth's atmosphere, so they can take much sharper pictures than telescopes on the ground.

The Hubble Space Telescope was launched in 1990 and orbits close to Earth. It mainly observes visible and ultraviolet light.

The James Webb Space Telescope was launched in 2021 and sits about 1.5 million kilometres from Earth. It observes infrared light, which lets it see very distant galaxies and look through dust clouds.

The main mirror of Webb is made of 18 gold-coated hexagonal segments that work together as one large mirror.

The International Space Station

The International Space Station, or ISS, is a research laboratory that circles Earth about every 90 minutes at a height of roughly 400 kilometres.

Astronauts on board run experiments in biology, medicine and materials science that cannot be done under normal gravity. Crews usually stay for about six months.

Air, water and electricity are recycled or generated on board. Large solar arrays supply power, and water from sweat and breath moisture is cleaned and reused.

Supply ships bring food, tools and new experiments from Earth every few months.
"""


def fixed_chunks(text, size=220, overlap=30):
    step = size - overlap
    return [text[i:i + size] for i in range(0, len(text), step)]


def sentence_chunks(text, per_chunk=2):
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]
    return [" ".join(sentences[i:i + per_chunk]) for i in range(0, len(sentences), per_chunk)]


recursive = RecursiveCharacterTextSplitter(
    chunk_size=220, chunk_overlap=30, separators=["\n\n", "\n", ". ", " ", ""]
)

strategies = {
    "fixed-size": fixed_chunks(space_text),
    "recursive": recursive.split_text(space_text),
    "sentence x2": sentence_chunks(space_text),
}


def ends_cleanly(chunk):
    return chunk.rstrip()[-1:] in (".", "!", "?")


print(f"{'strategy':<14}{'chunks':<8}{'avg len':<9}{'clean ends':<12}bar")
scores = {}
for name, chunks in strategies.items():
    clean = sum(ends_cleanly(c) for c in chunks)
    pct = 100 * clean / len(chunks)
    scores[name] = pct
    avg_len = sum(len(c) for c in chunks) / len(chunks)
    print(f"{name:<14}{len(chunks):<8}{avg_len:<9.0f}{pct:>5.0f}%      {'#' * int(pct / 5)}")

print("\nBest at keeping sentences whole:", max(scores, key=scores.get))
