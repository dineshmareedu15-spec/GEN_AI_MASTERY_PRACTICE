# Overlap sweep: same chunk_size, overlap = 0 / 30 / 60 -> chunk count and text repeated between neighbours
# (word-level separators are used on purpose so the overlap is easy to see)
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

flat_text = " ".join(space_text.split())     # one long line, no paragraph breaks


def shared_text(a, b):
    # length of the longest piece that ends chunk a and also starts chunk b
    for k in range(min(len(a), len(b)), 0, -1):
        if a[-k:] == b[:k]:
            return k
    return 0


print(f"{'overlap':<10}{'chunks':<9}{'avg length':<12}{'avg repeated chars'}")
for overlap in (0, 30, 60):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=150, chunk_overlap=overlap, separators=[" ", ""]
    )
    chunks = splitter.split_text(flat_text)
    repeats = [shared_text(chunks[i], chunks[i + 1]) for i in range(len(chunks) - 1)]
    avg_len = sum(len(c) for c in chunks) / len(chunks)
    avg_rep = sum(repeats) / max(len(repeats), 1)
    print(f"{overlap:<10}{len(chunks):<9}{avg_len:<12.1f}{avg_rep:.1f}")
