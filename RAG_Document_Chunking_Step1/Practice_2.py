# Paragraph packer from scratch: merge short paragraphs up to max_chars, split long ones by sentence, then self-check
import re

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


def pack_paragraphs(text, max_chars=320):
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    # step 1: cut paragraphs into units that fit; a heading (no end punctuation) is glued to the text after it
    units, heading = [], ""
    for para in paragraphs:
        if para[-1] not in ".!?":
            heading = para
            continue
        room = max_chars - (len(heading) + 2 if heading else 0)
        if len(para) <= room:
            body = [para]
        else:
            body, current = [], ""
            for sentence in re.split(r"(?<=[.!?])\s+", para):
                if current and len(current) + 1 + len(sentence) > room:
                    body.append(current)
                    current = sentence
                else:
                    current = f"{current} {sentence}".strip()
            body.append(current)
        if heading:
            body[0] = f"{heading}\n\n{body[0]}"
            heading = ""
        units.extend(body)
    if heading:
        units.append(heading)

    # step 2: pack neighbouring units into chunks without going over max_chars
    chunks, current = [], ""
    for unit in units:
        if current and len(current) + 2 + len(unit) > max_chars:
            chunks.append(current)
            current = unit
        else:
            current = f"{current}\n\n{unit}".strip()
    if current:
        chunks.append(current)
    return chunks


def squash(s):
    return re.sub(r"\s+", "", s)


for limit in (320, 200):
    chunks = pack_paragraphs(space_text, max_chars=limit)
    print(f"\n=== max_chars = {limit} -> {len(chunks)} chunks ===")
    for i, c in enumerate(chunks, 1):
        print(f"  #{i:<2} {len(c):>3} chars | {c[:55]!r}...")

    checks = {
        "no chunk longer than the limit": all(len(c) <= limit for c in chunks),
        "no text lost or added": squash("".join(chunks)) == squash(space_text),
        "every chunk ends at a sentence end": all(c[-1] in ".!?" for c in chunks),
    }
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
