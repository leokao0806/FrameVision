"""
Zero-Shot Vision-Language Model Framing Pipeline (Qwen2.5-VL through a local Ollama server)
CS 5134/6034 NLP Fall 2026 - Team FrameVision
Authors: Leo Kao, En-Shuo Zhang

The prompt adapts the structured JSON prompting idea of Arora et al. (EMNLP 2025) to the
SemEval taxonomy. Every raw model reply is saved so the predictions can be audited.
"""

import base64
import io
import json
import requests
from PIL import Image

from src.data_loader import FRAME_TO_ID, labels_to_vector

OLLAMA_URL = "http://localhost:11434/api/chat"
VLM_MODEL = "qwen2.5vl:7b"
MAX_TEXT_CHARS = 1500
MAX_IMAGE_SIDE = 1024

FRAME_DEFINITIONS = """Economic: costs, benefits, or other financial implications.
Capacity_and_resources: availability of physical, human, or financial resources.
Morality: religious or ethical implications, duty, honor, or social responsibility.
Fairness_and_equality: balance or distribution of rights, equality, discrimination.
Legality_Constitutionality_and_jurisprudence: rights, laws, courts, legal limits.
Policy_prescription_and_evaluation: specific policies proposed or evaluated.
Crime_and_punishment: criminal activity, law enforcement, arrests, sentencing.
Security_and_defense: threats to welfare, military forces, national or personal security.
Health_and_safety: healthcare, illness, public safety, injury.
Quality_of_life: wealth, happiness, well-being, hardship, community life.
Cultural_identity: social norms, traditions, heritage, identity of groups.
Public_opinion: attitudes of the general public, protests, polling.
Political: political actors, maneuvering, elections, lobbying, party stances.
External_regulation_and_reputation: foreign relations, treaties, international reputation."""

VARIANTS = ("image_only", "subtask_2a", "subtask_2b", "image_only_v2", "subtask_2a_title", "subtask_2a_fulltext")

# E10: image_only with an explicit None option and its definition; only the last paragraph differs.
NONE_INSTRUCTION_V2 = (
    "None: the image shows no clear visual evidence for any category above (for example, a generic "
    "portrait, a logo, or scenery unrelated to the issue).\n"
    "Select the framings conveyed by the IMAGE itself. Choose only framings with clear visual evidence. "
    'If no category has clear visual evidence, answer {"frames": ["None"]}. Answer with JSON only, '
    'in the form {"frames": ["Category_name"]}.'
)


def article_title(text):
    """The first non-empty line; every article text starts with its title."""
    return next((line.strip() for line in text.splitlines() if line.strip()), "")


def build_prompt(document, variant):
    """Builds the prompt; only subtask_2b sees the gold framings of the article text."""
    parts = [
        "You are analyzing how a news photograph frames an issue.",
        "Framing categories:\n" + FRAME_DEFINITIONS,
    ]
    if variant in ("subtask_2a", "subtask_2b"):
        parts.append("Article text (it may be in any language):\n" + document["text"][:MAX_TEXT_CHARS])
    # E12: same prompt as subtask_2a; only the amount of article text differs.
    if variant == "subtask_2a_title":
        parts.append("Article title (it may be in any language):\n" + article_title(document["text"]))
    if variant == "subtask_2a_fulltext":
        parts.append("Article text (it may be in any language):\n" + document["text"])
    if variant == "subtask_2b":
        parts.append("Human-annotated framings of the article text: " + (", ".join(document["text_frames"]) or "None"))
    if variant == "image_only_v2":
        parts.append(NONE_INSTRUCTION_V2)
    else:
        parts.append(
            "Select the framings conveyed by the IMAGE itself. Choose only framings with clear visual "
            "evidence; most images convey zero to three. Answer with JSON only, in the form "
            '{"frames": ["Category_name"]}. Use an empty list if no framing applies.'
        )
    return "\n\n".join(parts)


def _encode_image(path):
    image = Image.open(path).convert("RGB")
    image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def parse_frames(reply):
    """Returns (valid_frames, parse_ok); labels outside the taxonomy are discarded."""
    try:
        frames = json.loads(reply)["frames"]
        return sorted({f for f in frames if f in FRAME_TO_ID}), True
    except (json.JSONDecodeError, KeyError, TypeError):
        return [], False


def query_vlm(document, variant, seed=42):
    response = requests.post(OLLAMA_URL, timeout=600, json={
        "model": VLM_MODEL,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0, "seed": seed, "num_ctx": 8192},
        "messages": [{
            "role": "user",
            "content": build_prompt(document, variant),
            "images": [_encode_image(document["image_path"])],
        }],
    })
    response.raise_for_status()
    body = response.json()
    reply = body["message"]["content"]
    frames, parse_ok = parse_frames(reply)
    # prompt_eval_count shows whether the prompt fit in num_ctx (Ollama silently truncates longer prompts).
    return {"doc_id": document["doc_id"], "variant": variant, "raw_reply": reply,
            "frames": frames, "parse_ok": parse_ok, "labels": labels_to_vector(frames).tolist(),
            "prompt_eval_count": body.get("prompt_eval_count")}
