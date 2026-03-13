from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET


def _extract_text_from_level(root: ET.Element, level: str) -> list[str]:
    nodes = root.findall(f".//{{*}}{level}/{{*}}TextEquiv/{{*}}Unicode")
    return [node.text.strip() for node in nodes if node.text and node.text.strip()]


def parse_pagexml_text(path: str | Path) -> str:
    """Extract PAGE XML transcription without duplicating mixed granularity annotations.

    Preference order:
    1) TextLine/TextEquiv/Unicode
    2) Word/TextEquiv/Unicode
    3) Any TextEquiv/Unicode fallback
    """
    tree = ET.parse(path)
    root = tree.getroot()

    line_texts = _extract_text_from_level(root, "TextLine")
    if line_texts:
        return "\n".join(line_texts)

    word_texts = _extract_text_from_level(root, "Word")
    if word_texts:
        return "\n".join(word_texts)

    unicode_nodes = root.findall(".//{*}TextEquiv/{*}Unicode")
    fallback = [node.text.strip() for node in unicode_nodes if node.text and node.text.strip()]
    return "\n".join(fallback)
