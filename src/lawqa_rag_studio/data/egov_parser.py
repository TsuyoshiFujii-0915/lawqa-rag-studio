"""Parser for e-Gov XML into structured law trees."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable

from lawqa_rag_studio.data.law_tree import LawNode


def _clean(text: str | None) -> str:
    """Normalize text for indexing and display.

    Args:
        text: Raw text, possibly None.

    Returns:
        Cleaned text with collapsed whitespace.
    """
    if not text:
        return ""
    return " ".join(text.split())


def _local_name(tag: str) -> str:
    """Return the local tag name without namespaces.

    Args:
        tag: Element tag which may contain an XML namespace.

    Returns:
        Local tag name.
    """
    return tag.split("}", 1)[-1]


def _node_id(parent_node_id: str, node_type: str, num: str | None, idx: int) -> str:
    """Build a stable node_id using parent id and element number when available.

    Args:
        parent_node_id: Parent node identifier.
        node_type: Node type string.
        num: Optional XML number attribute.
        idx: 0-based index fallback when num is missing.

    Returns:
        Node identifier string.
    """
    suffix = _clean(num) if num else str(idx)
    return f"{parent_node_id}-{node_type}-{suffix}"


def _extract_sentences(container: ET.Element, sentence_container_tag: str) -> str:
    """Extract sentence text from a container element.

    Args:
        container: Element containing sentence container tags.
        sentence_container_tag: Tag name like 'ParagraphSentence' or 'ItemSentence'.

    Returns:
        Concatenated sentence text for the container.
    """
    parts: list[str] = []
    for sent_container in container.findall(f"./{sentence_container_tag}"):
        text = _clean("".join(sent_container.itertext()))
        if text:
            parts.append(text)
    return "\n".join(parts)


def _parse_item(item_el: ET.Element, parent_node_id: str, law_id: str, idx: int) -> LawNode:
    """Parse an <Item> element into a LawNode.

    Args:
        item_el: XML element for the item.
        parent_node_id: Parent node identifier.
        law_id: Law identifier.
        idx: Item index within the paragraph.

    Returns:
        Parsed item node.
    """
    num = item_el.get("Num")
    title = _clean(item_el.findtext("./ItemTitle")) or (num or None)
    text = _extract_sentences(item_el, "ItemSentence")
    return {
        "node_id": _node_id(parent_node_id, "item", num, idx),
        "type": "item",
        "title": title or None,
        "text": text or None,
        "children": [],
        "meta": {"law_id": law_id, "item_num": num, "item_index": idx},
    }


def _parse_paragraph(
    para_el: ET.Element,
    parent_node_id: str,
    law_id: str,
    article_num: str | None,
    idx: int,
) -> LawNode:
    """Parse a <Paragraph> element into a LawNode.

    Args:
        para_el: XML element for the paragraph.
        parent_node_id: Parent node identifier.
        law_id: Law identifier.
        article_num: Article number string if available.
        idx: Paragraph index within the article.

    Returns:
        Parsed paragraph node including any item children.
    """
    num = para_el.get("Num")
    paragraph_num_text = _clean(para_el.findtext("./ParagraphNum")) or ""
    title = paragraph_num_text or (num or None)
    text = _extract_sentences(para_el, "ParagraphSentence")

    paragraph_node_id = _node_id(parent_node_id, "para", num, idx)
    items: list[LawNode] = []
    for item_idx, item_el in enumerate(para_el.findall("./Item")):
        items.append(_parse_item(item_el, paragraph_node_id, law_id, item_idx))

    return {
        "node_id": paragraph_node_id,
        "type": "paragraph",
        "title": title or None,
        "text": text or None,
        "children": items,
        "meta": {
            "law_id": law_id,
            "article_num": article_num,
            "paragraph_num": num,
            "paragraph_index": idx,
        },
    }


def _parse_article(article_el: ET.Element, parent_node_id: str, law_id: str, idx: int) -> LawNode:
    """Parse an <Article> element into a LawNode.

    Args:
        article_el: XML element for the article.
        parent_node_id: Parent node identifier.
        law_id: Law identifier.
        idx: Article index within its container.

    Returns:
        Parsed article node including paragraph children.
    """
    num = article_el.get("Num")
    article_title = _clean(article_el.findtext("./ArticleTitle"))
    caption = _clean(article_el.findtext("./ArticleCaption"))
    title = " ".join([p for p in [article_title, caption] if p]) or (num or f"Article {idx}")

    article_node_id = _node_id(parent_node_id, "article", num, idx)
    paragraphs: list[LawNode] = []
    for p_idx, para_el in enumerate(article_el.findall("./Paragraph")):
        paragraphs.append(_parse_paragraph(para_el, article_node_id, law_id, num, p_idx))

    return {
        "node_id": article_node_id,
        "type": "article",
        "title": title or None,
        "text": None,
        "children": paragraphs,
        "meta": {"law_id": law_id, "article_num": num, "article_index": idx},
    }


def _parse_subsection(
    subsection_el: ET.Element,
    parent_node_id: str,
    law_id: str,
    idx: int,
) -> LawNode:
    """Parse a <Subsection> element into a LawNode.

    Args:
        subsection_el: XML element for the subsection.
        parent_node_id: Parent node identifier.
        law_id: Law identifier.
        idx: Subsection index within the section.

    Returns:
        Parsed subsection node.
    """
    num = subsection_el.get("Num")
    title = _clean(subsection_el.findtext("./SubsectionTitle")) or (num or f"Subsection {idx}")
    node_id = _node_id(parent_node_id, "subsection", num, idx)

    children: list[LawNode] = []
    for child_idx, child_el in enumerate(list(subsection_el)):
        tag = _local_name(child_el.tag)
        if tag == "Division":
            children.append(_parse_division(child_el, node_id, law_id, child_idx))
        elif tag == "Article":
            children.append(_parse_article(child_el, node_id, law_id, child_idx))

    return {
        "node_id": node_id,
        "type": "subsection",
        "title": title or None,
        "text": None,
        "children": children,
        "meta": {"law_id": law_id, "subsection_num": num, "subsection_index": idx},
    }


def _parse_section(section_el: ET.Element, parent_node_id: str, law_id: str, idx: int) -> LawNode:
    """Parse a <Section> element into a LawNode.

    Args:
        section_el: XML element for the section.
        parent_node_id: Parent node identifier.
        law_id: Law identifier.
        idx: Section index within the chapter.

    Returns:
        Parsed section node.
    """
    num = section_el.get("Num")
    title = _clean(section_el.findtext("./SectionTitle")) or (num or f"Section {idx}")
    node_id = _node_id(parent_node_id, "section", num, idx)

    children: list[LawNode] = []
    for child_idx, child_el in enumerate(list(section_el)):
        tag = _local_name(child_el.tag)
        if tag == "Subsection":
            children.append(_parse_subsection(child_el, node_id, law_id, child_idx))
        elif tag == "Division":
            children.append(_parse_division(child_el, node_id, law_id, child_idx))
        elif tag == "Article":
            children.append(_parse_article(child_el, node_id, law_id, child_idx))

    return {
        "node_id": node_id,
        "type": "section",
        "title": title or None,
        "text": None,
        "children": children,
        "meta": {"law_id": law_id, "section_num": num, "section_index": idx},
    }


def _parse_chapter(chapter_el: ET.Element, parent_node_id: str, law_id: str, idx: int) -> LawNode:
    """Parse a <Chapter> element into a LawNode.

    Args:
        chapter_el: XML element for the chapter.
        parent_node_id: Parent node identifier.
        law_id: Law identifier.
        idx: Chapter index within main provision.

    Returns:
        Parsed chapter node.
    """
    num = chapter_el.get("Num")
    title = _clean(chapter_el.findtext("./ChapterTitle")) or (num or f"Chapter {idx}")
    node_id = _node_id(parent_node_id, "chapter", num, idx)

    children: list[LawNode] = []
    for child_idx, child_el in enumerate(list(chapter_el)):
        tag = _local_name(child_el.tag)
        if tag == "Section":
            children.append(_parse_section(child_el, node_id, law_id, child_idx))
        elif tag == "Subsection":
            children.append(_parse_subsection(child_el, node_id, law_id, child_idx))
        elif tag == "Division":
            children.append(_parse_division(child_el, node_id, law_id, child_idx))
        elif tag == "Article":
            children.append(_parse_article(child_el, node_id, law_id, child_idx))

    return {
        "node_id": node_id,
        "type": "chapter",
        "title": title or None,
        "text": None,
        "children": children,
        "meta": {"law_id": law_id, "chapter_num": num, "chapter_index": idx},
    }


def _parse_main_provision(main_el: ET.Element, law_id: str) -> list[LawNode]:
    """Parse the <MainProvision> element into top-level children.

    Args:
        main_el: Main provision element.
        law_id: Law identifier.

    Returns:
        Top-level nodes under the law (chapters, sections, subsections, or articles).
    """
    children: list[LawNode] = []
    for idx, el in enumerate(list(main_el)):
        tag = _local_name(el.tag)
        if tag == "Chapter":
            children.append(_parse_chapter(el, law_id, law_id, idx))
        elif tag == "Section":
            children.append(_parse_section(el, law_id, law_id, idx))
        elif tag == "Subsection":
            children.append(_parse_subsection(el, law_id, law_id, idx))
        elif tag == "Division":
            children.append(_parse_division(el, law_id, law_id, idx))
        elif tag == "Article":
            children.append(_parse_article(el, law_id, law_id, idx))
    return children


def _parse_division(division_el: ET.Element, parent_node_id: str, law_id: str, idx: int) -> LawNode:
    """Parse a <Division> element into a LawNode.

    Args:
        division_el: XML element for the division.
        parent_node_id: Parent node identifier.
        law_id: Law identifier.
        idx: Division index within the container.

    Returns:
        Parsed division node.
    """
    num = division_el.get("Num")
    title = _clean(division_el.findtext("./DivisionTitle")) or (num or f"Division {idx}")
    node_id = _node_id(parent_node_id, "division", num, idx)

    children: list[LawNode] = []
    for child_idx, child_el in enumerate(list(division_el)):
        tag = _local_name(child_el.tag)
        if tag == "Article":
            children.append(_parse_article(child_el, node_id, law_id, child_idx))

    return {
        "node_id": node_id,
        "type": "division",
        "title": title or None,
        "text": None,
        "children": children,
        "meta": {"law_id": law_id, "division_num": num, "division_index": idx},
    }


def parse_egov_xml(xml_path: Path) -> LawNode:
    """Parse a single e-Gov XML file into a structured law tree.

    Args:
        xml_path: Path to the XML file.

    Returns:
        Parsed `LawNode` tree preserving the document hierarchy.
    """
    law_id = xml_path.stem
    tree = ET.parse(xml_path)
    root = tree.getroot()

    title = _clean(root.findtext("./LawBody/LawTitle")) or _clean(root.findtext("LawTitle")) or law_id

    law_node: LawNode = {
        "node_id": law_id,
        "type": "law",
        "title": title,
        "text": None,
        "children": [],
        "meta": {"source_path": str(xml_path)},
    }

    law_body = root.find("./LawBody")
    if law_body is None:
        law_body = root

    # Main provision (usually structured under chapters/sections)
    main = law_body.find("./MainProvision")
    main_container = main if main is not None else law_body
    children: list[LawNode] = _parse_main_provision(main_container, law_id)

    # Supplementary provisions often contain articles directly.
    for suppl_idx, suppl_el in enumerate(law_body.findall("./SupplProvision")):
        suppl_title = _clean(suppl_el.findtext("./SupplProvisionLabel")) or "Supplementary Provisions"
        amend = _clean(suppl_el.get("AmendLawNum"))
        title = f"{suppl_title} {amend}".strip()
        node_id = _node_id(law_id, "suppl_provision", suppl_el.get("AmendLawNum"), suppl_idx)
        suppl_children = _parse_main_provision(suppl_el, law_id)
        children.append(
            {
                "node_id": node_id,
                "type": "suppl_provision",
                "title": title or None,
                "text": None,
                "children": suppl_children,
                "meta": {"law_id": law_id, "amend_law_num": amend or None, "suppl_index": suppl_idx},
            }
        )

    law_node["children"] = children

    return law_node


def parse_multiple(files: Iterable[Path]) -> list[LawNode]:
    """Parse multiple XML files.

    Args:
        files: Iterable of XML file paths.

    Returns:
        List of parsed law trees.
    """
    return [parse_egov_xml(path) for path in files]


__all__ = ["parse_egov_xml", "parse_multiple"]
