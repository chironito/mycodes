#!/usr/bin/env python3
"""
PDF → JSON / XML pipeline (no AGPL/GPL deps)

Stack (permissive licenses):
  - pdfplumber (MIT) → text, layout primitives, tables
  - pdfminer.six (MIT) → used under the hood by pdfplumber
  - (optional) pypdfium2 (BSD-3) → page-rendering if you later want images/thumbnails

What this does now (hardened):
  - Parses text per page and keeps basic structure
  - Stricter heading detection: requires min text length & lexical complexity; rejects vertical/ornamental lines
  - Detects bullet/numbered lists and coalesces into list blocks
  - Dual-pass table extraction (lattice → stream) + plausibility checks
  - Writes JSON and parallel XML (ElementTree)
  - Adds bbox (x0,y0,x1,y1) to blocks; coalesced blocks get union bbox
  - Prints a small QA summary at the end

Notes:
  - Images are ignored
  - Tune heading thresholds via `HeadingHeuristics` and CLI flags
  - Table quality depends on PDFs; adjust `TABLE_SETTINGS` and `is_plausible_table`

Usage:
  python pdf_to_json_xml_pipeline.py input.pdf --out-base out/mydoc

Outputs:
  out/mydoc.json
  out/mydoc.xml
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import requests
import xml.etree.ElementTree as ET
from dotenv import load_dotenv
assert load_dotenv("env.env"), "Environment File not Found."
import pdfplumber  # MIT
from xml.etree.ElementTree import Element, SubElement, ElementTree

def strip_json_fencing(text: str) -> str:
    match = re.search(r"```(?:json)?(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip("`").lstrip("json").strip()

JSON_CORRECTION_PROMPT = """You are a JSON structure normalizer for clinical-study documents.

Input:
A valid JSON with schema:
{
  "document": {
    "meta": {...},
    "pages": [
      {
        "page_number": <int>,
        "blocks": [
          { "type": "heading"|"paragraph"|"list"|"table", ... }
        ]
      }
    ]
  }
}

Tasks (light-touch; no hallucinations):
1) Deduplicate identical or near-identical tables on the same page (keep one).
2) Remove paragraph blocks that merely repeat table headers already present in a table on the same page.
3) Recover bullets/numbered items inside paragraphs into proper { "type":"list", "ordered": bool, "items":[...] } blocks.
4) Merge split timeline items: any consecutive blocks starting with "Day <number>:" belong in a single ordered list of items for that section.
5) Preserve page order and within-page block order except where deduplication removes a duplicate.
6) Do not invent new values or keys. Only regroup existing text.
7) Ensure the output remains valid JSON in the same schema.

Output ONLY the corrected JSON. No commentary, no code fences."""

XML_CORRECTION_PROMPT = """You are an XML structure normalizer for clinical-study documents.

Input:
Well-formed XML using elements:
<document>, <page>, <heading>, <p>, <list ordered="true|false">, <item>, <table>, <row>, <cell>

Tasks (light-touch; no hallucinations):
1) Deduplicate identical or near-identical <table> nodes on the same <page> (keep one).
2) Remove <p> nodes that merely repeat table headers already present in a <table> on the same <page>.
3) Recover bullets/numbered items inside <p> into a <list> with <item> children (set ordered="true" for numbered, "false" for bullets).
4) Merge split timeline lines: consecutive blocks starting with "Day <number>:" become a single ordered <list> of <item> within the same section/page.
5) Preserve <page> order and within-page block order except where deduplication removes a duplicate.
6) Do not invent new rows/columns/values. Only regroup existing text.
7) Output must remain well-formed XML using the same elements.

Return ONLY the corrected XML. No commentary, no code fences."""

JSON_FIX_PROMPT = """You are a JSON syntax corrector.

Input:
1. A possibly malformed JSON string.
2. A parser error message that hints at where the problem is.

Task:
- Output valid JSON that preserves the original structure and data as much as possible.
- Fix only syntax issues (e.g. missing commas, unescaped quotes, mismatched brackets, trailing commas).
- Do not invent new keys or values; only repair what is broken.
- Ensure the output parses cleanly with a standard JSON parser.

Return ONLY the corrected JSON, nothing else."""

XML_FIX_PROMPT = """You are an XML syntax corrector.

Input:
1. A possibly malformed XML document.
2. A parser error message that hints at where the problem is.

Task:
- Output valid XML that preserves the original structure and data as much as possible.
- Fix only syntax issues (e.g. mismatched or unclosed tags, missing quotes, wrong nesting).
- Do not invent new elements or attributes; only repair what is broken.
- Ensure the output parses cleanly with a standard XML parser.

Return ONLY the corrected XML, nothing else."""

def call_llm(system_prompt: str, broken: str, error_msg: str) -> str:
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {os.environ['togetherai_api_key']}"}
    payload = {"messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": broken}], "model": "openai/gpt-oss-120b"}
    r = requests.post(os.environ['togetherai_api_endpoint'], headers=headers, json=payload)
    return r.json()["choices"][0]["message"]["content"]

def fix_json(broken: str, error_msg: str, max_retries: int = 3) -> str:
    """Fix JSON string until it parses."""
    attempt = broken
    for _ in range(max_retries):
        try:
            json.loads(attempt)
            return attempt  # already valid
        except Exception as e:
            attempt = call_llm(JSON_FIX_PROMPT, attempt, str(e))
    # final check
    json.loads(attempt)
    return attempt


def fix_xml(broken: str, error_msg: str, max_retries: int = 3) -> str:
    """Fix XML string until it parses."""
    attempt = broken
    for _ in range(max_retries):
        try:
            ET.fromstring(attempt)
            return attempt  # already valid
        except Exception as e:
            attempt = call_llm(XML_FIX_PROMPT, attempt, str(e))
    # final check
    ET.fromstring(attempt)
    return attempt

def json_correction_llm(json_string: str):
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {os.environ['togetherai_api_key']}"}
    payload = {"messages": [{"role": "system", "content": JSON_CORRECTION_PROMPT}, {"role": "user", "content": json_string}], "model": "openai/gpt-oss-120b"}
    try:
        r = requests.post(os.environ['togetherai_api_endpoint'], json=payload, headers=headers)
        corrected_json_string = strip_json_fencing(r.json()["choices"][0]["message"]["content"])
    except Exception as exp:
        raise
    try:
        corrected_json = json.loads(corrected_json_string)
    except Exception as exp:
        corrected_json = json.loads(strip_json_fencing(fix_json(corrected_json_string, str(exp))))
    finally:
        return corrected_json

def xml_correction_llm(xml_string: str):
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {os.environ['togetherai_api_key']}"}
    payload = {"messages": [{"role": "system", "content": XML_CORRECTION_PROMPT}, {"role": "user", "content": xml_string}], "model": "openai/gpt-oss-120b"}
    try:
        r = requests.post(os.environ['togetherai_api_endpoint'], json=payload, headers=headers)
        corrected_xml_string = r.json()["choices"][0]["message"]["content"]
    except Exception as exp:
        raise
    try:
        ET.fromstring(corrected_xml_string)
    except Exception as exp:
        return fix_xml(corrected_xml_string, str(exp))
    else:
        return corrected_xml_string

# -----------------------------
# Tunables
# -----------------------------

BULLET_REGEX = re.compile(
    r"^\s*([•·◦\-\*\u2022\u25CF\u25E6]|\(?[0-9ivxlcdm]+\)|[0-9]+[\.)]|[A-Za-z]\.)\s+",
    re.IGNORECASE,
)

TABLE_SETTINGS = {
    # Lattice (ruled) detection first
    "vertical_strategy": "lines",
    "horizontal_strategy": "lines",
    "intersection_tolerance": 5,
    "snap_tolerance": 3,
    "join_tolerance": 3,
    "edge_min_length": 3,
    "min_words_vertical": 1,
    "min_words_horizontal": 1,
}

@dataclass
class HeadingHeuristics:
    # multipliers above median font size to be considered heading
    h1_factor: float = 1.90
    h2_factor: float = 1.60
    h3_factor: float = 1.35
    # Minimal absolute font size to consider any heading
    min_abs_pt: float = 10.0
    # minimal lexical requirements
    min_len: int = 3               # at least this many characters
    min_alpha: int = 2             # at least this many alphabetic chars
    allow_short_if_colon: bool = True  # allow short headings ending with ':'

# Strings that often appear on covers / ornamental areas — reject as headings
COVER_NOISE = re.compile(
    r"(advance\s+unedited|summary\s+of\s+results|world\s+population\s+prospects)\b",
    re.I,
)

# -----------------------------
# Helpers
# -----------------------------

def bbox_of_chars(chars: List[Dict[str, Any]]) -> Optional[Tuple[float, float, float, float]]:
    if not chars:
        return None
    x0 = min(float(c.get("x0", 0)) for c in chars)
    y0 = min(float(c.get("top", 0)) for c in chars)
    x1 = max(float(c.get("x1", 0)) for c in chars)
    y1 = max(float(c.get("bottom", 0)) for c in chars)
    return (x0, y0, x1, y1)

def union_bbox(a: Optional[Tuple[float,float,float,float]], b: Optional[Tuple[float,float,float,float]]):
    if a is None:
        return b
    if b is None:
        return a
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))

def group_chars_to_lines(chars: List[Dict[str, Any]], y_tolerance: float = 3.0) -> List[List[Dict[str, Any]]]:
    """Greedy grouping of characters into lines by y coordinate proximity."""
    if not chars:
        return []
    chars_sorted = sorted(chars, key=lambda c: (round(c.get("top", 0), 1), c.get("x0", 0)))
    lines: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = [chars_sorted[0]]

    def same_line(c1, c2) -> bool:
        return abs(c1.get("top", 0) - c2.get("top", 0)) <= y_tolerance

    for ch in chars_sorted[1:]:
        if same_line(current[-1], ch):
            current.append(ch)
        else:
            lines.append(sorted(current, key=lambda c: c.get("x0", 0)))
            current = [ch]
    lines.append(sorted(current, key=lambda c: c.get("x0", 0)))
    return lines

def line_text_and_size(line_chars: List[Dict[str, Any]]) -> Tuple[str, float]:
    text = "".join(c.get("text", "") for c in line_chars)
    sizes = [float(c.get("size", 0)) for c in line_chars if c.get("size")]
    mean_size = sum(sizes) / len(sizes) if sizes else 0.0
    return text, mean_size

def estimate_font_median(chars: List[Dict[str, Any]]) -> float:
    sizes = [float(c.get("size", 0)) for c in chars if c.get("size")]
    if not sizes:
        return 0.0
    sizes_sorted = sorted(sizes)
    mid = len(sizes_sorted) // 2
    if len(sizes_sorted) % 2 == 0:
        return (sizes_sorted[mid - 1] + sizes_sorted[mid]) / 2
    return sizes_sorted[mid]

def is_vertical_or_ornamental(line_chars: List[Dict[str, Any]], text: str) -> bool:
    """Reject vertical stacks / decorative lines likely from covers or side text."""
    if not line_chars:
        return False
    bb = bbox_of_chars(line_chars)
    if not bb:
        return False
    x0, y0, x1, y1 = bb
    width = max(1e-3, x1 - x0)
    height = max(1e-3, y1 - y0)
    tall_narrow = height / width > 2.5   # very tall & narrow
    xs = [float(c.get("x0", 0)) for c in line_chars]
    x_var = (max(xs) - min(xs))
    little_x_spread = x_var < 3.0
    many_single_chars = sum(1 for c in text if c.strip()) <= max(3, len(text) // 5)
    has_noise = bool(COVER_NOISE.search(text))
    return tall_narrow or (little_x_spread and height > 12) or has_noise or many_single_chars

def qualifies_as_heading(
    text: str, size: float, median_size: float, hh: HeadingHeuristics, line_chars: List[Dict[str, Any]]
) -> Optional[int]:
    if is_vertical_or_ornamental(line_chars, text):
        return None
    clean = text.strip()
    # lexical constraints
    if len(clean) < hh.min_len and not (hh.allow_short_if_colon and clean.endswith(":")):
        return None
    alpha = sum(ch.isalpha() for ch in clean)
    if alpha < hh.min_alpha:
        return None
    # size constraints
    if size >= max(hh.min_abs_pt, median_size * hh.h1_factor):
        return 1
    if size >= max(hh.min_abs_pt, median_size * hh.h2_factor):
        return 2
    if size >= max(hh.min_abs_pt, median_size * hh.h3_factor):
        return 3
    return None

def classify_line(
    line_chars: List[Dict[str, Any]],
    text: str,
    size: float,
    median_size: float,
    hh: HeadingHeuristics
) -> Dict[str, Any]:
    bb = bbox_of_chars(line_chars)
    level = qualifies_as_heading(text, size, median_size, hh, line_chars)
    if level is not None:
        return {"type": "heading", "level": level, "text": text.strip(), "bbox": bb}

    m = BULLET_REGEX.match(text)
    if m:
        bullet = m.group(1)
        ordered = bool(re.match(r"^\(?[0-9ivxlcdm]+\)|^[0-9]+[\.)]|^[A-Za-z]\.", bullet, re.IGNORECASE))
        clean = BULLET_REGEX.sub("", text).strip()
        return {"type": "list_item", "ordered": ordered, "text": clean, "bbox": bb}

    return {"type": "paragraph", "text": text.strip(), "bbox": bb}

def coalesce_blocks(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge consecutive list_items into lists; join paragraphs that wrap lines; compute union bbox."""
    out: List[Dict[str, Any]] = []
    i = 0
    while i < len(blocks):
        b = blocks[i]
        if b["type"] == "list_item":
            ordered = b.get("ordered", False)
            items = [b["text"]]
            bbox = b.get("bbox")
            i += 1
            while i < len(blocks) and blocks[i]["type"] == "list_item" and blocks[i].get("ordered", False) == ordered:
                items.append(blocks[i]["text"])
                bbox = union_bbox(bbox, blocks[i].get("bbox"))
                i += 1
            out.append({"type": "list", "ordered": ordered, "items": items, "bbox": bbox})
            continue
        if b["type"] == "paragraph":
            para = b["text"]
            bbox = b.get("bbox")
            i += 1
            while i < len(blocks) and blocks[i]["type"] == "paragraph":
                nxt = blocks[i]["text"]
                para = (para + " " + nxt).strip()
                bbox = union_bbox(bbox, blocks[i].get("bbox"))
                i += 1
            out.append({"type": "paragraph", "text": para.strip(), "bbox": bbox})
            continue
        out.append(b)
        i += 1
    return out

def is_plausible_table(rows: List[List[Optional[str]]]) -> bool:
    if not rows:
        return False
    nrows = len(rows)
    ncols = max(len(r) for r in rows) if rows else 0
    if nrows < 2 or ncols < 2:
        return False
    total = nrows * ncols
    non_empty = sum(1 for r in rows for c in r if (c or "").strip())
    if non_empty / total < 0.35:  # too sparse
        return False
    return True

def extract_tables(page: pdfplumber.page.Page) -> List[List[List[Optional[str]]]]:
    """Extract tables with lattice first, then stream; filter implausible ones."""
    tables: List[List[List[Optional[str]]]] = []

    def _normalize(tbl):
        return [[(c.strip() if isinstance(c, str) else ("" if c is None else str(c))) for c in row] for row in tbl]

    # Pass 1: lattice
    try:
        for tbl in page.extract_tables(TABLE_SETTINGS) or []:
            norm = _normalize(tbl)
            if is_plausible_table(norm):
                tables.append(norm)
    except Exception:
        pass

    # Pass 2: stream fallback (no settings)
    try:
        for tbl in page.extract_tables() or []:
            norm = _normalize(tbl)
            if is_plausible_table(norm):
                tables.append(norm)
    except Exception:
        pass

    return tables

# -----------------------------
# Core pipeline
# -----------------------------

def parse_pdf(path: str, heading_rules: HeadingHeuristics) -> Dict[str, Any]:
    doc: Dict[str, Any] = {
        "document": {
            "meta": {
                "source_pdf": os.path.abspath(path),
                "created_utc": dt.datetime.utcnow().isoformat() + "Z",
                "generator": "pdf→json/xml pipeline (pdfplumber)",
                "licenses": {"pdfplumber": "MIT", "pdfminer.six": "MIT"},
            },
            "pages": [],
        }
    }

    with pdfplumber.open(path) as pdf:
        for pageno, page in enumerate(pdf.pages, start=1):
            chars = page.chars
            median_size = estimate_font_median(chars)
            lines = group_chars_to_lines(chars)
            line_blocks: List[Dict[str, Any]] = []
            for line in lines:
                text, sz = line_text_and_size(line)
                if not text.strip():
                    continue
                line_blocks.append(classify_line(line, text, sz, median_size, heading_rules))

            blocks = coalesce_blocks(line_blocks)

            # tables
            for t in extract_tables(page):
                blocks.append({"type": "table", "rows": t, "bbox": None})

            doc["document"]["pages"].append({
                "page_number": pageno,
                "blocks": blocks,
            })
    return doc

# -----------------------------
# Serialization
# -----------------------------

def write_json(doc: Dict[str, Any], out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)

def json_to_xml(doc: Dict[str, Any]) -> Element:
    dmeta = doc["document"].get("meta", {})
    root = Element("document")
    root.set("source_pdf", str(dmeta.get("source_pdf", "")))
    created = dmeta.get("created_utc")
    if created:
        root.set("created_utc", created)

    def add_bbox(el, bb):
        if not bb:
            return
        x0,y0,x1,y1 = bb
        el.set("x0", f"{x0:.2f}")
        el.set("y0", f"{y0:.2f}")
        el.set("x1", f"{x1:.2f}")
        el.set("y1", f"{y1:.2f}")

    for page in doc["document"].get("pages", []):
        p_el = SubElement(root, "page")
        p_el.set("number", str(page.get("page_number", "")))
        for b in page.get("blocks", []):
            btype = b.get("type")
            if btype == "heading":
                h = SubElement(p_el, "heading")
                h.set("level", str(b.get("level", 1)))
                add_bbox(h, b.get("bbox"))
                h.text = b.get("text", "")
            elif btype == "paragraph":
                p = SubElement(p_el, "p")
                add_bbox(p, b.get("bbox"))
                p.text = b.get("text", "")
            elif btype == "list":
                lst = SubElement(p_el, "list")
                lst.set("ordered", "true" if b.get("ordered") else "false")
                add_bbox(lst, b.get("bbox"))
                for item in b.get("items", []):
                    it = SubElement(lst, "item")
                    it.text = item
            elif btype == "table":
                t = SubElement(p_el, "table")
                add_bbox(t, b.get("bbox"))
                for row in b.get("rows", []):
                    r = SubElement(t, "row")
                    for cell in row:
                        c = SubElement(r, "cell")
                        c.text = cell if cell is not None else ""
            else:
                g = SubElement(p_el, "block")
                g.set("kind", str(btype))
                add_bbox(g, b.get("bbox"))
                g.text = json.dumps(b, ensure_ascii=False)
    return root

def write_xml(doc: Dict[str, Any], out_path: str) -> None:
    root = json_to_xml(doc)
    tree = ElementTree(root)
    tree.write(out_path, encoding="utf-8", xml_declaration=True)

# -----------------------------
# CLI
# -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Parse PDF → JSON & XML (no AGPL/GPL)")
    ap.add_argument("pdf", help="Path to input PDF")
    ap.add_argument("--out-base", required=False, default=None, help="Output base path without extension (default: alongside PDF)")
    ap.add_argument("--h1", type=float, default=HeadingHeuristics.h1_factor, help="Heading-1 factor vs median font size")
    ap.add_argument("--h2", type=float, default=HeadingHeuristics.h2_factor, help="Heading-2 factor vs median font size")
    ap.add_argument("--h3", type=float, default=HeadingHeuristics.h3_factor, help="Heading-3 factor vs median font size")
    ap.add_argument("--minpt", type=float, default=HeadingHeuristics.min_abs_pt, help="Minimum absolute font size for headings")
    ap.add_argument("--minlen", type=int, default=HeadingHeuristics.min_len, help="Minimum text length for headings")
    ap.add_argument("--minalpha", type=int, default=HeadingHeuristics.min_alpha, help="Minimum alphabetic chars for headings")
    args = ap.parse_args()

    if args.out_base is None:
        base, _ = os.path.splitext(args.pdf)
        out_base = base
    else:
        out_base = args.out_base
    os.makedirs(os.path.dirname(out_base) or ".", exist_ok=True)

    hh = HeadingHeuristics(args.h1, args.h2, args.h3, args.minpt, args.minlen, args.minalpha)
    doc = parse_pdf(args.pdf, hh)

    json_path = out_base + ".json"
    xml_path = out_base + ".xml"

    write_json(doc, json_path)
    write_xml(doc, xml_path)
    
    with open(json_path) as f:
        json_generated = f.read()
    with open(json_path, "w") as f:
        json.dump(json_correction_llm(json_generated), f)
    with open(xml_path) as f:
        xml_generated = f.read()
    with open(xml_path, "w") as f:
        f.write(xml_correction_llm(xml_generated))

    # QA summary
    pages = len(doc["document"]["pages"])
    blocks = sum(len(p.get("blocks", [])) for p in doc["document"]["pages"])
    headings = sum(1 for p in doc["document"]["pages"] for b in p.get("blocks", []) if b.get("type") == "heading")
    tables = sum(1 for p in doc["document"]["pages"] for b in p.get("blocks", []) if b.get("type") == "table")
    lists = sum(1 for p in doc["document"]["pages"] for b in p.get("blocks", []) if b.get("type") == "list")
    print(f"Wrote: {json_path}\nWrote: {xml_path}")
    print(f"QA — pages: {pages}, blocks: {blocks}, headings: {headings}, lists: {lists}, tables: {tables}")

if __name__ == "__main__":
    main()
