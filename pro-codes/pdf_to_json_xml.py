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
import difflib
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

DESTRUCTURE_PROMPT = """You are a strict document-structure corrector for parsed PDFs.

INPUT:
- A JSON object with a single page: { "page_number": int, "blocks": [ ... ] }.
- Each block has: { "type": "paragraph" | "heading" | "list" | "table", ... }.
- Sometimes a paragraph improperly contains an inlined, flattened table (header + data) plus real narrative sentences.
- A nearby proper table block for the SAME content may also exist.

TASK:
1) Do NOT invent content. Only use tokens present in the input blocks.
2) If a paragraph contains both real narrative and table residue:
   - Preserve the narrative sentences as a clean paragraph.
   - Remove the duplicated table residue (header/data tokens).
3) If the paragraph begins with a title-like phrase (e.g., “Safety Summary”, “Efficacy Results”), convert that phrase into a separate heading block (level=2 or 3). Keep the remaining sentences as a paragraph.
4) If a proper table block exists for the same content, DO NOT alter its cell values and DO NOT duplicate the table in text.
5) Keep original block order as much as possible, except for splitting a paragraph into [heading?, paragraph].
6) Preserve existing 'bbox' fields when splitting text; use the paragraph's bbox for the new heading and paragraph; leave table bbox unchanged.
7) If the document-structure requires no correction, return the document as is.
8) Return JSON ONLY in this schema:

{
  "page_number": <int>,
  "blocks": [
    { "type": "heading", "level": 2|3, "text": "<...>", "bbox": [ ... ] }?,
    { "type": "paragraph", "text": "<...>", "bbox": [ ... ] }?,
    { "type": "table", "rows": [[...],[...],...], "bbox": <null or [ ... ]> }?,
    ...
  ],
  "notes": {
    "changed_blocks": <int>,           // how many blocks were modified or split
    "removed_table_residue": true|false,
    "explanations": ["short notes..."]
  }
}

Rules:
- NEVER add new rows/cells or change numbers in tables.
- NEVER produce extra keys or different schema.
- If nothing needs fixing, return the input unchanged (same schema)."""

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

def destructure_llm(block, retries: int=0, max_retries: int=3):
    if not isinstance(block, str):
        block_string = json.dumps(block)
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {os.environ['togetherai_api_key']}"}
    payload = {"messages": [{"role": "system", "content": DESTRUCTURE_PROMPT}, {"role": "user", "content": block_string}], "model": "openai/gpt-oss-20b"}
    try:
        r = requests.post(os.environ['togetherai_api_endpoint'], json=payload, headers=headers)
        try:
            return json.loads(strip_json_fencing(r.json()["choices"][0]["message"]["content"]))
        except json.JSONDecodeError as exp:
            return json.loads(strip_json_fencing(fix_json(strip_json_fencing(r.json()["choices"][0]["message"]["content"]), str(exp))))
    except Exception as exp:
        if retries >= max_retries:
            raise RuntimeError("Max retries reached.")
        return destructure_llm(block, retries+1)
        
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

BULLET_RE = re.compile(r"^\s*([•·◦\-\*\u2022\u25CF\u25E6]|\(?[0-9ivxlcdm]+\)|[0-9]+[\.)]|[A-Za-z]\.)\s+", re.IGNORECASE)
MD_TABLE_SEP = re.compile(r"^\s*\|.*\|\s*$")

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

def remove_duplicates(page_blocks):
    tables_present = []
    i = 0
    while i < len(page_blocks):
        entry = page_blocks[i]
        if entry in tables_present:
            page_blocks.pop(i)
        else:
            tables_present.append(entry)
            i += 1
    return page_blocks
    
def remove_doc_duplicates(doc):
    try:
        for page_entry in doc['document']['pages']:
            remove_duplicates(page_entry['blocks'])
    except KeyError:
        pass
    return doc

import re
from typing import List

def _md_escape(s: str) -> str:
    # Escape pipes; keep other chars verbatim
    return (s or "").replace("|", r"\|").strip()

def _looks_header(row: List[str]) -> bool:
    """
    Heuristic: header if all cells are non-empty AND
    at least half are non-numeric-ish.
    """
    if not row or any(c.strip() == "" for c in row):
        return False
    def non_numeric(c: str) -> bool:
        c = c.strip()
        return not re.fullmatch(r"[0-9.,%()\-+\s]+", c)
    nonnum = sum(1 for c in row if non_numeric(c))
    return nonnum >= max(1, len(row) // 2)

def _split_if_single_col(row: List[str]) -> List[str]:
    """
    If we got a 'table' row with a single fat cell (common when upstream
    lost cell boundaries), try splitting by 2+ spaces or tabs.
    """
    if len(row) != 1:
        return row
    txt = row[0]
    # If there are pipes already, respect them.
    if "|" in txt:
        return [c.strip() for c in txt.strip().strip("|").split("|")]
    # Split on runs of 2+ spaces or tabs
    parts = re.split(r"(?: {2,}|\t+)", txt.strip())
    # Require at least 2 columns to accept the split
    return [p.strip() for p in parts] if len(parts) >= 2 else row

def _normalize_table_rows(rows: List[List[str]]) -> List[List[str]]:
    # Drop fully empty rows; split single-wide rows opportunistically
    cleaned = []
    for r in rows:
        r = [ (c if c is not None else "").strip() for c in r ]
        if len(r) == 1:
            r = _split_if_single_col(r)
        if any(c.strip() for c in r):
            cleaned.append(r)

    if not cleaned:
        return []

    # Make rectangular: pad to max columns
    max_cols = max(len(r) for r in cleaned)
    rect = [ (r + [""]*(max_cols - len(r))) for r in cleaned ]

    # Trim trailing empty columns
    # If the last column is empty across all rows, drop it (repeat)
    while rect and all((row and row[-1].strip() == "") for row in rect) and len(rect[0]) > 1:
        rect = [row[:-1] for row in rect]

    return rect

def table_to_md(rows: List[List[str]]) -> str:
    """
    Robust Markdown table:
      - makes rows rectangular
      - synthesizes a header if needed
      - escapes pipes; converts newlines in cells to <br>
    """
    rect = _normalize_table_rows(rows)
    if not rect:
        return ""

    # Convert embedded newlines to <br> to avoid breaking Markdown tables
    rect = [[_md_escape(c).replace("\n", "<br>") for c in r] for r in rect]

    # If first row doesn't look like a header, synthesize one
    header = rect[0]
    body = rect[1:]
    if not _looks_header(header):
        header = [f"Col {i+1}" if (not c) else c for i, c in enumerate(header)]
        # Keep original first row as body
        body = rect

    sep = ["---"] * len(header)
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(sep)    + " |",
    ]
    for r in body:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)

def doc_to_markdown(doc: dict, page_separator: str = "\n\n---\n\n") -> str:
    """
    Render the normalized JSON schema to Markdown.
    More defensive table handling; everything else unchanged.
    """
    out_pages = []
    for page in doc.get("document", {}).get("pages", []):
        chunks = []
        for b in page.get("blocks", []):
            t = b.get("type")
            if t == "heading":
                lvl = min(3, int(b.get("level", 3)))
                text = (b.get("text") or "").strip()
                if text:
                    chunks.append("#" * lvl + " " + text)
            elif t == "paragraph":
                text = (b.get("text") or "").strip()
                if text:
                    chunks.append(text)
            elif t == "list":
                items = b.get("items") or []
                ordered = bool(b.get("ordered"))
                for i, it in enumerate(items, 1):
                    it = (it or "").strip()
                    if not it:
                        continue
                    prefix = f"{i}. " if ordered else "- "
                    chunks.append(prefix + it)
            elif t == "table":
                rows = b.get("rows") or []
                md = table_to_md(rows)
                if md:
                    chunks.append(md)
            else:
                # Graceful fallback
                text = (b.get("text") or b.get("raw") or "").strip()
                if text:
                    chunks.append(text)
        out_pages.append("\n\n".join(chunks))
    return page_separator.join(out_pages)

_WS_COL_SPLIT = re.compile(r"(?:\s{2,}|\t+)")
_PIPE_ROW = re.compile(r"^\s*\|.*\|\s*$")

def _norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()

def _tokenize(s: str) -> List[str]:
    return re.findall(r"[a-z0-9%]+", _norm_text(s))

def _jaccard(a: List[str], b: List[str]) -> float:
    if not a or not b:
        return 0.0
    A, B = set(a), set(b)
    inter = len(A & B)
    union = len(A | B)
    return inter / union if union else 0.0

def _looks_like_joined_header(p: str, header_cells: List[str]) -> bool:
    p_norm = _norm_text(p)
    # try pipe-joined
    if "|".join([c.strip() for c in header_cells]).lower() in p_norm.replace(" ", ""):
        return True
    # try explicit pipes
    if _PIPE_ROW.match(p):  # already looks like a markdown table row
        return True
    # try whitespace columns
    parts = [x for x in _WS_COL_SPLIT.split(p.strip()) if x.strip()]
    if len(parts) >= max(2, len(header_cells) // 2):
        # if many parts match header cells, count it
        hits = sum(1 for part in parts for hc in header_cells if _norm_text(hc) == _norm_text(part))
        if hits >= max(1, len(header_cells) // 2):
            return True
    return False

def _is_duplicate_para_of_header(para_text: str, header_cells: List[str],
                                 jaccard_thr: float = 0.75, fuzzy_thr: float = 0.82) -> bool:
    p_tok = _tokenize(para_text)
    h_tok = _tokenize(" ".join(header_cells))
    jac = _jaccard(p_tok, h_tok)
    if jac >= jaccard_thr:
        return True
    ratio = difflib.SequenceMatcher(None, _norm_text(para_text), _norm_text(" ".join(header_cells))).ratio()
    if ratio >= fuzzy_thr:
        return True
    if _looks_like_joined_header(para_text, header_cells):
        return True
    return False

def suppress_flat_header_paragraphs(doc: Dict) -> Tuple[Dict, Dict]:
    """
    Remove paragraph blocks that duplicate the header of a nearby table.
    Returns (new_doc, stats).
    """
    removed = 0
    inspected = 0
    for page in doc.get("document", {}).get("pages", []):
        blocks = page.get("blocks", [])
        keep: List[Dict] = []
        i = 0
        while i < len(blocks):
            b = blocks[i]
            if b.get("type") == "paragraph":
                inspected += 1
                # look ahead for a table within 2 positions
                found_tbl = None
                for j in range(i+1, min(i+3, len(blocks))):
                    if blocks[j].get("type") == "table":
                        found_tbl = blocks[j]
                        break
                if found_tbl:
                    rows = found_tbl.get("rows") or []
                    header = rows[0] if rows else []
                    if header and _is_duplicate_para_of_header(b.get("text", ""), header):
                        removed += 1
                        i += 1
                        continue  # drop this paragraph
            keep.append(b)
            i += 1
        page["blocks"] = keep
    stats = {"paragraphs_inspected": inspected, "paragraphs_removed": removed}
    return doc, stats

def parse_pdf(path: str, heading_rules: HeadingHeuristics, use_llm: bool=True) -> Dict[str, Any]:
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
    doc = remove_doc_duplicates(doc)
    doc, stats = suppress_flat_header_paragraphs(doc)
    print("[dedupe]", stats)
    if use_llm:
        try:
            for i, page_entry in enumerate(doc["document"]["pages"]):
                if any(block_entry.get('type', 'N/A') == 'table' for block_entry in page_entry['blocks']):
                    print(f"Fixing table block in page: {i+1}")
                    doc["document"]["pages"][i] = destructure_llm(page_entry)
        except KeyError:
            pass
    return doc, doc_to_markdown(doc)

def lexical_heading_ok(text: str, hh) -> bool:
    """
    Decide if a text line is a heading based on lexical heuristics only.
    - At least min_len characters
    - At least min_alpha alphabetic characters
    - Allow short headings ending with ':' if allow_short_if_colon is True
    - Bias toward ALL CAPS or Title Case with few words
    """
    s = text.strip()
    if not s:
        return False
    if len(s) < hh.min_len and not (hh.allow_short_if_colon and s.endswith(":")):
        return False
    if sum(ch.isalpha() for ch in s) < hh.min_alpha:
        return False
    if s.endswith(":"):
        return True
    if s.isupper() and len(s) >= hh.min_len:
        return True
    if s.istitle() and len(s.split()) <= 8:
        return True
    return False

def as_heading(text: str, level: int) -> Dict:
    return {"type": "heading", "level": level, "text": text.strip()}


def as_paragraph(text: str) -> Dict:
    return {"type": "paragraph", "text": text.strip()}


def as_list(items: List[str], ordered: bool) -> Dict:
    return {"type": "list", "ordered": ordered, "items": [i.strip() for i in items if i.strip()]}


def as_table(rows: List[List[str]]) -> Dict:
    norm = [[("" if c is None else str(c)).strip() for c in row] for row in rows]
    return {"type": "table", "rows": norm}

def size_to_level(size: float, median: float, hh) -> Optional[int]:
    """
    Decide heading level based on font size relative to median.
    Expects hh to be a HeadingHeuristics object with h1_factor, h2_factor, h3_factor, min_abs_pt.
    """
    if size >= max(hh.min_abs_pt, median * hh.h1_factor):
        return 1
    if size >= max(hh.min_abs_pt, median * hh.h2_factor):
        return 2
    if size >= max(hh.min_abs_pt, median * hh.h3_factor):
        return 3
    return None

def parse_pdf_pymupdf(path: str, heading_rules: HeadingHeuristics, use_llm: bool=True) -> Dict[str, Any]:
    import fitz  # PyMuPDF

    doc = {
        "document": {
            "meta": {
                "source_pdf": os.path.abspath(path),
                "created_utc": dt.datetime.utcnow().isoformat() + "Z",
                "generator": "pdf→json/xml pipeline (pymupdf)",
                "licenses": {"pymupdf": "AGPL-3 (commercial license available)"},
            },
            "pages": [],
        }
    }

    with fitz.open(path) as pdf:
        for pageno, page in enumerate(pdf, 1):
            # --- text lines (unchanged from your current version) ---
            td = page.get_text("dict")
            sizes, lines = [], []
            for b in td.get("blocks", []):
                for l in b.get("lines", []):
                    spans = l.get("spans", [])
                    if not spans: 
                        continue
                    text = "".join(s.get("text", "") for s in spans).strip()
                    if not text:
                        continue
                    mean_sz = sum(s.get("size", 0) for s in spans) / len(spans)
                    sizes.append(mean_sz)
                    lines.append((text, mean_sz))

            median = (sorted(sizes)[len(sizes)//2] if sizes else 0.0)
            blocks = []
            for text, sz in lines:
                lvl = size_to_level(sz, median, heading_rules)
                if lvl and lexical_heading_ok(text, heading_rules):
                    blocks.append({"type": "heading", "level": lvl, "text": text})
                elif BULLET_REGEX.match(text):
                    ordered = bool(re.match(r"^\s*(\(?[0-9ivxlcdm]+\)|[0-9]+[.)]|[A-Za-z]\.)\s+", text, re.I))
                    blocks.append({"type": "list", "ordered": ordered, "items": [BULLET_REGEX.sub("", text).strip()]})
                else:
                    blocks.append({"type": "paragraph", "text": text})

            # --- tables via PyMuPDF ---
            try:
                # strategy options: "lines", "lines_strict", "text"
                tab_finder = page.find_tables(strategy="lines")
                for tbl in tab_finder.tables or []:
                    rows = tbl.extract()  # -> list[list[str]]
                    if rows and len(rows) >= 2 and max(len(r) for r in rows) >= 2:
                        blocks.append({"type": "table", "rows": rows})
            except Exception:
                # fallback to no tables if detection fails
                pass

            doc["document"]["pages"].append({"page_number": pageno, "blocks": blocks})

    doc = remove_doc_duplicates(doc)
    doc, stats = suppress_flat_header_paragraphs(doc)
    print("[dedupe]", stats)
    if use_llm:
        try:
            for i, page_entry in enumerate(doc["document"]["pages"]):
                if any(block_entry.get('type', 'N/A') == 'table' for block_entry in page_entry['blocks']):
                    print(f"Fixing table block in page: {i+1}")
                    doc["document"]["pages"][i] = destructure_llm(page_entry)
        except KeyError:
            pass
    return doc, doc_to_markdown(doc)


def parse_pdf_pdfminer(path: str, heading_rules: HeadingHeuristics) -> Dict[str, Any]:
    from pdfminer.high_level import extract_text

    doc: Dict[str, Any] = {
        "document": {
            "meta": {
                "source_pdf": os.path.abspath(path),
                "created_utc": dt.datetime.utcnow().isoformat() + "Z",
                "generator": "pdf→json/xml pipeline (pdfminer.six)",
                "licenses": {"pdfminer.six": "MIT"},
            },
            "pages": [],
        }
    }

    def classify_text_line(text: str) -> Dict[str, Any]:
        s = text.strip()
        if not s:
            return {"type": "paragraph", "text": s}
        # light lexical heading heuristic (since we lack font sizes)
        looks_heading = (
            (s.endswith(":") and len(s) >= heading_rules.min_len) or
            (sum(ch.isalpha() for ch in s) >= heading_rules.min_alpha and
             (s.isupper() or s.istitle()) and len(s) >= heading_rules.min_len)
        )
        if looks_heading:
            return {"type": "heading", "level": 3, "text": s}
        if BULLET_REGEX.match(s):
            ordered = bool(re.match(r"^\s*(\(?[0-9ivxlcdm]+\)|[0-9]+[.)]|[A-Za-z]\.)\s+", s, re.I))
            return {"type": "list", "ordered": ordered, "items": [BULLET_REGEX.sub("", s).strip()]}
        return {"type": "paragraph", "text": s}

    full = extract_text(path) or ""
    pages = full.split("\x0c")  # pdfminer inserts form feed between pages

    for pageno, page_text in enumerate(pages, start=1):
        raw_lines = [ln for ln in (page_text or "").splitlines()]
        blocks: List[Dict[str, Any]] = []
        # simple paragraph coalescing by blank lines
        buf: List[str] = []
        def flush_para():
            nonlocal buf
            if buf:
                blocks.append({"type": "paragraph", "text": " ".join(x.strip() for x in buf).strip()})
                buf = []

        for line in raw_lines:
            if not line.strip():
                flush_para()
                continue
            # classify single lines that clearly look like headings/bullets; else buffer as paragraph
            if BULLET_REGEX.match(line) or line.strip().endswith(":"):
                flush_para()
                blocks.append(classify_text_line(line))
            else:
                # try heading first
                maybe = classify_text_line(line)
                if maybe["type"] == "heading":
                    flush_para()
                    blocks.append(maybe)
                elif maybe["type"] == "list":
                    flush_para()
                    blocks.append(maybe)
                else:
                    buf.append(line)
        flush_para()
        doc["document"]["pages"].append({"page_number": pageno, "blocks": blocks})

    doc = remove_doc_duplicates(doc)
    doc, stats = suppress_flat_header_paragraphs(doc)
    print("[dedupe]", stats)
    return doc, doc_to_markdown(doc)


def parse_pdf_pypdf2(path: str, heading_rules: HeadingHeuristics) -> Dict[str, Any]:
    import PyPDF2

    doc: Dict[str, Any] = {
        "document": {
            "meta": {
                "source_pdf": os.path.abspath(path),
                "created_utc": dt.datetime.utcnow().isoformat() + "Z",
                "generator": "pdf→json/xml pipeline (PyPDF2)",
                "licenses": {"PyPDF2": "BSD-3-Clause"},
            },
            "pages": [],
        }
    }

    def classify_text_line(text: str) -> Dict[str, Any]:
        s = text.strip()
        if not s:
            return {"type": "paragraph", "text": s}
        looks_heading = (
            (s.endswith(":") and len(s) >= heading_rules.min_len) or
            (sum(ch.isalpha() for ch in s) >= heading_rules.min_alpha and
             (s.isupper() or s.istitle()) and len(s) >= heading_rules.min_len)
        )
        if looks_heading:
            return {"type": "heading", "level": 3, "text": s}
        if BULLET_REGEX.match(s):
            ordered = bool(re.match(r"^\s*(\(?[0-9ivxlcdm]+\)|[0-9]+[.)]|[A-Za-z]\.)\s+", s, re.I))
            return {"type": "list", "ordered": ordered, "items": [BULLET_REGEX.sub("", s).strip()]}
        return {"type": "paragraph", "text": s}

    reader = PyPDF2.PdfReader(path)
    for pageno, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        raw_lines = text.splitlines()
        blocks: List[Dict[str, Any]] = []
        buf: List[str] = []

        def flush_para():
            nonlocal buf
            if buf:
                blocks.append({"type": "paragraph", "text": " ".join(x.strip() for x in buf).strip()})
                buf = []

        for line in raw_lines:
            if not line.strip():
                flush_para()
                continue
            if BULLET_REGEX.match(line) or line.strip().endswith(":"):
                flush_para()
                blocks.append(classify_text_line(line))
            else:
                maybe = classify_text_line(line)
                if maybe["type"] in ("heading", "list"):
                    flush_para(); blocks.append(maybe)
                else:
                    buf.append(line)
        flush_para()
        doc["document"]["pages"].append({"page_number": pageno, "blocks": blocks})

    doc = remove_doc_duplicates(doc)
    doc, stats = suppress_flat_header_paragraphs(doc)
    print("[dedupe]", stats)
    return doc, doc_to_markdown(doc)


def parse_pdf_pypdfium2(path: str, heading_rules: HeadingHeuristics) -> Dict[str, Any]:
    import pypdfium2 as pdfium

    doc: Dict[str, Any] = {
        "document": {
            "meta": {
                "source_pdf": os.path.abspath(path),
                "created_utc": dt.datetime.utcnow().isoformat() + "Z",
                "generator": "pdf→json/xml pipeline (pypdfium2)",
                "licenses": {"pypdfium2": "BSD-3-Clause"},
            },
            "pages": [],
        }
    }

    def classify_text_line(text: str) -> Dict[str, Any]:
        s = text.strip()
        if not s:
            return {"type": "paragraph", "text": s}
        looks_heading = (
            (s.endswith(":") and len(s) >= heading_rules.min_len) or
            (sum(ch.isalpha() for ch in s) >= heading_rules.min_alpha and
             (s.isupper() or s.istitle()) and len(s) >= heading_rules.min_len)
        )
        if looks_heading:
            return {"type": "heading", "level": 3, "text": s}
        if BULLET_REGEX.match(s):
            ordered = bool(re.match(r"^\s*(\(?[0-9ivxlcdm]+\)|[0-9]+[.)]|[A-Za-z]\.)\s+", s, re.I))
            return {"type": "list", "ordered": ordered, "items": [BULLET_REGEX.sub("", s).strip()]}
        return {"type": "paragraph", "text": s}

    pdf = pdfium.PdfDocument(path)
    try:
        for pageno in range(len(pdf)):
            page = pdf.get_page(pageno)
            textpage = page.get_textpage()
            text = textpage.get_text_range() or ""
            textpage.close(); page.close()

            raw_lines = text.splitlines()
            blocks: List[Dict[str, Any]] = []
            buf: List[str] = []

            def flush_para():
                nonlocal buf
                if buf:
                    blocks.append({"type": "paragraph", "text": " ".join(x.strip() for x in buf).strip()})
                    buf = []

            for line in raw_lines:
                if not line.strip():
                    flush_para()
                    continue
                if BULLET_REGEX.match(line) or line.strip().endswith(":"):
                    flush_para()
                    blocks.append(classify_text_line(line))
                else:
                    maybe = classify_text_line(line)
                    if maybe["type"] in ("heading", "list"):
                        flush_para(); blocks.append(maybe)
                    else:
                        buf.append(line)
            flush_para()
            doc["document"]["pages"].append({"page_number": pageno+1, "blocks": blocks})
    finally:
        pdf.close()
    doc, stats = suppress_flat_header_paragraphs(doc)
    print("[dedupe]", stats)
    doc = remove_doc_duplicates(doc)

    return doc, doc_to_markdown(doc)

def text_only_segment(
    lines: List[str],
    hh: Optional[HeadingHeuristics] = None,
) -> List[Dict[str, Any]]:
    """
    Segment plain text lines into blocks:
      - blank line → paragraph break
      - Markdown headings:  # / ## / ###
      - bullets / ordered lists (•, -, *, 1., a., i., etc.)
      - Markdown tables: lines starting/continuing with pipes
      - fallback lexical headings (ALL-CAPS/Title + trailing ':')

    Returns a list of blocks in your common schema.
    """
    if hh is None:
        hh = HeadingHeuristics()

    blocks: List[Dict[str, Any]] = []
    buf: List[str] = []

    HEADING_MD = re.compile(r"^(#{1,6})\s+(.*)$")
    ORDERED_PREFIX = re.compile(r"^\s*(?:\(?[0-9ivxlcdm]+\)|[0-9]+[.)]|[A-Za-z][.)])\s+", re.IGNORECASE)

    def flush_para():
        nonlocal buf
        if buf:
            para = " ".join(s.strip() for s in buf).strip()
            if para:
                blocks.append({"type": "paragraph", "text": para})
            buf = []

    def lexical_heading_ok(text: str) -> bool:
        s = text.strip()
        if not s:
            return False
        if len(s) < hh.min_len and not (hh.allow_short_if_colon and s.endswith(":")):
            return False
        if sum(ch.isalpha() for ch in s) < hh.min_alpha:
            return False
        # Gentle bias toward headings when text is ALL CAPS or Title Case, or ends with colon
        if s.endswith(":"):
            return True
        if s.isupper() and len(s) >= hh.min_len:
            return True
        if s.istitle() and len(s.split()) <= 8:
            return True
        return False

    i = 0
    N = len(lines)
    while i < N:
        line = (lines[i] or "").rstrip("\n")

        # Paragraph break
        if not line.strip():
            flush_para()
            i += 1
            continue

        # Markdown heading (#, ##, ### ...)
        m = HEADING_MD.match(line)
        if m:
            flush_para()
            level = min(3, len(m.group(1)))
            text = m.group(2).strip()
            if text:
                blocks.append({"type": "heading", "level": level, "text": text})
            i += 1
            continue

        # Markdown table block (one or more consecutive pipe-lines)
        if MD_TABLE_SEP.match(line):
            flush_para()
            tbl_lines = [line]
            i += 1
            while i < N and MD_TABLE_SEP.match(lines[i] or ""):
                tbl_lines.append((lines[i] or "").rstrip("\n"))
                i += 1
            # Simple split by pipe; caller can post-process if needed
            rows = [
                [c.strip() for c in row.strip().strip("|").split("|")]
                for row in tbl_lines
            ]
            if rows:
                blocks.append({"type": "table", "rows": rows})
            continue

        # Bullet / ordered lists: consume consecutive bullet lines
        if BULLET_RE.match(line):
            flush_para()
            items: List[str] = []
            # detect ordered vs unordered from the first item
            ordered = bool(ORDERED_PREFIX.match(line))
            while i < N and (lines[i] or "").strip() and BULLET_RE.match(lines[i] or ""):
                item_text = BULLET_RE.sub("", (lines[i] or "")).strip()
                if item_text:
                    items.append(item_text)
                i += 1
            if items:
                blocks.append({"type": "list", "ordered": ordered, "items": items})
            continue

        # Fallback: treat standalone “heading-like” lines as headings
        if lexical_heading_ok(line):
            flush_para()
            blocks.append({"type": "heading", "level": 3, "text": line.strip()})
            i += 1
            continue

        # Otherwise, accumulate into a paragraph
        buf.append(line)
        i += 1

    flush_para()
    return blocks

def parse_markdown_string(md: str) -> List[Dict[str, Any]]:
    lines = md.splitlines()
    return text_only_segment(lines)

def make_doc_meta(src_pdf: str) -> Dict[str, Any]:
    return {
    "source_pdf": os.path.abspath(src_pdf),
    "created_utc": dt.datetime.utcnow().isoformat() + "Z",
    "generator": "multi-backend pdf→json/xml",
    }

def new_doc(src_pdf: str) -> Dict[str, Any]:
    return {"document": {"meta": make_doc_meta(src_pdf), "pages": []}}

def save_markdown(md: str, out_base: str, backend: str):
    md_path = out_base + f".{backend}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"[+] wrote markdown {md_path}")
    return md_path

def parse_with_docling(path: str, hh: HeadingHeuristics) -> Tuple[Dict[str, Any], Optional[str]]:
    from docling.document_converter import DocumentConverter
    conv = DocumentConverter()
    result = conv.convert(path)
    md = result.document.export_to_markdown()
    blocks = parse_markdown_string(md)
    doc = new_doc(path)
    doc["document"]["pages"].append({"page_number": 1, "blocks": blocks})
    return doc, md

def parse_with_markitdown(path: str, hh: HeadingHeuristics) -> Tuple[Dict[str, Any], Optional[str]]:
    from markitdown import MarkItDown
    md = MarkItDown().convert(path).text_content
    blocks = parse_markdown_string(md)
    doc = new_doc(path)
    doc["document"]["pages"].append({"page_number": 1, "blocks": blocks})
    return doc, md

def parse_with_markdrop(path: str, hh):
    """
    Markdrop backend. Writes Markdown/HTML to an output dir; we read the .md back.
    WARNING: License appears inconsistent (PyPI=MIT, repo=GPL-3.0). Verify before use.
    Returns (doc, md_str).
    """

    import os, glob, subprocess, shlex, tempfile
    from pathlib import Path
    
    outdir = Path(tempfile.mkdtemp(prefix="markdrop_out_"))
    md_text = None

    # Try Python API first
    try:
        from markdrop import markdrop as md_convert
        from markdrop import MarkDropConfig
        cfg = MarkDropConfig()  # you can tune image/table options here
        html_path = md_convert(path, str(outdir), cfg)  # writes .md in outdir
    except Exception:
        # Fallback to CLI
        cmd = f"markdrop convert {shlex.quote(path)} --output_dir {shlex.quote(str(outdir))}"
        try:
            subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except Exception as e:
            raise RuntimeError(f"markdrop failed: {e}")

    # Find the Markdown file Markdrop wrote
    mds = sorted(glob.glob(str(outdir / "*.md")))
    if not mds:
        raise RuntimeError(f"markdrop produced no .md in {outdir}")
    md_file = mds[0]
    with open(md_file, "r", encoding="utf-8") as f:
        md_text = f.read()

    # Parse Markdown → blocks using your existing segmenter
    blocks = parse_markdown_string(md_text)
    doc = new_doc(path)
    doc["document"]["pages"].append({"page_number": 1, "blocks": blocks})
    return doc, md_text

def parse_with_marker(path: str, hh: HeadingHeuristics):
    """
    Marker backend (marker-pdf). Converts to Markdown (default), then we parse MD → blocks.
    Requires: pip install marker-pdf (and PyTorch). License: GPL (code) + OpenRAIL-M (weights).
    """
    try:
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
        from marker.output import text_from_rendered
    except Exception as e:
        raise RuntimeError(f"marker not installed or import failed: {e}")

    # Build converter; you can pass config to get JSON/HTML directly if preferred.
    converter = PdfConverter(artifact_dict=create_model_dict())
    rendered = converter(path)  # default output_format=markdown
    md, _meta, _images = text_from_rendered(rendered)

    # Reuse your Markdown -> blocks segmenter
    blocks = parse_markdown_string(md)
    doc = new_doc(path)
    doc["document"]["pages"].append({"page_number": 1, "blocks": blocks})
    return doc, md  # keep returning md so you can --save-md

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

backend_map = {
        "pdfplumber": parse_pdf,
        "pymupdf": parse_pdf_pymupdf,
        "pypdf2": parse_pdf_pypdf2,
        "pdfminer": parse_pdf_pdfminer,
        "pypdfium2": parse_pdf_pypdfium2,
        "docling": parse_with_docling,
        "markitdown": parse_with_markitdown,
        "markdrop": parse_with_markdrop,
        "markerpdf": parse_with_marker,
    }

# -----------------------------
# CLI
# -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Parse PDF → JSON & XML (no AGPL/GPL)")
    ap.add_argument("pdf", help="Path to input PDF")
    ap.add_argument("--out-base", required=False, default=None, help="Output base path without extension (default: alongside PDF)")
    ap.add_argument("--h1", type=float, default=HeadingHeuristics.h1_factor, help=f"Heading-1 factor vs median font size (default: {HeadingHeuristics.h1_factor})")
    ap.add_argument("--h2", type=float, default=HeadingHeuristics.h2_factor, help=f"Heading-2 factor vs median font size (default: {HeadingHeuristics.h2_factor})")
    ap.add_argument("--h3", type=float, default=HeadingHeuristics.h3_factor, help=f"Heading-3 factor vs median font size (default: {HeadingHeuristics.h3_factor})")
    ap.add_argument("--minpt", type=float, default=HeadingHeuristics.min_abs_pt, help=f"Minimum absolute font size for headings (default: {HeadingHeuristics.min_abs_pt})")
    ap.add_argument("--minlen", type=int, default=HeadingHeuristics.min_len, help=f"Minimum text length for headings (default: {HeadingHeuristics.min_len})")
    ap.add_argument("--minalpha", type=int, default=HeadingHeuristics.min_alpha, help=f"Minimum alphabetic chars for headings (default: {HeadingHeuristics.min_alpha})")
    ap.add_argument("--backend", type=str, default="pdfplumber", help="Available backends: {backends}".format(backends="\n".join(f"- {backend_name}" for backend_name in backend_map)))
    # ap.add_argument("--use_llm", action=argparse.BooleanOptionalAction)
    args = ap.parse_args()

    if args.out_base is None:
        base, _ = os.path.splitext(args.pdf)
        out_base = base
    else:
        out_base = args.out_base
    os.makedirs(os.path.dirname(out_base) or ".", exist_ok=True)

    hh = HeadingHeuristics(args.h1, args.h2, args.h3, args.minpt, args.minlen, args.minalpha)

    pdf_parse_function = backend_map.get(args.backend, backend_map["pdfplumber"])

    try:
        print(f"Using backend engine: {pdf_parse_function.__name__}")
    except AttributeError:
        print(f"Using backend engine: {pdf_parse_function}")

    doc, md = pdf_parse_function(args.pdf, hh)

    json_path = out_base + ".json"
    xml_path = out_base + ".xml"
    md_path = out_base + ".md"

    try:
        os.mkdir(args.backend + "_output")
    except FileExistsError:
        pass

    write_json(doc, os.path.join(args.backend + "_output", json_path))
    write_xml(doc, os.path.join(args.backend + "_output", xml_path))
    
    with open(os.path.join(args.backend + "_output", md_path), "w") as f:
        f.write(md)
        print(f"Written Markdown to: {md_path}")

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
