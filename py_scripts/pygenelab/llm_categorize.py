"""
LLM-based pathway label categorization.

Groups MSigDB-style pathway names (GOBP_*, REACTOME_*, KEGG_*, ...) into a
small set of KEGG-BRITE-inspired top-level functional categories using the
Anthropic Claude API. Designed for downstream Sankey-style summaries where
many enriched pathways need to collapse onto a handful of buckets.
"""

from __future__ import annotations

import json
import os
import re
from typing import Iterable

import pandas as pd


# ---------------------------------------------------------------------------
# Category vocabulary (KEGG BRITE top-level + "Other" fallback)
# ---------------------------------------------------------------------------
PATHWAY_TOP_CATEGORIES = [
    "Metabolism",
    "Genetic Information Processing",
    "Environmental Information Processing",
    "Cellular Processes",
    "Organismal Systems",
    "Human Diseases",
    "Other",
]

_DB_PREFIX_RE = re.compile(
    r"(?i)^(GOBP|GOMF|GOCC|REACTOME|KEGG|KEGG_MEDICUS|HALLMARK|WP|BIOCARTA|PID|"
    r"GRAESSMANN|TABULA_MURIS_SENIS|MIR|HP|CHR|GTRD)_(.+)$"
)


def strip_pathway_label(name: str) -> str:
    """Remove the leading source-DB token (GOBP_, REACTOME_, KEGG_, ...) and
    convert remaining underscores to spaces.

    GOBP_PROTEIN_FOLDING -> "PROTEIN FOLDING"
    REACTOME_MITOTIC_PROMETAPHASE -> "MITOTIC PROMETAPHASE"
    """
    if not isinstance(name, str):
        return ""
    m = _DB_PREFIX_RE.match(name)
    stripped = m.group(2) if m else name
    return stripped.replace("_", " ").strip()


def extract_gene_set(name: str) -> str:
    """Return the source-DB token (GOBP, REACTOME, ...) or 'UNKNOWN'."""
    if not isinstance(name, str):
        return "UNKNOWN"
    m = _DB_PREFIX_RE.match(name)
    return m.group(1).upper() if m else "UNKNOWN"


# ---------------------------------------------------------------------------
# Anthropic API call (batched + prompt-cached)
# ---------------------------------------------------------------------------
_SYSTEM_TEMPLATE = (
    "You are a curator that assigns biological pathway names to one of these "
    "top-level functional categories (KEGG BRITE convention):\n"
    "{categories}\n\n"
    "Guidance:\n"
    "- Metabolism: anabolic/catabolic pathways, biosynthesis, energy metabolism, "
    "lipid/amino-acid/carbohydrate/nucleotide metabolism, OXPHOS, TCA, glycolysis.\n"
    "- Genetic Information Processing: DNA replication/repair, transcription, "
    "translation, RNA processing, ribosome biogenesis, chromatin, proteostasis.\n"
    "- Environmental Information Processing: signal transduction, receptor "
    "signaling (MAPK/PI3K/Wnt/Notch/TGFb/etc.), membrane transport, ECM-receptor.\n"
    "- Cellular Processes: cell cycle, apoptosis, autophagy, cytoskeleton, "
    "motility, vesicle trafficking, cell adhesion/junctions, senescence.\n"
    "- Organismal Systems: immune, nervous, endocrine, circulatory, digestive, "
    "muscular, skeletal, reproductive, sensory, development.\n"
    "- Human Diseases: cancer, neurodegeneration, infections, metabolic disease, "
    "cardiovascular disease, immune disease pathways named after a disease.\n"
    "- Other: only when no category fits.\n\n"
    "Output requirement: respond with a single JSON object mapping each given "
    "pathway name (verbatim, as provided) to exactly one category string from "
    "the list above. No prose, no markdown fences."
)


def _build_system_text(categories: Iterable[str]) -> str:
    return _SYSTEM_TEMPLATE.format(
        categories="\n".join(f"- {c}" for c in categories)
    )


def _parse_json_object(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            return {}
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return {}


def categorize_pathways_via_llm(
    pathway_names: Iterable[str],
    *,
    model: str = "claude-haiku-4-5-20251001",
    batch_size: int = 60,
    max_tokens: int = 4096,
    categories: Iterable[str] = PATHWAY_TOP_CATEGORIES,
    api_key: str | None = None,
    verbose: bool = True,
) -> dict[str, str]:
    """
    Classify each pathway name into one of `categories` via the Anthropic API.

    Uses prompt caching on the system prompt so per-batch cost is small.
    Returns {pathway_name: category}. Names that the model omits or labels
    with an unknown string fall back to "Other".

    Requires the `anthropic` package and ANTHROPIC_API_KEY (or `api_key`).
    """
    from anthropic import Anthropic

    client = Anthropic(api_key=api_key) if api_key else Anthropic()
    cats = list(categories)
    valid_cats = set(cats)
    system_text = _build_system_text(cats)

    names = [n for n in pathway_names if isinstance(n, str) and n]
    out: dict[str, str] = {}

    for i in range(0, len(names), batch_size):
        batch = names[i:i + batch_size]
        user_text = (
            "Pathway names to categorize:\n"
            + "\n".join(f"- {n}" for n in batch)
            + "\n\nReturn only the JSON object."
        )
        msg = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=[{
                "type": "text",
                "text": system_text,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": user_text}],
        )
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        parsed = _parse_json_object(text)
        for name in batch:
            cat = parsed.get(name)
            if cat not in valid_cats:
                cat = "Other"
            out[name] = cat
        if verbose:
            print(f"  categorized {min(i + batch_size, len(names))}/{len(names)}")

    return out


# ---------------------------------------------------------------------------
# High-level helper: filter a plot_df and categorize
# ---------------------------------------------------------------------------
def categorize_enriched_plot_df(
    plot_df: pd.DataFrame,
    *,
    shared: Iterable[str] | None = None,
    direction: str = "Up",
    term_col: str = "Term",
    group_col: str = "Group",
    score_col: str = "score",
    pval_col: str = "pval",
    model: str = "claude-haiku-4-5-20251001",
    api_key: str | None = None,
    categories: Iterable[str] = PATHWAY_TOP_CATEGORIES,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Filter `plot_df` to the shared pathways enriched in `direction` (Up/Down),
    strip DB prefixes, classify each pathway via the Claude API, and return a
    tidy DataFrame ready for Sankey plotting.

    Output columns:
        Term            – original MSigDB pathway name
        stripped_label  – DB prefix removed, underscores -> spaces
        Gene_set        – source DB (GOBP / REACTOME / KEGG / ...)
        category        – top-level functional bucket from the LLM
        Group           – "Up" or "Down"
        score, pval, -log10(pval)
    """
    df = plot_df.copy()
    if shared is not None:
        df = df[df[term_col].isin(set(shared))]
    if direction is not None and group_col in df.columns:
        df = df[df[group_col].astype(str) == direction]

    if df.empty:
        return df.assign(stripped_label=[], Gene_set=[], category=[])

    df = df.copy()
    df["stripped_label"] = df[term_col].map(strip_pathway_label)
    df["Gene_set"] = df[term_col].map(extract_gene_set)

    name_to_cat = categorize_pathways_via_llm(
        df[term_col].tolist(),
        model=model,
        api_key=api_key,
        categories=categories,
        verbose=verbose,
    )
    df["category"] = df[term_col].map(name_to_cat).fillna("Other")

    keep_cols = [term_col, "stripped_label", "Gene_set", "category"]
    if group_col in df.columns:
        keep_cols.append(group_col)
    for c in (score_col, pval_col, "-log10(pval)"):
        if c in df.columns:
            keep_cols.append(c)

    df = df[keep_cols].reset_index(drop=True)
    sort_cols = [c for c in ("category", "-log10(pval)") if c in df.columns]
    if sort_cols:
        df = df.sort_values(
            sort_cols,
            ascending=[True] + [False] * (len(sort_cols) - 1),
        ).reset_index(drop=True)
    return df
