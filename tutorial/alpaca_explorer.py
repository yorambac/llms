"""
Alpaca dataset explorer.
Run: streamlit run tutorial/alpaca_explorer.py
"""

import random
import re
import streamlit as st
from datasets import load_dataset

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Alpaca Explorer",
    page_icon="🦙",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.inst-box  { background:#0d1f33; border-left:4px solid #4a9eff; padding:12px 16px; border-radius:6px; margin-bottom:8px; color:#d0e8ff; }
.input-box { background:#0d2b1a; border-left:4px solid #3ddc84; padding:12px 16px; border-radius:6px; margin-bottom:8px; color:#c8f5d8; }
.resp-box  { background:#1e1030; border-left:4px solid #c084fc; padding:12px 16px; border-radius:6px; margin-bottom:8px; color:#e8d5ff; }
.label     { font-size:0.72rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase; opacity:0.85; margin-bottom:4px; }
.tag       { display:inline-block; background:#333; border-radius:4px; padding:2px 8px; font-size:0.75rem; margin:2px; }
</style>
""", unsafe_allow_html=True)

# ── Categories (keyword-based) ─────────────────────────────────────────────────

CATEGORIES = {
    "Coding":        ["code", "program", "function", "python", "javascript", "algorithm", "sql", "debug", "script", "implement"],
    "Writing":       ["write", "essay", "paragraph", "poem", "story", "letter", "draft", "compose", "summarize", "describe"],
    "Classification":["classify", "identify", "categorize", "label", "which", "odd one out", "determine"],
    "Math":          ["calculate", "solve", "equation", "math", "number", "sum", "average", "percentage", "geometry"],
    "Brainstorm":    ["brainstorm", "suggest", "ideas", "list", "ways to", "tips", "strategies", "recommend"],
    "Factual Q&A":   ["what is", "who is", "when did", "where is", "explain", "define", "history", "how does"],
    "Reasoning":     ["why", "compare", "difference", "advantage", "pros and cons", "analyze", "evaluate"],
    "Instruction":   ["steps", "how to", "instructions", "guide", "procedure", "recipe", "process"],
}

def detect_category(instruction: str) -> str:
    low = instruction.lower()
    for cat, keywords in CATEGORIES.items():
        if any(k in low for k in keywords):
            return cat
    return "Other"

# ── Load data ─────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading Alpaca dataset…")
def load_data():
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    rows = []
    for ex in ds:
        rows.append({
            "instruction": ex["instruction"],
            "input":       ex["input"],
            "output":      ex["output"],
            "has_input":   bool(ex["input"].strip()),
            "category":    detect_category(ex["instruction"]),
        })
    return rows

data = load_data()

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🦙 Alpaca Explorer")
    st.caption(f"{len(data):,} examples · tatsu-lab/alpaca")
    st.divider()

    st.subheader("Filter")
    search = st.text_input("Search instructions", placeholder="e.g. python, recipe, compare…")

    all_cats = ["All"] + sorted(CATEGORIES.keys()) + ["Other"]
    category = st.selectbox("Category", all_cats)

    has_input = st.radio("Has extra input?", ["Any", "Yes", "No"])

    st.divider()
    st.subheader("Display")
    per_page = st.select_slider("Examples per page", [5, 10, 20, 50], value=10)
    st.divider()

    # Stats
    st.subheader("Dataset stats")
    cat_counts = {}
    for r in data:
        cat_counts[r["category"]] = cat_counts.get(r["category"], 0) + 1
    for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
        pct = cnt / len(data) * 100
        st.markdown(f"`{cat}` — {cnt:,} ({pct:.0f}%)")

# ── Filter data ────────────────────────────────────────────────────────────────

filtered = data
if search:
    pat = search.lower()
    filtered = [r for r in filtered if pat in r["instruction"].lower() or pat in r["output"].lower()]
if category != "All":
    filtered = [r for r in filtered if r["category"] == category]
if has_input == "Yes":
    filtered = [r for r in filtered if r["has_input"]]
elif has_input == "No":
    filtered = [r for r in filtered if not r["has_input"]]

# ── Header ─────────────────────────────────────────────────────────────────────

col1, col2, col3 = st.columns([3, 1, 1])
with col1:
    st.header(f"{'🔍 ' if search or category != 'All' else ''}{len(filtered):,} examples"
              + (f" matching **{search}**" if search else "")
              + (f" in **{category}**" if category != "All" else ""))
with col2:
    if st.button("🎲 Random sample", use_container_width=True):
        st.session_state["page"] = random.randint(0, max(0, len(filtered) // per_page - 1))
with col3:
    sort_by = st.selectbox("Sort", ["Default", "Shortest response", "Longest response"], label_visibility="collapsed")

if sort_by == "Shortest response":
    filtered = sorted(filtered, key=lambda r: len(r["output"]))
elif sort_by == "Longest response":
    filtered = sorted(filtered, key=lambda r: -len(r["output"]))

# ── Pagination ─────────────────────────────────────────────────────────────────

total_pages = max(1, (len(filtered) + per_page - 1) // per_page)
if "page" not in st.session_state:
    st.session_state["page"] = 0
st.session_state["page"] = min(st.session_state["page"], total_pages - 1)

pcol1, pcol2, pcol3 = st.columns([1, 3, 1])
with pcol1:
    if st.button("◀ Prev", disabled=st.session_state["page"] == 0, use_container_width=True):
        st.session_state["page"] -= 1
        st.rerun()
with pcol2:
    st.markdown(f"<div style='text-align:center;padding-top:8px'>Page {st.session_state['page']+1} / {total_pages}</div>", unsafe_allow_html=True)
with pcol3:
    if st.button("Next ▶", disabled=st.session_state["page"] >= total_pages - 1, use_container_width=True):
        st.session_state["page"] += 1
        st.rerun()

st.divider()

# ── Examples ───────────────────────────────────────────────────────────────────

start = st.session_state["page"] * per_page
page_items = filtered[start: start + per_page]

if not page_items:
    st.info("No examples match your filters.")
else:
    for i, row in enumerate(page_items):
        idx = start + i
        cat = row["category"]

        with st.expander(f"**#{idx+1}** · `{cat}` · {row['instruction'][:90]}{'…' if len(row['instruction'])>90 else ''}", expanded=(per_page <= 5)):
            # Instruction
            st.markdown(f'<div class="inst-box"><div class="label">Instruction</div>{row["instruction"]}</div>', unsafe_allow_html=True)

            # Input (if present)
            if row["has_input"]:
                st.markdown(f'<div class="input-box"><div class="label">Input</div>{row["input"]}</div>', unsafe_allow_html=True)

            # Response
            st.markdown(f'<div class="resp-box"><div class="label">Response</div>{row["output"]}</div>', unsafe_allow_html=True)

            # Meta
            meta1, meta2, meta3 = st.columns(3)
            meta1.metric("Response length", f"{len(row['output'])} chars")
            meta2.metric("Has extra input", "Yes" if row["has_input"] else "No")
            meta3.metric("Category", cat)
