import os
import json
from io import BytesIO

import streamlit as st
import pandas as pd                                      
import chardet                                           
import tabula                                            
from PyPDF2 import PdfReader                             

import faiss                                             
import numpy as np                                       
import openai                                           

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder  
from transformers import AutoTokenizer
from openai import OpenAI                    

# ─── 0) CONFIG & AUTH ─────────────────────────────────────────────────────────
load_dotenv()
st.sidebar.title("🔑 OpenRouter API Key")
router_key = st.sidebar.text_input(
    "Optional override", type="password",
    placeholder="Paste your OpenRouter key here"
)
OPENROUTER_KEY = router_key.strip() or os.getenv("OPENROUTER_API_KEY","").strip()
if not OPENROUTER_KEY:
    st.sidebar.error("❗️ OpenRouter API key is required.")

# ─── 0b) LLM MODEL SELECTION ──────────────────────────────────────────────────
st.sidebar.title("🤖 LLM Model")
model_override = st.sidebar.text_input(
    "Model identifier:",
    placeholder="e.g. mistralai/mistral-small-3.2-24b-instruct:free"
)
DEFAULT_MODEL = "mistralai/mistral-small-3.2-24b-instruct:free"
MODEL_NAME = model_override.strip() or DEFAULT_MODEL
st.sidebar.caption(f"Using `{MODEL_NAME}` {'(default)' if not model_override else ''}")

# instantiate OpenRouter client
from openai import OpenAI
router = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_KEY
)

# ─── 1) SETTINGS ───────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Settings")

MAX_ROWS = st.sidebar.slider(
    "🔢 Max CSV rows to index", 100, 10000, 1000, step=100,
    help="Load at most this many rows from large CSVs to keep indexing fast."
)

CHUNK_SIZE  = st.sidebar.slider(
    "📄 Chunk size (tokens)", 100, 500, 200,
    help="Max number of tokens per text chunk. Smaller values mean finer-grained chunks (more precise retrieval) but a larger index."
)
st.sidebar.markdown("*(How many tokens each chunk can have. Smaller → more chunks, finer retrieval.)*")

TOP_K       = st.sidebar.slider(
    "🔍 top-K texts", 1, 10, 5,
    help="Number of top-retrieved chunks shown to the LLM. Higher values give more context but may include less relevant passages."
)
st.sidebar.markdown("*(How many retrieved snippets to pass to the model.)*")

HYDE_THRESH = st.sidebar.slider(
    "🤖 HyDE threshold", 0.1, 0.9, 0.3, step=0.05,
    help="Recall similarity cutoff below which a 'pseudo-answer' (HyDE) is generated to improve retrieval. Lower → fewer HyDE calls."
)
st.sidebar.markdown("*(Cosine similarity below which we ask the LLM to draft a hypothetical answer.)*")


# ─── 2) UPLOAD, PREVIEW & JSON CONVERSION ──────────────────────────────────────
uploaded = st.file_uploader("Upload a CSV or PDF", type=["csv", "pdf"])
if not uploaded:
    st.info("Please upload a CSV or PDF file to begin.")
    st.stop()

texts, metas = [], []
table_json = None  # will hold full-table JSON for numeric/date queries

if uploaded.type == "text/csv":
    st.subheader("📊 CSV Preview")
    raw = uploaded.read()
    enc = chardet.detect(raw).get("encoding", "utf-8")
    try:
        df = pd.read_csv(BytesIO(raw), encoding=enc, nrows=MAX_ROWS)
    except UnicodeDecodeError:
        df = pd.read_csv(BytesIO(raw), encoding="latin-1", nrows=MAX_ROWS)
    st.dataframe(df.head(5))

    # Convert entire CSV to JSON
    table_json = json.dumps(df.to_dict(orient="records"), indent=2)
    with st.expander("🔎 View CSV as JSON"):
        st.code(table_json, language="json")

    # Flatten rows for embedding
    for i, row in df.iterrows():
        txt = "; ".join(f"{col}: {row[col]}" for col in df.columns)
        texts.append(txt)
        metas.append(("csv", i))

else:
    st.subheader("📄 PDF Preview (first page snippet)")
    raw_pdf = uploaded.read()
    reader = PdfReader(BytesIO(raw_pdf))
    pages = [p.extract_text() or "" for p in reader.pages]
    st.text(pages[0][:500] + "…")

    # Try extracting tables via tabula
    try:
        with open("temp.pdf", "wb") as f:
            f.write(raw_pdf)
        dfs = tabula.read_pdf("temp.pdf", pages="all", multiple_tables=True)
        st.write(f"Detected {len(dfs)} tables")
    except Exception:
        dfs = []
        st.warning("⚠️ No tables found—falling back to text pages.")

    # Convert each table to JSON and flatten rows
    if dfs:
        full_tables = {}
        for ti, tdf in enumerate(dfs):
            # show head
            st.subheader(f"📋 Table {ti+1} Preview")
            st.dataframe(tdf.head(5))
            # JSON
            j = json.dumps(tdf.to_dict(orient="records"), indent=2)
            full_tables[f"table_{ti}"] = tdf.to_dict(orient="records")
            with st.expander(f"🔎 View Table {ti+1} as JSON"):
                st.code(j, language="json")
            # flatten rows
            for i, row in tdf.iterrows():
                txt = "; ".join(f"{c}: {row[c]}" for c in tdf.columns)
                texts.append(txt)
                metas.append(("pdf_table", i))
        table_json = json.dumps(full_tables, indent=2)

    # Fallback: treat each page as a chunk in JSON
    for idx, pg in enumerate(pages):
        if idx == 0:
            st.text(pg[:500] + "…")
        texts.append(pg)
        metas.append(("pdf_page", idx))
    if table_json is None:
        pages_list = [{"page": i+1, "text": pg[:2000]} for i, pg in enumerate(pages)]
        table_json = json.dumps({"pages": pages_list}, indent=2)

# ─── 3) CHUNKING & MODEL INIT ───────────────────────────────────────────────────
@st.cache_resource
def init_models():
    tok   = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2", use_fast=True)
    emb   = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    rer   = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")
    return tok, emb, rer

tokenizer, embedder, reranker = init_models()

def chunk_text(txt):
    toks = tokenizer.tokenize(txt)
    max_model = getattr(tokenizer, "model_max_length", 512)
    chunk_size = min(CHUNK_SIZE, max_model - 2)
    for i in range(0, len(toks), chunk_size):
        yield tokenizer.convert_tokens_to_string(toks[i:i+chunk_size])

with st.spinner("🔄 Chunking documents…"):
    chunks, chunk_meta = [], []
    for (kind, idx), doc in zip(metas, texts):
        for c in chunk_text(doc):
            chunks.append(c)
            chunk_meta.append((kind, idx))

# ─── 4) BUILD & CACHE FAISS INDEX ───────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def build_index(chunks):
    embs    = embedder.encode(chunks, convert_to_numpy=True).astype("float32")
    dim     = embs.shape[1]
    n_total = embs.shape[0]

    # If we don't have ≥ 39 points per centroid, skip IVF
    nlist = min(100, n_total)
    if n_total < nlist * 39:
        st.warning(
            f"Only {n_total} vectors for {nlist} clusters—"
            " using FlatL2 index instead of IVF to avoid poor k-means."
        )
        index = faiss.IndexFlatL2(dim)
        index.add(embs)
        return index

    # Otherwise build IVF normally
    quantizer = faiss.IndexFlatL2(dim)
    index     = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)
    index.train(embs)
    index.add(embs)
    index.nprobe = min(10, nlist)
    return index

with st.spinner("🔢 Building FAISS index…"):
    index = build_index(chunks)

# ─── 5) HYBRID RETRIEVE & ANSWER ────────────────────────────────────────────────
st.subheader("❓ Ask a question about your upload")
query = st.text_input("Enter your question:")
if st.button("Retrieve & Answer") and query:
    lowq = query.lower()

    # 5A) Numeric/Date/ID queries → Pandas or JSON-LLM
    if any(tok in lowq for tok in (
        "max","min","sum","average","count",
        "first","last","earliest","latest")):

        # CSV Pandas fallback
        if uploaded.type == "text/csv":
            numeric = df.select_dtypes(include=[np.number]).columns.to_list()
            datecols = df.select_dtypes(include=["datetime","datetime64"]).columns.to_list()
            # MAX example
            if "max" in lowq or "highest" in lowq:
                if numeric:
                    col = numeric[-1]
                    mv  = df[col].max()
                    row = df[df[col]==mv].iloc[0]
                    st.write(f"Highest {col} is {mv} (row id {row[df.columns[0]]}).")
                    st.stop()
            # FIRST/DATES example
            if "earliest" in lowq or "first" in lowq:
                if datecols:
                    dcol = datecols[0]
                    earliest = pd.to_datetime(df[dcol]).min()
                    st.write(f"Earliest {dcol} is {earliest.date()}.")
                    st.stop()

        # JSON-LLM fallback for other table queries
        all_records = json.loads(table_json)
        n_show = min(20, len(all_records))
        sample    = all_records[:n_show]

        with st.spinner(f"🧠 Calling JSON-LLM on {n_show} rows…"):
            snippet_json = json.dumps(sample, indent=2)
            prompt = (
                f"Here are the first {n_show} rows of the table as JSON:\n"
                f"```json\n{snippet_json}\n```\n"
                f"Question: {query}\n"
                f"Please answer based on these rows only."
            )
            resp = router.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role":"user","content":prompt}],
                temperature=0
            )

        # extract answer safely
        msg     = resp.choices[0].message
        answer  = getattr(msg, "content", None) \
                or getattr(msg, "reasoning_content", None) \
                or getattr(resp.choices[0], "text", "")

        st.subheader("💬 Answer (JSON snippet)")
        st.write(answer)
        st.stop()

    # 5B) Semantic RAG for free-text queries
    # bi-encoder retrieval
    q_emb = embedder.encode([query], convert_to_numpy=True).astype("float32")
    D, I  = index.search(q_emb, 20)

    # HyDE if recall low
    if D.max() < HYDE_THRESH:
        hyde_resp = router.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role":"user","content":f"Answer briefly: {query}"}],
            temperature=0.7, max_tokens=100
        )
        msg = hyde_resp.choices[0].message
        hyde_ans = getattr(msg, "content", None) \
                or getattr(msg, "reasoning_content", None) \
                or getattr(hyde_resp.choices[0], "text", "")
        q_emb = embedder.encode([hyde_ans], convert_to_numpy=True).astype("float32")
        D, I  = index.search(q_emb, 20)

    # cross-encoder rerank
    cands  = [chunks[i] for i in I[0]]
    metasK = [chunk_meta[i] for i in I[0]]
    scores = reranker.predict([[query, c] for c in cands])

    k = min(TOP_K, len(cands))
    if k == 0:
        st.warning("No chunks to retrieve—try a different query or upload.")
        st.stop()

    top_idxs   = np.argsort(scores)[-k:]
    top_chunks = [cands[i] for i in top_idxs]
    top_meta   = [metasK[i] for i in top_idxs]

    with st.expander("🔍 Retrieved Chunks", expanded=False):
        for txt, (kind, idx) in zip(top_chunks, top_meta):
            label = {
                "csv":       f"CSV row {idx}",
                "pdf_table": f"PDF table row {idx}",
                "pdf_page":  f"PDF page {idx+1}"
            }[kind]
            st.markdown(f"> **[{label}]** {txt}")

    # final generation
    context = "\n\n".join(top_chunks)
    prompt  = f"Context:\n{context}\n\nQuestion: {query}"
    with st.spinner("🧠 Generating with Mistral…"):
        gen = router.chat.completions.create(
            model="mistralai/mistral-small-3.2-24b-instruct:free",
            messages=[{"role":"user","content":prompt}],
            temperature=0.7, max_tokens=300
        )
    
    # Safe choice extraction
    choice = None
    if getattr(gen, "choices", None):
        choice = gen.choices[0]
    if not choice or not getattr(choice, "message", None):
        st.error("⚠️ The model didn’t return any text—try reducing your context (fewer rows or smaller JSON snippet).")
        st.stop()

    msg    = gen.choices[0].message
    answer = getattr(msg, "content", None) \
           or getattr(msg, "reasoning_content", None) \
           or getattr(gen.choices[0], "text", "")
    st.subheader("💬 Answer")
    st.write(answer)
