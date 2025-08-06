# Upload-Driven-RAG

---

## Overview
A Streamlit app & notebook tutorial that lets users upload their own CSV or PDF, builds a FAISS-backed retrieval index, and answers questions via OpenRouter LLM 

![The streamlit app](https://github.com/audichandra/Upload-Driven-RAG/blob/main/results_img/qtest1.png)

Features: 
- Upload CSV or PDF (tables + free text)
- Automatic JSON conversion for exact numeric/date QA
- Chunking & SBERT embeddings + FAISS (Flat / IVF fallback)
- HyDE pseudo-answer recall boost
- Cross-encoder reranking
- Mistral LLM via OpenRouter for final generation
- Configurable sliders: row limit, chunk size, top-K, HyDE threshold
- Collapsible context preview & downloadable HTML report
- Jupyter notebook guide with code + explanations

---
## Installation 

1. Clone this repo
   ```
   git clone https://github.com/audichandra/Upload-Driven-RAG.git
   cd upload-driven-rag
   ```
2. Create & activate a virtual environment
   ```
   python -m venv .venv
   source .venv/bin/activate  # or .venv\Scripts\activate on Windows
   ```
3. Install dependencies
   ```
   pip install -r requirements.txt
   ```
4. Copy ```env.example``` to ```.env``` and fill in your ```OPENROUTER_API_KEY```

--- 
## Running the App
```
streamlit run app/upload_rag.py
```
- Open http://localhost:8501 in your browser.
- Upload a CSV or PDF, tune the sliders, ask questions, download results. (in the datasets folder, 4 dataset examples cover from csv table into a pdf with table)  

--- 
## Architecture & Flow
1. **Data Ingestion**

2. **JSON Conversion** for exact QA

3. **Chunking** (token limit enforcement)

4. **Embedding & FAISS Index** (Flat/IVF)

5. **Hybrid Retrieval**
   - Numeric/Date → Pandas or JSON snippet
   - Free-text → HyDE → FAISS → rerank → LLM

6. **Answer Generation** + Collapsible context

---
