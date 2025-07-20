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
- Jupyter notebook tutorial with code + explanations


---
