![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-Production-green?logo=fastapi)
![RAG](https://img.shields.io/badge/RAG-Hybrid%20Retrieval-blueviolet)
![LLM](https://img.shields.io/badge/LLM-OpenAI%20Compatible-black?logo=openai)
![VectorDB](https://img.shields.io/badge/VectorDB-ChromaDB-red)
![Explainable AI](https://img.shields.io/badge/XAI-Explainable%20AI-important)
![Production Ready](https://img.shields.io/badge/Status-Production--Style-success)


# Explainable Document AI – RAG System

An **industry-oriented, production-style Retrieval-Augmented Generation (RAG) system**
for structured technical documents such as research papers, manuals, policies, and reports.

This project demonstrates how to build an **explainable, citation-grounded AI assistant**
using **FastAPI, embeddings, hybrid retrieval (Dense + BM25), and optional LLM generation**.

---

## 🚀 Key Features

- 📄 **Section-aware document chunking**
- 🔍 **Hybrid Retrieval**
  - Semantic search (Sentence Transformers)
  - Keyword search (BM25)
  - Lightweight reranking
- 📌 **Page-level citations**
- 🧠 **Explainable retrieval pipeline**
- ⚡ **FastAPI backend**
- 🎨 **UI-ready endpoints (JSON / HTML-ready)**
- 🔐 Works **with or without OpenAI API key**
- 🏭 Designed with **production patterns** in mind

---

## 🧠 Why This Project Matters

This system reflects **real-world LLM engineering practices** used in:

- Enterprise Knowledge Assistants
- AI Search Engines
- Internal Documentation Q&A
- Research & Compliance Systems
- Trustworthy / Auditable AI

It focuses on **accuracy, traceability, and explainability** — not just raw generation.

---

## 🏗️ Architecture Overview

```
User Query
   ↓
Hybrid Retrieval
 ├─ Dense Vector Search (Embeddings)
 ├─ Sparse BM25 Keyword Search
 ├─ Lightweight Reranking
 └─ Noise Filtering (TOC / Lists)
   ↓
Top-K Evidence Chunks
   ↓
Answer Generation
 ├─ LLM (if API key present)
 └─ Extractive fallback (no hallucinations)
   ↓
Final Answer + Page Citations
```

---

## 🛠️ Tech Stack

- **Backend**: FastAPI
- **Vector Store**: ChromaDB
- **Embeddings**: Sentence-Transformers (MiniLM)
- **Sparse Search**: BM25
- **LLM (Optional)**: OpenAI GPT models
- **PDF Parsing**: PyPDF
- **Language**: Python 3.10+

---

## 📂 Project Structure

```
Explainable-Document-AI-RAG-System/
│
├── app.py                # Main FastAPI application
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
├── .gitignore
│
└── data/
    ├── document.pdf      # Your input document (ignored by git)
    └── README.md         # Instructions for data folder
```

---

## ▶️ How to Run

### 1️⃣ Create Virtual Environment

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ (Optional) Set OpenAI API Key

```bash
setx OPENAI_API_KEY "your_api_key_here"
```

Restart terminal after setting.

### 4️⃣ Start Server

```bash
python -m uvicorn app:app --reload
```

Open browser:

```
http://127.0.0.1:8000/ui
```

---

## 🔌 API Endpoints

### Ingest Document
```http
POST /ingest
```
Reads and indexes `data/document.pdf`

### Ask a Question
```http
POST /ask
```
Example payload:
```json
{
  "query": "What is the proposed architecture?"
}
```

### Debug BM25
```http
GET /debug_bm25
```

---

## 🧩 Example Question Types

- “What problem does the framework address?”
- “Explain the system architecture.”
- “How does the proposed method differ from prior work?”
- “Which section discusses evaluation?”

This system is **domain-agnostic** — usable for:
- Legal documents
- Technical manuals
- Research papers
- Policies
- Whitepapers

---

## 🔍 Explainability

Each answer includes:
- Supporting document excerpts
- Page numbers
- Section titles
- Retrieval scores (optional)

This ensures **trust, auditability, and zero hallucination**.

---

## 📌 Use Cases

- Enterprise AI Search
- Research Assistants
- Compliance QA
- Internal Documentation Bots
- Academic / Technical Review Tools

---

## 📜 License

MIT License

---

## 👩‍💻 Author

**Nahida Chowdhury**  
AI / ML Engineer | LLM & RAG Systems  

GitHub: https://github.com/nahida-dev
