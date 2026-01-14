# data/ Directory

This directory is used to store **input documents** for the Explainable Document AI RAG System.

## 📄 Supported Documents

- PDF documents (research papers, reports, manuals, books, etc.)
- Example default file name:
  - `document.pdf`

> ⚠️ **PDF files are intentionally excluded from version control** via `.gitignore`.
> Do NOT commit private, copyrighted, or sensitive documents to GitHub.

---

## 📌 How This Folder Is Used

When you run the ingestion endpoint:

```bash
POST /ingest
```

The system:
1. Loads the PDF from this directory
2. Performs **section-aware chunking**
3. Builds:
   - Semantic embeddings (ChromaDB)
   - BM25 sparse index
4. Enables explainable, citation-backed question answering

---

## 🧪 Example Workflow

1. Place your document here:
   ```
   data/document.pdf
   ```

2. Ingest the document:
   ```bash
   curl -X POST http://127.0.0.1:8000/ingest
   ```

3. Ask questions:
   ```bash
   curl -X POST http://127.0.0.1:8000/ask \
     -H "Content-Type: application/json" \
     -d '{"query":"Explain the core methodology"}'
   ```

---

## 🔐 Security & Privacy Notes

- PDFs are **ignored by Git**
- This folder may contain:
  - Proprietary documents
  - Academic papers
  - Internal technical documentation
- Always verify permissions before sharing documents

---

## 🧠 Tip

This system is **document-agnostic**:
- Swap the PDF to use the system in **any domain**
  - Healthcare
  - Finance
  - Legal
  - Engineering
  - Research

No code changes required 🚀
