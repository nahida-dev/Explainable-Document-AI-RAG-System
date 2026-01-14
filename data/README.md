\# Data Directory



This directory is used to store \*\*input PDF documents\*\* for the Explainable Document RAG Assistant.



---



\## 📄 Supported Files



\- `document.pdf` (default)

\- Any other PDF specified via the `/ingest` API



The system is designed to work with \*\*long, structured documents\*\*, such as:

\- Research papers

\- Theses and dissertations

\- Technical manuals

\- Policy or compliance documents

\- Whitepapers and reports



---



\## 📥 How to Add a Document



1\. Place your PDF file in this directory:

data/document.pdf





2\. Ingest the document using the API:

```bash

curl -X POST http://127.0.0.1:8000/ingest



Or specify a custom path:



curl -X POST http://127.0.0.1:8000/ingest \\

&nbsp; -H "Content-Type: application/json" \\

&nbsp; -d '{"pdf\_path": "./data/your\_document.pdf"}'





⚠️ Important Notes

Do not commit PDF files to the repository

(they may be large, copyrighted, or private).



The repository is document-agnostic and works with any suitable PDF.



Only the structure and content of the document are used for retrieval and citation.



🔐 Privacy \& Safety

Documents are processed locally



No data is uploaded or shared externally by default



Suitable for private, academic, or enterprise use



ℹ️ This folder intentionally contains no PDF files in the public repository.

