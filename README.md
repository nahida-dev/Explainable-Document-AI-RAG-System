📄 Explainable Document AI (RAG System)



An explainable, production-style \*\*Retrieval-Augmented Generation (RAG)\*\* system that combines \*\*LLM-based generation\*\* with \*\*hybrid information retrieval\*\* to answer grounded questions over long, structured documents such as research papers, technical reports, manuals, and policies.



The system emphasizes \*\*accuracy, explainability, and production readiness\*\*, providing page-level citations and transparent retrieval decisions.



✨ Key Features



📑 Section-aware document chunking

Preserves document structure using heading detection and paragraph-aware chunking.



🔎 Hybrid retrieval

* Dense semantic search using embeddings
* Sparse keyword search using BM25
* Candidate merging with weighted scoringg for precision



🎯 Relevance optimization

* Lightweight reranking for improved precision
* Conceptual vs. reference query detection
* Penalties for TOC / list-of-tables noise



📌 Explainable answers

* Page-level citations
* Highlighted source snippets
* Transparent scoring and retrieval rationale



🌐 UI-ready APIs

* JSON API for programmatic access
* HTML endpoint for clean, user-friendly UI integration



🧠 AI \& LLM Capabilities

* Retrieval-Augmented Generation (RAG) architecture
* LLM-grounded answer generation with strict context constraints
* Hybrid search (semantic embeddings + BM25)
* Hallucination mitigation via context-only generation
* Deterministic extractive fallback when LLM APIs are unavailable



🧱 System Architecture

flowchart TD

&nbsp;   User\[User / UI] -->|Question| API\[FastAPI Backend]



&nbsp;   API --> HR\[Hybrid Retrieval Engine]



&nbsp;   HR --> Dense\[Dense Search<br/>Semantic Embeddings]

&nbsp;   HR --> Sparse\[BM25 Keyword Search]



&nbsp;   Dense --> Merge\[Candidate Union]

&nbsp;   Sparse --> Merge



&nbsp;   Merge --> Filter\[Filtering \& Reranking]

&nbsp;   Filter --> Context\[Relevant Context Chunks]



&nbsp;   Context --> LLM\[LLM Generator<br/>(Optional)]

&nbsp;   Context --> Extractive\[Extractive Answer]



&nbsp;   LLM --> Answer\[Answer + Citations]

&nbsp;   Extractive --> Answer



&nbsp;   Answer --> User





🛠️ Tech Stack

AI / LLM Stack

* Retrieval-Augmented Generation (RAG)
* Sentence Transformers (dense embeddings)
* BM25 (sparse retrieval)
* OpenAI-compatible LLM APIs
* Vector database (ChromaDB)



Backend

* Python
* FastAPI
* RESTful APIs



NLP / IR Techniques

* Semantic search
* Hybrid retrieval
* Lightweight reranking
* Context grounding





🚀 Getting Started

1️⃣ Clone the repository

git clone https://github.com/<your-username>/<repo-name>.git

cd <repo-name>



2️⃣ Create and activate a virtual environment

python -m venv .venv

source .venv/bin/activate  # Windows: .venv\\Scripts\\activate



3️⃣ Install dependencies

pip install -r requirements.txt



4️⃣ Add your document



Place a PDF in the data/ directory:



data/

&nbsp;└── document.pdf





(PDFs are ignored by git by default.)



5️⃣ Run the server

uvicorn app:app --reload



📡 API Endpoints

POST /ingest



Ingests and indexes a document.



{}





Optional:



{ "pdf\_path": "./data/document.pdf" }



POST /ask



Ask a question about the document.



{

&nbsp; "query": "What problem does this document address?"

}





Returns:

* Answer
* Page-level citations
* Scoring breakdown

Explainability metadata



GET /ui



Simple web UI for interactive Q\&A.





🧪 Example Questions



* What problem does this document address?
* How does the proposed approach work?
* What are the main differences from existing methods?
* What assumptions does the system make?
* What are the limitations discussed?
* Summarize the key contributions.



🧩 Key Design Decisions

* Used hybrid retrieval (dense + sparse) to support both conceptual questions and exact references

(tables, equations, identifiers).

* Implemented section-aware chunking to avoid splitting definitions and formulas.
* Added reranking and TOC/list penalties to reduce noise common in long technical documents.
* Enforced context-only generation to improve reliability and reduce hallucinations.
* Designed APIs to be stateless, modular, and UI-ready.



🚀 Production Readiness

* Modular retrieval pipeline
* Stateless FastAPI services
* Pluggable LLM providers
* Graceful degradation when LLM APIs are unavailable
* Built-in explainability and auditability



🎯 Use Cases

* Research paper understanding
* Technical documentation Q\&A
* Policy and compliance review
* Knowledge base exploration
* Enterprise document intelligence



🧩 Project Philosophy



Accuracy > Explainability > Speed



The goal is not just to answer questions, but to ensure users can verify where answers come from and why they were selected.



📄 License



MIT License



🙌 Acknowledgements



Inspired by modern RAG architectures and real-world challenges in trustworthy AI and information retrieval.



⭐ If you find this useful



Feel free to star ⭐ the repository or fork it to adapt for your own document intelligence workflows.

