🤖 Amazon Platform Chatbot
“Local RAG-based chatbot for Amazon buyers and sellers powered by embeddings, FAISS vector search, and local LLM inference with Mistral via Ollama.”

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![FAISS](https://img.shields.io/badge/FAISS-VectorSearch-orange)
![SentenceTransformers](https://img.shields.io/badge/SentenceTransformer-Embeddings-purple)
![Ollama](https://img.shields.io/badge/Ollama-Mistral_7B-black)
![RAG](https://img.shields.io/badge/Architecture-RAG-brightgreen)


This chatbot delivers context-aware answers to buyer and seller queries on Amazon by combining vector search retrieval with local LLM generation and fallback handling.

text
              ┌────────────────────────────┐
              │      User Query Input       │
              └─────────────┬──────────────┘
                            │
                            ▼
          ╔══════════════════════════╗
          ║  SentenceTransformer     ║ ← User query semantically embedded locally
          ║  (MiniLM L6 v2 Model)    ║
          ╚══════════════════════════╝
                            │
                            ▼
          ╔══════════════════════════╗
          ║     FAISS Vector Index   ║ ← Retrieves top-N relevant help doc sentences
          ╚══════════════════════════╝
                            │
                            ▼
          ╔══════════════════════════╗
          ║   Contextual Prompt Gen  ║ ← Builds prompt from retrieved sentences +
          ║     + User Query         ║   fallback instructions if no retrieval
          ╚══════════════════════════╝
                            │
                            ▼
          ╔══════════════════════════╗
          ║    Local LLM (Mistral)   ║ ← Generates a professional, stepwise answer
          ║       via Ollama          ║   strictly relying on provided context
          ╚══════════════════════════╝
                            │
                            ▼
          ╔══════════════════════════╗
          ║   Verification via LLM   ║ ← Checks if answer is grounded in context 
          ╚══════════════════════════╝
                            │
                            ▼
              ┌─────────────┴─────────────┐
              │                           │
        Answer verified               Answer NOT verified  
      (returns AI response)      (returns fallback and triggers support modal)
              │                           │
              ▼                           ▼
       ┌─────────────┐            ┌───────────────┐
       │  Frontend UI│            │ Support &      │
       │  Chat Panel │            │ Feedback Forms │
       └─────────────┘            └───────────────┘
🛠️ Core Features
🔧 Feature	📖 Description
RAG Architecture	Retrieval-Augmented Generation combining local semantic search with local LLM answer generation
Semantic Search	Sentence embeddings with MiniLM L6 + FAISS vector DB for efficient nearest neighbor search
Local LLM Inference	Runs Mistral 7B locally via Ollama CLI for zero-API-cost generation
Fallback Handling	Strict fallback message if no relevant context or answer fails verification, prompting contact support
Answer Verification	Uses LLM to verify answer is strictly based on retrieved context, rejecting hallucinations
Web UI Chat Interface	Responsive chat with buyer & seller context, feedback rating, and support request modals
Context Management	Upload and ingestion pipeline for refreshing help doc embeddings and FAISS index
Support & Feedback	User feedback collection and support requests stored as local files for team follow-up
📚 Explanation of Context
All chatbot answers rely strictly on a knowledge base of help document sentences — about 100 lines of Amazon buyer and seller instructions, navigation tips, and FAQs. Using semantic search, the system retrieves the most relevant lines for each query, providing clear, explicit context for the LLM to answer confidently and professionally, citing source line numbers. This allows the chatbot to avoid hallucinated answers, ensuring trustworthy help.

🚩 Why RAG?
Retrieval-Augmented Generation (RAG) enables the chatbot to:

🔍 Retrieve the most relevant help document snippets for the user’s query, grounding responses in factual content.

🤖 Pass this context plus the user question to a local LLM (Mistral via Ollama) to generate a coherent answer.

✋ Fall back with a fixed polite message and support option if retrieval is insufficient or the answer is unverifiable, ensuring a safe user experience.

▶️ Sample Fallback Behavior
When the system cannot retrieve context above a confidence threshold, it does not guess but returns:
"Sorry, I cannot answer that from the provided document. Would you like to contact support?"

If the LLM answer fails strict verification for grounding in the retrieved context, the same fallback triggers.

The web UI automatically shows a support form on fallback to let users submit detailed queries for human assistance.

🚀 Quick Start
1️⃣ Prerequisites
Python 3.8+

Git

Ollama CLI installed & Mistral 7B model pulled locally

Local MiniLM L6 v2 SentenceTransformer model downloaded

2️⃣ Setup Ollama & Mistral Model
text
# Install Ollama from https://ollama.com/download  
# Then pull the Mistral model for local inference:
ollama pull mistral
3️⃣ Install Python Dependencies
bash
git clone https://github.com/yourusername/amazon-platform-chatbot.git
cd amazon-platform-chatbot
pip install -r requirements.txt
4️⃣ Prepare Sentence Transformer Model
Run once to cache MiniLM-L6 locally:

python
from sentence_transformers import SentenceTransformer
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
Alternatively, configure and use your preferred local path in the embeddings.py.

5️⃣ Ingest Help Document to Build Index
Place amazon_help_doc.txt in project root and run:

bash
python ingest.py
This builds the FAISS index and stores embeddings, metadata for semantic retrieval.

6️⃣ Run the FastAPI Chat Server
bash
uvicorn main:APP --reload
Access the chatbot UI locally at http://127.0.0.1:8000

📂 Folder Structure
text
amazon-platform-chatbot/
│
├── embeddings.py       # Sentence Transformer embedding cache/load
├── generator.py        # Build prompts and generate answers with Ollama Mistral
├── ingest.py           # Ingest help doc and build FAISS index
├── retrieval.py        # FAISS semantic search logic
├── verifier.py         # Verify answers grounding strictness
├── main.py             # FastAPI app + chat UI code
├── amazon_help_doc.txt # Help document with buyer/seller instructions
├── requirements.txt    # Python dependencies
└── storage/            # FAISS index, embeddings, metadata files after ingestion
⭐ Features Summary
💬 Buyer & Seller Support: Stepwise directions starting from Amazon homepage for both user types.

🔍 Semantic Retrieval: MiniLM embeddings + FAISS for fast similarity search.

🤖 Local LLM Generation: Mistral 7B model running offline via Ollama CLI.

⚠️ Fallback Handling: Polite fixed message + auto support modal for unanswered queries.

✅ Answer Verification: Ensures answers rely ONLY on retrieved help document for trustworthiness.

📝 User Feedback & Support: In-UI star rating, feedback comments, and detailed support request submission.

🔄 Dynamic Context Upload: Ability to update help docs and rebuild FAISS index without redeploy.

🤝 Need Help or Want to Contribute?
Ollama Official Site

Sentence-Transformers Documentation

FAISS Github

FastAPI Docs and Tutorials

If you find bugs or want support features, contributions are welcome!

📧 Contact
Email: velamalapavankrishna@gmail.com

Thank you for exploring this local-first, trustworthy Amazon chatbot powered by state-of-the-art embeddings and LLM technology! 🚀✨
