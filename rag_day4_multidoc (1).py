# ============================================================
# RAG DAY 4 — Multi-Document Support
# Goal: Upload multiple PDFs, switch between them,
#       search across all or one specific document
# Run each section in Jupyter Notebook cell by cell
# ============================================================

import os
import uuid
from datetime import datetime
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq
from fpdf import FPDF

load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "your-groq-key-here")


# ============================================================
# PART 1: Why Multi-Document?
# ============================================================

print("=" * 55)
print("PART 1: Why Multi-Document?")
print("=" * 55)
print("""
Single document RAG (Day 2-3):
  User uploads ONE PDF → asks questions about it
  Problem: real users have MULTIPLE documents

Multi-document RAG (today):
  User uploads MULTIPLE PDFs and can ask:
  → "What does Document 1 say about X?"
  → "What does Document 2 say about X?"
  → "Compare what both documents say about X?"

Key challenge: How do we know which chunks came
               from which document in ChromaDB?
Answer:        Metadata! Store doc_id in every chunk.
""")


# ============================================================
# PART 2: Create Multiple Test PDFs
# ============================================================

print("=" * 55)
print("PART 2: Creating Test Documents")
print("=" * 55)

def create_test_pdf(filename, content_pages):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    for content in content_pages:
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, content)
    pdf.output(filename)
    print(f"Created: {filename}")

# Document 1 — ML textbook
create_test_pdf("ml_textbook.pdf", [
    """Introduction to Machine Learning

Machine learning is a branch of artificial intelligence that enables
computers to learn from data without being explicitly programmed.
The field was pioneered by Arthur Samuel in 1959.

There are three main paradigms: supervised learning where models learn
from labeled examples, unsupervised learning where models find hidden
patterns, and reinforcement learning where agents learn from rewards.""",

    """Deep Learning and Neural Networks

Deep learning uses neural networks with multiple layers. CNNs excel
at image recognition. RNNs handle sequential data. Transformers have
revolutionized NLP since 2017.

Applications include image classification, speech recognition,
natural language processing, and autonomous vehicles.""",

    """Applications of Machine Learning

In healthcare ML models detect diseases from medical images.
In finance algorithms detect fraudulent transactions in real time.
Recommendation systems use collaborative filtering to suggest content.
Self driving cars use computer vision and reinforcement learning."""
])

# Document 2 — Python guide
create_test_pdf("python_guide.pdf", [
    """Python Programming Fundamentals

Python is a high-level language created by Guido van Rossum in 1991.
It emphasizes readability and simplicity. Python uses indentation
instead of curly braces. Variables are dynamically typed.""",

    """Python for Data Science

Key libraries include NumPy for numerical computing, Pandas for data
manipulation, and Matplotlib for visualization. Scikit-learn provides
machine learning tools. PyTorch and TensorFlow are deep learning
frameworks. Jupyter Notebooks allow interactive development.""",

    """Python Best Practices

Follow PEP 8 style guide. Functions should do one thing well.
Use meaningful variable names. Write docstrings for all functions.
Use virtual environments to isolate dependencies. Always pin
dependencies in requirements.txt for reproducibility."""
])

# Document 3 — Career guide
create_test_pdf("career_guide.pdf", [
    """Getting Started in AI/ML Careers

The AI job market is growing rapidly. Entry level positions require
Python skills, knowledge of ML algorithms, and practical projects.
Building a strong portfolio is essential. Employers want deployed
projects with real world applications, not just notebooks.""",

    """Essential Skills for AI Engineers

Technical skills: Python, PyTorch, TensorFlow, statistics, linear
algebra, and cloud platforms. Soft skills matter equally — explain
complex models to non-technical stakeholders. Continuous learning
is essential as the field evolves rapidly with new techniques."""
])

print("\nCreated 3 test documents:")
print("  1. ml_textbook.pdf  — 3 pages")
print("  2. python_guide.pdf — 3 pages")
print("  3. career_guide.pdf — 2 pages")


# ============================================================
# PART 3: MultiDocumentRAG Class
# ============================================================

print("\n" + "=" * 55)
print("PART 3: Multi-Document Manager")
print("=" * 55)

class MultiDocumentRAG:
    """
    RAG system supporting multiple documents.

    Design:
    - ONE ChromaDB collection for ALL documents
    - Each chunk stores doc_id in metadata
    - Filter by doc_id to search one document
    - No filter to search all documents
    """

    def __init__(self):
        print("Loading embedding model...")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

        # One collection for everything
        self.client = chromadb.PersistentClient(path="./multi_doc_db")
        try:
            self.client.delete_collection("all_documents")
        except:
            pass
        self.collection = self.client.create_collection("all_documents")

        # Track uploaded documents
        self.documents = {}     # { doc_id: info_dict }
        self.histories = {}     # { doc_id: [messages] }
        self.global_history = []

        self.groq = Groq(api_key=GROQ_API_KEY)
        print("✅ MultiDocumentRAG ready!")


    def add_document(self, pdf_path, doc_name=None):
        """Add a PDF. Returns doc_id."""
        doc_id   = str(uuid.uuid4())[:8]
        doc_name = doc_name or os.path.basename(pdf_path)

        print(f"\nAdding: {doc_name} (ID: {doc_id})")

        # Load + chunk
        loader   = PyPDFLoader(pdf_path)
        pages    = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=50
        )
        chunks = splitter.split_documents(pages)

        # Embed + store
        texts      = [c.page_content for c in chunks]
        embeddings = self.embedder.encode(texts)

        # 🔑 KEY: store doc_id in every chunk's metadata
        self.collection.add(
            documents  = texts,
            embeddings = embeddings.tolist(),
            metadatas  = [
                {
                    "doc_id":   doc_id,
                    "doc_name": doc_name,
                    "page":     c.metadata.get("page", 0) + 1
                }
                for c in chunks
            ],
            ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
        )

        # Register
        self.documents[doc_id] = {
            "name":  doc_name,
            "pages": len(pages),
            "chunks": len(chunks),
            "added": datetime.now().strftime("%H:%M")
        }
        self.histories[doc_id] = []

        print(f"✅ {doc_name}: {len(pages)} pages, {len(chunks)} chunks")
        return doc_id


    def list_documents(self):
        """Show all documents"""
        print("\nUploaded Documents:")
        print("-" * 40)
        if not self.documents:
            print("None yet")
            return
        for doc_id, info in self.documents.items():
            print(f"[{doc_id}] {info['name']}")
            print(f"  Pages: {info['pages']} | Chunks: {info['chunks']} | Added: {info['added']}")


    def remove_document(self, doc_id):
        """Remove a document and all its chunks"""
        if doc_id not in self.documents:
            print(f"Document {doc_id} not found")
            return
        name    = self.documents[doc_id]["name"]
        results = self.collection.get(where={"doc_id": doc_id})
        if results["ids"]:
            self.collection.delete(ids=results["ids"])
        del self.documents[doc_id]
        del self.histories[doc_id]
        print(f"✅ Removed: {name}")


    def retrieve(self, question, doc_id=None, n=3):
        """
        Retrieve chunks.
        doc_id given → search that document only
        doc_id None  → search all documents
        """
        q_emb = self.embedder.encode([question]).tolist()

        # 🔑 where filter restricts search to one document
        query_args = dict(
            query_embeddings = q_emb,
            n_results        = n,
            include          = ["documents", "metadatas", "distances"]
        )
        if doc_id:
            query_args["where"] = {"doc_id": doc_id}

        results = self.collection.query(**query_args)

        return [
            {
                "text":      results["documents"][0][i],
                "page":      results["metadatas"][0][i]["page"],
                "doc_name":  results["metadatas"][0][i]["doc_name"],
                "doc_id":    results["metadatas"][0][i]["doc_id"],
                "relevance": round(1 - results["distances"][0][i], 2)
            }
            for i in range(len(results["documents"][0]))
        ]


    def chat(self, question, doc_id=None):
        """
        Ask a question with conversation memory.
        doc_id=None → global search across all docs
        doc_id=xyz  → search within that doc only
        """
        history = self.histories.get(doc_id, []) if doc_id \
                  else self.global_history

        chunks  = self.retrieve(question, doc_id)
        context = "\n\n".join(
            f"[{c['doc_name']} — Page {c['page']}]: {c['text']}"
            for c in chunks
        )
        sources = [
            {"doc": c["doc_name"], "page": c["page"],
             "relevance": c["relevance"]}
            for c in chunks
        ]

        scope = f"document '{self.documents[doc_id]['name']}'" \
                if doc_id else "all uploaded documents"

        system_msg = {
            "role": "system",
            "content": f"""You are a helpful assistant searching {scope}.
Answer using ONLY the context. Cite as (DocumentName, Page X).
Say "I don't have that information" if context is insufficient.

Context:
{context}"""
        }

        messages = [system_msg] + history + [
            {"role": "user", "content": question}
        ]

        response = self.groq.chat.completions.create(
            model       = "llama-3.3-70b-versatile",
            messages    = messages,
            max_tokens  = 400,
            temperature = 0.1
        )
        answer = response.choices[0].message.content

        # Update history (trim to last 10)
        updated = history + [
            {"role": "user",      "content": question},
            {"role": "assistant", "content": answer}
        ]
        updated = updated[-10:]

        if doc_id:
            self.histories[doc_id] = updated
        else:
            self.global_history = updated

        return {"answer": answer, "sources": sources,
                "mode": "single" if doc_id else "global"}


    def compare(self, question, doc_id_1, doc_id_2):
        """Compare what two documents say about a topic"""
        name1   = self.documents[doc_id_1]["name"]
        name2   = self.documents[doc_id_2]["name"]
        chunks1 = self.retrieve(question, doc_id_1, n=2)
        chunks2 = self.retrieve(question, doc_id_2, n=2)

        ctx1 = "\n".join(f"[Page {c['page']}]: {c['text']}" for c in chunks1)
        ctx2 = "\n".join(f"[Page {c['page']}]: {c['text']}" for c in chunks2)

        prompt = f"""Compare what these two documents say about: "{question}"

{name1}:
{ctx1}

{name2}:
{ctx2}

Structure your answer as:
1. {name1} says...
2. {name2} says...
3. Key similarities...
4. Key differences..."""

        response = self.groq.chat.completions.create(
            model      = "llama-3.3-70b-versatile",
            messages   = [{"role": "user", "content": prompt}],
            max_tokens = 600,
            temperature= 0.1
        )
        return response.choices[0].message.content


    def reset(self, doc_id=None):
        """Reset chat history"""
        if doc_id:
            self.histories[doc_id] = []
            print(f"Reset history for {self.documents[doc_id]['name']}")
        else:
            self.global_history = []
            print("Reset global history")


# ============================================================
# PART 4: Test Everything
# ============================================================

print("\n" + "=" * 55)
print("PART 4: Testing")
print("=" * 55)

rag = MultiDocumentRAG()

# Add documents
doc1 = rag.add_document("ml_textbook.pdf",  "ML Textbook")
doc2 = rag.add_document("python_guide.pdf", "Python Guide")
doc3 = rag.add_document("career_guide.pdf", "Career Guide")

rag.list_documents()

# Test 1: Single document search
print("\nTEST 1: Within ML Textbook only")
r = rag.chat("What are the types of machine learning?", doc_id=doc1)
print(f"A: {r['answer']}")
print(f"Sources: {r['sources']}")

# Test 2: Different document
print("\nTEST 2: Within Python Guide only")
r = rag.chat("What libraries are used for data science?", doc_id=doc2)
print(f"A: {r['answer']}")
print(f"Sources: {r['sources']}")

# Test 3: Global search
print("\nTEST 3: Across ALL documents")
r = rag.chat("What skills do I need to learn AI?")
print(f"A: {r['answer']}")
print(f"Sources: {r['sources']}")

# Test 4: Follow-up conversation
print("\nTEST 4: Follow-up within ML Textbook")
r1 = rag.chat("What is deep learning?", doc_id=doc1)
print(f"Q1: What is deep learning?")
print(f"A1: {r1['answer'][:150]}...")
r2 = rag.chat("Give me examples of it", doc_id=doc1)
print(f"\nQ2: Give me examples of it")
print(f"A2: {r2['answer'][:150]}...")

# Test 5: Compare documents
print("\nTEST 5: Compare ML Textbook vs Python Guide")
comparison = rag.compare(
    "What Python tools or libraries are mentioned?",
    doc1, doc2
)
print(comparison)

# Test 6: Remove document
print("\nTEST 6: Remove Career Guide")
rag.remove_document(doc3)
rag.list_documents()


# ============================================================
# DAY 4 CHALLENGE
# ============================================================

print("\n" + "=" * 55)
print("DAY 4 CHALLENGE")
print("=" * 55)
print("""
1. Add your OWN PDF to the system.
   Ask the same question to your doc and ML Textbook.
   Do the answers differ? Which is more relevant?

2. Ask a question that spans multiple documents:
   "What Python tools are used in ML careers?"
   Does global search pull from the right docs?

3. Ask a specific doc about something NOT in it but
   that IS in another doc. What does the LLM say?
   Does it correctly say it doesn't have that info?

4. Test compare() with a meaningful question.
   "How do both documents explain machine learning?"

5. Most important — explain in your own words:
   How does ChromaDB know which chunks belong to which
   document? What would break if we didn't store doc_id
   in the metadata?
""")

# ─────────────────────────────────────────────
# Day 4 done when you can use the system with
# 3+ documents and answer challenge 5.
#
# Come back and say "RAG day 5" — Flask API
# that wraps everything so React can call it.
# ─────────────────────────────────────────────
