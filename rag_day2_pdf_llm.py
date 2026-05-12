# ============================================================
# RAG DAY 2 — Real PDF + LLM Answers + Page Citations
# Goal: Load a real PDF, split into chunks, store in ChromaDB,
#       ask questions and get answers with page numbers
# Run each section in Jupyter Notebook cell by cell
# ============================================================

# Make sure these are installed:
# pip install langchain langchain-community chromadb
#             sentence-transformers pypdf groq langchain-groq

import os
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq


# ============================================================
# STEP 1: Get a PDF to work with
# ============================================================

print("=" * 55)
print("STEP 1: PDF Setup")
print("=" * 55)

# Option A — Use any PDF you already have on your laptop
# Just change the path below to point to your PDF
PDF_PATH = "document.pdf"  # ← change this to your PDF path

# Option B — Download a free sample PDF for testing
# Run this in terminal to download a sample:
# curl -L "https://www.africau.edu/images/default/sample.pdf" -o document.pdf

# Option C — Use Python to create a test PDF
# Run this if you don't have a PDF handy:
try:
    # Try to install fpdf if not present
    import subprocess
    subprocess.run(["pip", "install", "fpdf"], capture_output=True)
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Page 1
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, """Introduction to Machine Learning

Machine learning is a subset of artificial intelligence that enables systems to learn 
and improve from experience without being explicitly programmed. It focuses on 
developing computer programs that can access data and use it to learn for themselves.

The process begins with observations or data, such as examples, direct experience, 
or instruction. Machine learning algorithms use computational methods to learn 
information directly from data without relying on a predetermined equation as a model.

There are three main types of machine learning: supervised learning, unsupervised 
learning, and reinforcement learning. Each type has different use cases and 
applications in the real world.""")

    # Page 2
    pdf.add_page()
    pdf.multi_cell(0, 10, """Supervised Learning

Supervised learning is the most common type of machine learning. In supervised 
learning, the algorithm is trained on a labeled dataset, which means each training 
example is paired with an output label.

Common supervised learning algorithms include linear regression for predicting 
continuous values, logistic regression for binary classification, decision trees 
for both classification and regression, random forests which combine multiple 
decision trees, and support vector machines for classification tasks.

Applications of supervised learning include email spam detection, image recognition, 
medical diagnosis, and sentiment analysis of text data.""")

    # Page 3
    pdf.add_page()
    pdf.multi_cell(0, 10, """Neural Networks and Deep Learning

Neural networks are computing systems inspired by biological neural networks in 
the human brain. They consist of layers of interconnected nodes or neurons that 
process information using connectionist approaches to computation.

Deep learning uses neural networks with many layers, called deep neural networks. 
These networks can learn representations of data with multiple levels of abstraction. 
Deep learning has revolutionized fields like computer vision, natural language 
processing, and speech recognition.

Key components include input layers that receive raw data, hidden layers that 
extract features, and output layers that produce final predictions. Training 
involves backpropagation and gradient descent to minimize prediction errors.""")

    # Page 4
    pdf.add_page()
    pdf.multi_cell(0, 10, """Natural Language Processing

Natural language processing (NLP) is a branch of artificial intelligence that 
helps computers understand, interpret, and manipulate human language. NLP draws 
from many disciplines including computer science and computational linguistics.

Key NLP tasks include text classification, sentiment analysis, named entity 
recognition, machine translation, and question answering. Modern NLP relies 
heavily on transformer models like BERT and GPT which use attention mechanisms 
to understand context in text.

RAG or Retrieval Augmented Generation combines information retrieval with language 
generation. It retrieves relevant documents from a knowledge base and uses them 
as context for generating accurate, grounded responses to user queries.""")

    pdf.output("document.pdf")
    print("✅ Test PDF created: document.pdf (4 pages)")
    PDF_PATH = "document.pdf"

except Exception as e:
    print(f"Could not create PDF: {e}")
    print("Please provide your own PDF and set PDF_PATH above")


# ============================================================
# STEP 2: Load the PDF
# ============================================================

print("\n" + "=" * 55)
print("STEP 2: Loading PDF")
print("=" * 55)

# PyPDFLoader reads your PDF page by page
# Each page becomes a Document object with:
# - page_content: the text on that page
# - metadata: { source: filename, page: page_number }

loader = PyPDFLoader(PDF_PATH)
pages  = loader.load()

print(f"✅ PDF loaded!")
print(f"Total pages: {len(pages)}")
print(f"\nPage 1 preview:")
print(pages[0].page_content[:300] + "...")
print(f"\nPage 1 metadata: {pages[0].metadata}")

# 🔑 metadata["page"] gives us the page number
# This is what we use for citations later!


# ============================================================
# STEP 3: Split into Chunks
# ============================================================

print("\n" + "=" * 55)
print("STEP 3: Splitting into Chunks")
print("=" * 55)

# Why split? LLMs have a context limit — they can't read
# a 100-page PDF all at once. We split into small chunks
# so we only send the RELEVANT parts to the LLM.

# RecursiveCharacterTextSplitter tries to split on:
# 1. Paragraphs (\n\n) first
# 2. Then sentences (. )
# 3. Then words ( )
# 4. Then characters
# This keeps sentences intact as much as possible

splitter = RecursiveCharacterTextSplitter(
    chunk_size    = 500,   # max 500 characters per chunk
    chunk_overlap = 50,    # overlap so context isn't lost at boundaries
    length_function = len,
)

chunks = splitter.split_documents(pages)

print(f"Total chunks: {len(chunks)}")
print(f"Average chunk size: {sum(len(c.page_content) for c in chunks) / len(chunks)} chars")

print(f"\nChunk 1:")
print(f"  Text: {chunks[0].page_content[:200]}...")
print(f"  Metadata: {chunks[0].metadata}")

print(f"\nChunk 2:")
print(f"  Text: {chunks[1].page_content[:200]}...")
print(f"  Metadata: {chunks[1].metadata}")

# 🔑 Each chunk keeps its metadata including page number
# So when we retrieve a chunk, we know which page it came from


# ============================================================
# STEP 4: Create Embeddings + Store in ChromaDB
# ============================================================

print("\n" + "=" * 55)
print("STEP 4: Embeddings + ChromaDB Storage")
print("=" * 55)

# Load embedding model (same as Day 1)
print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
print("✅ Model ready!")

# Generate embeddings for all chunks
print(f"Generating embeddings for {len(chunks)} chunks...")
texts      = [chunk.page_content for chunk in chunks]
embeddings = embedder.encode(texts, show_progress_bar=True)
print(f"✅ Embeddings shape: {embeddings.shape}")

# Store in ChromaDB (persistent — survives restarts)
print("\nStoring in ChromaDB...")
client = chromadb.PersistentClient(path="./rag_chroma_db")

# Delete existing collection if it exists (fresh start)
try:
    client.delete_collection("pdf_documents")
except:
    pass

collection = client.create_collection("pdf_documents")

# Add all chunks with their embeddings and metadata
collection.add(
    documents  = texts,
    embeddings = embeddings.tolist(),
    metadatas  = [
        {
            "page":   chunk.metadata.get("page", 0) + 1,  # +1 for human page numbers
            "source": chunk.metadata.get("source", PDF_PATH),
            "chunk":  i
        }
        for i, chunk in enumerate(chunks)
    ],
    ids = [f"chunk_{i}" for i in range(len(chunks))]
)

print(f"✅ Stored {collection.count()} chunks in ChromaDB!")


# ============================================================
# STEP 5: Retrieval — Find Relevant Chunks
# ============================================================

print("\n" + "=" * 55)
print("STEP 5: Retrieval")
print("=" * 55)

def retrieve_chunks(question, n_results=3):
    """Find most relevant chunks for a question"""

    # Embed the question
    q_embedding = embedder.encode([question]).tolist()

    # Search ChromaDB
    results = collection.query(
        query_embeddings = q_embedding,
        n_results        = n_results,
        include          = ["documents", "metadatas", "distances"]
    )

    # Format results
    chunks_found = []
    for i in range(len(results["documents"][0])):
        chunks_found.append({
            "text":       results["documents"][0][i],
            "page":       results["metadatas"][0][i]["page"],
            "source":     results["metadatas"][0][i]["source"],
            "relevance":  round(1 - results["distances"][0][i], 2)
        })

    return chunks_found

# Test retrieval
question = "What is supervised learning?"
chunks_found = retrieve_chunks(question)

print(f"Question: '{question}'")
print(f"Found {len(chunks_found)} relevant chunks:")
for i, chunk in enumerate(chunks_found):
    print(f"\n  Chunk {i+1} — Page {chunk['page']} (relevance: {chunk['relevance']})")
    print(f"  {chunk['text'][:150]}...")


# ============================================================
# STEP 6: Generation — Get LLM Answer with Citations
# ============================================================

print("\n" + "=" * 55)
print("STEP 6: LLM Answer with Citations")
print("=" * 55)

# Make sure GROQ_API_KEY is in your .env or set it here:
# os.environ["GROQ_API_KEY"] = "your-key-here"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "your-groq-key-here")

groq_client = Groq(api_key=GROQ_API_KEY)

def ask_question(question, n_chunks=3):
    """
    Full RAG pipeline:
    1. Retrieve relevant chunks
    2. Build context with page numbers
    3. Send to LLM
    4. Return answer with citations
    """
    print(f"\nQuestion: '{question}'")
    print("=" * 55)

    # Step 1: Retrieve
    chunks_found = retrieve_chunks(question, n_chunks)

    # Step 2: Build context with page citations
    context = ""
    for chunk in chunks_found:
        context += f"[Page {chunk['page']}]: {chunk['text']}\n\n"

    # Step 3: Build prompt
    # 🔑 We explicitly tell the LLM to:
    # a) Only use the provided context
    # b) Include page numbers in the answer
    # c) Say "I don't know" if context doesn't help
    prompt = f"""You are a helpful assistant that answers questions based ONLY on the provided context.
Always cite the page number when you use information from the context.
If the context doesn't contain enough information, say "I don't have enough information in the document to answer this."

Context from document:
{context}

Question: {question}

Answer (include page numbers as citations like "According to Page X..."):"""

    # Step 4: Call Groq LLM
    try:
        response = groq_client.chat.completions.create(
            model    = "llama-3.3-70b-versatile",
            messages = [{"role": "user", "content": prompt}],
            max_tokens = 500,
            temperature = 0.1,  # low temperature = more factual, less creative
        )
        answer = response.choices[0].message.content

    except Exception as e:
        answer = f"LLM error: {e}\nCheck your GROQ_API_KEY"

    # Step 5: Show answer + sources used
    print(f"\nAnswer:\n{answer}")
    print(f"\nSources used:")
    for chunk in chunks_found:
        print(f"  → Page {chunk['page']} (relevance: {chunk['relevance']})")

    return {
        "question": question,
        "answer":   answer,
        "sources":  chunks_found
    }

# Test with different questions
result1 = ask_question("What is supervised learning and what are its applications?")
result2 = ask_question("How do neural networks learn?")
result3 = ask_question("What is RAG in NLP?")

# Test with a question NOT in the document
result4 = ask_question("What is the weather like today?")
# Should say it doesn't have enough information


# ============================================================
# STEP 7: Full Pipeline Function
# Clean version you'll use in your Flask API
# ============================================================

print("\n" + "=" * 55)
print("STEP 7: Clean Pipeline Function")
print("=" * 55)

class RAGPipeline:
    """
    Clean RAG pipeline class.
    This is what your Flask API will use.
    """

    def __init__(self, pdf_path, collection_name="documents"):
        print(f"Initializing RAG pipeline for: {pdf_path}")

        # Load and chunk PDF
        loader   = PyPDFLoader(pdf_path)
        pages    = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=50
        )
        self.chunks = splitter.split_documents(pages)

        # Create embeddings
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        texts         = [c.page_content for c in self.chunks]
        embeddings    = self.embedder.encode(texts)

        # Store in ChromaDB
        self.client = chromadb.PersistentClient(path="./rag_db")
        try:
            self.client.delete_collection(collection_name)
        except:
            pass
        self.collection = self.client.create_collection(collection_name)
        self.collection.add(
            documents  = texts,
            embeddings = embeddings.tolist(),
            metadatas  = [
                {"page": c.metadata.get("page", 0) + 1}
                for c in self.chunks
            ],
            ids = [f"chunk_{i}" for i in range(len(self.chunks))]
        )

        # Groq client
        self.groq = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        print(f"✅ Ready! Indexed {len(self.chunks)} chunks from {len(pages)} pages")

    def ask(self, question, n_chunks=3):
        """Ask a question and get answer with citations"""

        # Retrieve
        q_emb   = self.embedder.encode([question]).tolist()
        results = self.collection.query(
            query_embeddings = q_emb,
            n_results        = n_chunks,
            include          = ["documents", "metadatas", "distances"]
        )

        # Build context
        context = ""
        sources = []
        for i in range(len(results["documents"][0])):
            page = results["metadatas"][0][i]["page"]
            text = results["documents"][0][i]
            relevance = round(1 - results["distances"][0][i], 2)
            context += f"[Page {page}]: {text}\n\n"
            sources.append({"page": page, "relevance": relevance})

        # Generate answer
        prompt = f"""Answer based ONLY on this context. Cite page numbers.
If context is insufficient, say so.

Context:
{context}

Question: {question}
Answer:"""

        response = self.groq.chat.completions.create(
            model      = "llama-3.3-70b-versatile",
            messages   = [{"role": "user", "content": prompt}],
            max_tokens = 400,
            temperature= 0.1
        )

        return {
            "answer":  response.choices[0].message.content,
            "sources": sources,
            "question": question
        }

# Test the clean pipeline
    
rag = RAGPipeline(PDF_PATH)
result = rag.ask("What types of machine learning exist?")
print(f"\nQ: {result['question']}")
print(f"A: {result['answer']}")
print(f"Sources: {result['sources']}")


# ============================================================
# DAY 2 CHALLENGE
# ============================================================

print("\n" + "=" * 55)
print("DAY 2 CHALLENGE")
print("=" * 55)
print("""
1. Use your OWN PDF — any PDF you have (notes, textbook,
   article). Change PDF_PATH and run the whole pipeline.
   Does it find the right answers?

2. Change chunk_size from 500 to 200, retrain.
   Then ask the same question. Is the answer better or worse?
   What does this tell you about chunk size choice?

3. Change n_chunks from 3 to 1 in ask_question().
   Does the answer quality drop? Why?

4. Ask a question that is NOT in your document.
   Does the LLM correctly say it doesn't know?
   Or does it hallucinate an answer?
   This is called "hallucination testing" — important in ML jobs.

5. Most important — explain in your own words:
   Why do we need BOTH the embedding model AND the LLM?
   What does each one do that the other can't?
""")

# ─────────────────────────────────────────────
# Day 2 done when you have a working pipeline
# that answers questions from a real PDF with citations.
#
# Come back and say "RAG day 3" — we add chat history
# so follow-up questions work naturally.
# ─────────────────────────────────────────────
