# ============================================================
# RAG DAY 1 — Embeddings + ChromaDB + Basic Pipeline
# Goal: Understand embeddings, store text in ChromaDB,
#       and do your first semantic search
# Run each section in Jupyter Notebook cell by cell
# ============================================================

# First install these if not done already:
# pip install langchain langchain-community chromadb
#             sentence-transformers pypdf groq langchain-groq

from sentence_transformers import SentenceTransformer
import chromadb
import numpy as np


# ============================================================
# PART 1: Embeddings — Understanding them deeply
# ============================================================

print("=" * 55)
print("PART 1: Embeddings")
print("=" * 55)

# Load a free embedding model
# all-MiniLM-L6-v2 is the most popular beginner model
# - Runs on your laptop (no API key needed)
# - Converts any text to 384 numbers
# - Downloads automatically first time (~80MB)
print("\nLoading embedding model (first time takes ~1 min)...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
print("✅ Model loaded!")


# ─────────────────────────────────────────────
# SECTION 1.1: Convert text to vectors
# ─────────────────────────────────────────────

print("\n--- Converting text to vectors ---")

sentence = "The patient has a severe headache"
embedding = embedder.encode(sentence)

print(f"Sentence: '{sentence}'")
print(f"Vector shape: {embedding.shape}")      # (384,)
print(f"First 10 numbers: {embedding[:10]}")
print(f"These 384 numbers represent the MEANING of the sentence")


# ─────────────────────────────────────────────
# SECTION 1.2: Similarity — the core concept
# ─────────────────────────────────────────────

print("\n--- Similarity between sentences ---")

def cosine_similarity(a, b):
    """
    Measures how similar two vectors are.
    1.0 = identical meaning
    0.0 = completely unrelated
    Formula: dot product / (magnitude of a * magnitude of b)
    """
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

sentences = [
    "The patient has a severe headache",       # base sentence
    "The person is experiencing head pain",    # similar meaning
    "The doctor prescribed medication",        # related topic
    "I love eating pizza on weekends",         # completely unrelated
    "Machine learning is fascinating",         # unrelated
]

base = embedder.encode(sentences[0])
others = embedder.encode(sentences[1:])

print(f"\nBase: '{sentences[0]}'")
print("-" * 50)
for i, (sent, emb) in enumerate(zip(sentences[1:], others)):
    sim = cosine_similarity(base, emb)
    bar = "█" * int(sim * 20)
    print(f"{sim:.2f} {bar}")
    print(f"     '{sent}'")

# 🔑 KEY INSIGHT:
# "headache" vs "head pain" → ~0.85 (very similar)
# "headache" vs "pizza"     → ~0.10 (unrelated)
# This is WHY embeddings power RAG — semantic understanding


# ─────────────────────────────────────────────
# SECTION 1.3: Sentence embeddings (whole sentences)
# ─────────────────────────────────────────────

print("\n--- Whole sentence similarity ---")

questions = [
    "What are the side effects of the medicine?",
    "What adverse reactions does the drug have?",    # same meaning, different words
    "How do I cook pasta?",                          # unrelated
]

q_embeddings = embedder.encode(questions)
base_q = q_embeddings[0]

print(f"Base question: '{questions[0]}'")
print()
for i in range(1, len(questions)):
    sim = cosine_similarity(base_q, q_embeddings[i])
    print(f"Similarity: {sim:.2f} → '{questions[i]}'")

# 🔑 "side effects" vs "adverse reactions" → ~0.80
# This is why RAG finds the right chunk even when
# the user's question uses different words than the document


# ============================================================
# PART 2: ChromaDB — Your Vector Database
# ============================================================

print("\n" + "=" * 55)
print("PART 2: ChromaDB")
print("=" * 55)

# Initialize ChromaDB — runs entirely on your laptop
# No signup, no API key, free forever
client = chromadb.Client()

# Create a collection (like a table in MongoDB)
# Each collection stores embeddings + their text + metadata
collection = client.create_collection(
    name     = "my_first_collection",
    metadata = {"description": "learning ChromaDB"}
)
print("✅ ChromaDB collection created!")


# ─────────────────────────────────────────────
# SECTION 2.1: Adding documents
# ─────────────────────────────────────────────

print("\n--- Adding documents to ChromaDB ---")

# Sample text chunks (pretend these came from a PDF)
documents = [
    "Python is a high-level programming language known for its simplicity.",
    "Machine learning is a subset of AI that learns from data.",
    "Neural networks are inspired by the human brain structure.",
    "Flask is a lightweight Python web framework for building APIs.",
    "ChromaDB is a vector database that stores text embeddings.",
    "React is a JavaScript library for building user interfaces.",
    "MongoDB is a NoSQL document database used in web development.",
    "Sentiment analysis is the process of identifying emotion in text.",
    "TF-IDF converts text to numbers based on word frequency.",
    "Embeddings capture semantic meaning of text as dense vectors.",
]

# Metadata — store extra info like page numbers, source file
metadatas = [
    {"topic": "python",    "page": 1},
    {"topic": "ml",        "page": 2},
    {"topic": "ml",        "page": 3},
    {"topic": "flask",     "page": 4},
    {"topic": "chromadb",  "page": 5},
    {"topic": "react",     "page": 6},
    {"topic": "mongodb",   "page": 7},
    {"topic": "nlp",       "page": 8},
    {"topic": "nlp",       "page": 9},
    {"topic": "embeddings","page": 10},
]

# Generate embeddings for all documents
print("Generating embeddings...")
embeddings = embedder.encode(documents).tolist()

# Add to ChromaDB
collection.add(
    documents  = documents,
    embeddings = embeddings,
    metadatas  = metadatas,
    ids        = [str(i) for i in range(len(documents))]
)

print(f"✅ Added {len(documents)} documents to ChromaDB!")
print(f"Collection count: {collection.count()}")


# ─────────────────────────────────────────────
# SECTION 2.2: Querying — semantic search
# ─────────────────────────────────────────────

print("\n--- Semantic Search ---")

def search(query, n_results=3):
    """Search ChromaDB for most relevant chunks"""
    # Convert query to embedding
    query_embedding = embedder.encode([query]).tolist()

    # Find most similar chunks
    results = collection.query(
        query_embeddings = query_embedding,
        n_results        = n_results,
        include          = ["documents", "metadatas", "distances"]
    )

    print(f"\nQuery: '{query}'")
    print(f"Top {n_results} results:")
    print("-" * 50)

    for i in range(len(results["documents"][0])):
        doc      = results["documents"][0][i]
        meta     = results["metadatas"][0][i]
        distance = results["distances"][0][i]
        # Lower distance = more similar (opposite of similarity score)
        similarity = 1 - distance

        print(f"Result {i+1} (similarity: {similarity:.2f})")
        print(f"  Page {meta['page']}: {doc}")
    print()

# Test different queries
search("How do neural networks work?")
search("What is used for building web APIs?")
search("How does text get converted to numbers?")

# 🔑 Notice: "web APIs" finds Flask even though query didn't say Flask
# That's semantic search — understanding MEANING not just keywords


# ─────────────────────────────────────────────
# SECTION 2.3: Persistent storage
# ─────────────────────────────────────────────

print("--- Persistent ChromaDB (saves to disk) ---")

# The above used in-memory ChromaDB (resets when Python restarts)
# For your RAG app you need persistent storage (saves to disk)

import os

# Create a persistent client that saves to a folder
persistent_client = chromadb.PersistentClient(path="./chroma_db")

# Create or get existing collection
try:
    persistent_collection = persistent_client.create_collection("rag_documents")
    print("✅ Created new persistent collection")
except:
    persistent_collection = persistent_client.get_collection("rag_documents")
    print("✅ Loaded existing persistent collection")

# Add same documents
persistent_collection.add(
    documents  = documents,
    embeddings = embeddings,
    metadatas  = metadatas,
    ids        = [str(i) for i in range(len(documents))]
)

print(f"Saved {persistent_collection.count()} documents to ./chroma_db/")
print("These persist even after Python restarts!")


# ============================================================
# PART 3: Complete Mini RAG Pipeline (no LLM yet)
# Just retrieval — we add the LLM answer on Day 2
# ============================================================

print("\n" + "=" * 55)
print("PART 3: Mini RAG Pipeline")
print("=" * 55)

def mini_rag(question, n_chunks=3):
    """
    Simple RAG pipeline:
    1. Convert question to embedding
    2. Find most relevant chunks
    3. Return chunks as context (LLM added Day 2)
    """
    print(f"\nQuestion: '{question}'")
    print("=" * 50)

    # Step 1: Embed the question
    q_embedding = embedder.encode([question]).tolist()

    # Step 2: Retrieve relevant chunks
    results = collection.query(
        query_embeddings = q_embedding,
        n_results        = n_chunks,
        include          = ["documents", "metadatas", "distances"]
    )

    # Step 3: Build context
    print(f"Found {n_chunks} relevant chunks:")
    context_parts = []
    for i in range(len(results["documents"][0])):
        doc      = results["documents"][0][i]
        meta     = results["metadatas"][0][i]
        distance = results["distances"][0][i]
        similarity = round(1 - distance, 2)

        print(f"\n  Chunk {i+1} — Page {meta['page']} (relevance: {similarity})")
        print(f"  {doc}")
        context_parts.append(f"[Page {meta['page']}]: {doc}")

    context = "\n".join(context_parts)

    print(f"\nContext that would be sent to LLM:")
    print("-" * 50)
    print(context)
    print("-" * 50)
    print("(On Day 2 we send this to Groq LLM for the final answer)")

    return context

# Test the mini RAG pipeline
mini_rag("What tools are used for machine learning?")
mini_rag("How does text get understood by computers?")


# ============================================================
# DAY 1 CHALLENGE
# ============================================================

print("\n" + "=" * 55)
print("DAY 1 CHALLENGE")
print("=" * 55)
print("""
1. Add 5 more documents to the collection about any topic
   you want (movies, sports, cooking — anything).
   Then search for them with a question.
   Do the right chunks come back?

2. Try searching for something NOT in the collection.
   What happens? What does ChromaDB return?
   What does this tell you about RAG limitations?

3. Change n_results from 3 to 1 and then to 5.
   When would you want more chunks vs fewer chunks?
   Think about: accuracy vs context window limits.

4. Most important — in your own words, explain:
   What is the difference between what ChromaDB does
   and what MongoDB does?
   Write it as a comment in this file.
""")

# ─────────────────────────────────────────────
# Day 1 done when you can answer challenge 4
# clearly in your own words.
#
# Come back and say "RAG day 2" — we load a real PDF,
# split it into chunks, and add the Groq LLM to get
# actual answers with page citations.
# ─────────────────────────────────────────────
