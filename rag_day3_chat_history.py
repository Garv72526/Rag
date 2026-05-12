# ============================================================
# RAG DAY 3 — Chat History + Conversation Memory
# Goal: Make follow-up questions work naturally
#       "What else did it say about that?" just works
# Run each section in Jupyter Notebook cell by cell
# ============================================================

import os
import re
import string
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq

load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "your-groq-key-here")


# ============================================================
# PART 1: Why Chat History Matters
# ============================================================

print("=" * 55)
print("PART 1: The Problem Without Chat History")
print("=" * 55)

# Without chat history — every question is independent
# User: "What is supervised learning?"
# AI:   "Supervised learning is..."
# User: "What are its applications?"  ← WHO IS "ITS"???
# AI:   ??? model has no idea what "its" refers to

# With chat history — context carries over
# User: "What is supervised learning?"
# AI:   "Supervised learning is..."
# User: "What are its applications?"
# AI:   knows "its" = supervised learning → answers correctly

# 🔑 This is called "conversational context" and it's what
# separates a chatbot from a proper AI assistant


# ============================================================
# PART 2: Setup — Load PDF + ChromaDB (same as Day 2)
# ============================================================

print("\n" + "=" * 55)
print("PART 2: Setup")
print("=" * 55)

PDF_PATH = "document.pdf"  # use same PDF from Day 2

# Load and chunk PDF
loader   = PyPDFLoader(PDF_PATH)
pages    = loader.load()
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=50
)
chunks = splitter.split_documents(pages)
print(f"✅ Loaded {len(pages)} pages, {len(chunks)} chunks")

# Embeddings
print("Loading embedding model...")
embedder   = SentenceTransformer("all-MiniLM-L6-v2")
texts      = [c.page_content for c in chunks]
embeddings = embedder.encode(texts)
print("✅ Embeddings ready!")

# ChromaDB
client = chromadb.PersistentClient(path="./rag_day3_db")
try:
    client.delete_collection("day3_docs")
except:
    pass
collection = client.create_collection("day3_docs")
collection.add(
    documents  = texts,
    embeddings = embeddings.tolist(),
    metadatas  = [
        {"page": c.metadata.get("page", 0) + 1}
        for c in chunks
    ],
    ids = [f"chunk_{i}" for i in range(len(chunks))]
)
print(f"✅ ChromaDB ready with {collection.count()} chunks!")

# Groq client
groq_client = Groq(api_key=GROQ_API_KEY)


# ============================================================
# PART 3: Understanding Chat History Format
# ============================================================

print("\n" + "=" * 55)
print("PART 3: Chat History Format")
print("=" * 55)

# Chat history is just a list of messages
# Each message has a role and content
# role = "user"      → what the user said
# role = "assistant" → what the AI said

# Example history:
example_history = [
    {
        "role":    "user",
        "content": "What is supervised learning?"
    },
    {
        "role":    "assistant",
        "content": "Supervised learning is a type of ML where the model learns from labeled data..."
    },
    {
        "role":    "user",
        "content": "What are its applications?"  # "its" = supervised learning
    }
]

# When we send this full history to the LLM, it understands
# that "its" refers to supervised learning from message 1
# This is exactly how ChatGPT works internally

print("Chat history format:")
for msg in example_history:
    print(f"  {msg['role'].upper()}: {msg['content'][:60]}...")


# ============================================================
# PART 4: RAG with Chat History — Step by Step
# ============================================================

print("\n" + "=" * 55)
print("PART 4: RAG + Chat History")
print("=" * 55)

def retrieve_chunks(question, n_results=3):
    """Find most relevant chunks for a question"""
    q_embedding = embedder.encode([question]).tolist()
    results = collection.query(
        query_embeddings = q_embedding,
        n_results        = n_results,
        include          = ["documents", "metadatas", "distances"]
    )
    chunks_found = []
    for i in range(len(results["documents"][0])):
        chunks_found.append({
            "text":      results["documents"][0][i],
            "page":      results["metadatas"][0][i]["page"],
            "relevance": round(1 - results["distances"][0][i], 2)
        })
    return chunks_found


def ask_with_history(question, chat_history, n_chunks=3):
    """
    RAG pipeline with conversation memory.

    How it works:
    1. Retrieve relevant chunks for current question
    2. Build context from chunks
    3. Build messages list = system prompt + history + new question
    4. Send to LLM — it sees full conversation
    5. Get answer + add to history
    6. Return answer + updated history
    """

    # Step 1: Retrieve relevant chunks
    chunks_found = retrieve_chunks(question, n_chunks)

    # Step 2: Build context
    context = ""
    sources = []
    for chunk in chunks_found:
        context += f"[Page {chunk['page']}]: {chunk['text']}\n\n"
        sources.append(f"Page {chunk['page']}")

    # Step 3: Build messages
    # 🔑 Structure:
    # - system message: tells LLM who it is + gives context
    # - chat_history:   all previous Q&A pairs
    # - new question:   current user question
    system_message = {
        "role": "system",
        "content": f"""You are a helpful assistant that answers questions about documents.
Use ONLY the provided context to answer. Always cite page numbers.
If the context doesn't contain enough information, say so clearly.
Keep answers concise and clear.

Relevant context from document:
{context}"""
    }

    # Build full messages list
    messages = [system_message] + chat_history + [
        {"role": "user", "content": question}
    ]

    # Step 4: Call LLM with full conversation
    response = groq_client.chat.completions.create(
        model       = "llama-3.3-70b-versatile",
        messages    = messages,
        max_tokens  = 400,
        temperature = 0.1
    )
    answer = response.choices[0].message.content

    # Step 5: Update chat history
    # Add current Q&A to history for next question
    updated_history = chat_history + [
        {"role": "user",      "content": question},
        {"role": "assistant", "content": answer}
    ]

    return {
        "answer":  answer,
        "sources": sources,
        "history": updated_history
    }


# ============================================================
# PART 5: Test Conversation — See History in Action
# ============================================================

print("\n" + "=" * 55)
print("PART 5: Test Conversation")
print("=" * 55)

# Start with empty history
history = []

# Question 1
print("\n" + "─" * 50)
print("Q1: What is machine learning?")
result1  = ask_with_history("What is machine learning?", history)
history  = result1["history"]  # update history
print(f"A: {result1['answer']}")
print(f"Sources: {result1['sources']}")

# Question 2 — follow up using "it"
print("\n" + "─" * 50)
print("Q2: What are the main types of it?")  # "it" = machine learning
result2  = ask_with_history("What are the main types of it?", history)
history  = result2["history"]
print(f"A: {result2['answer']}")
print(f"Sources: {result2['sources']}")

# Question 3 — another follow up
print("\n" * 1 + "─" * 50)
print("Q3: Which one is most commonly used?")
result3  = ask_with_history("Which one is most commonly used?", history)
history  = result3["history"]
print(f"A: {result3['answer']}")
print(f"Sources: {result3['sources']}")

# Question 4 — completely new topic
print("\n" + "─" * 50)
print("Q4: Tell me about neural networks")
result4  = ask_with_history("Tell me about neural networks", history)
history  = result4["history"]
print(f"A: {result4['answer']}")
print(f"Sources: {result4['sources']}")

# Question 5 — refer back to earlier topic
print("\n" + "─" * 50)
print("Q5: How does it compare to what you told me about ML types?")
result5  = ask_with_history(
    "How does it compare to what you told me about ML types?",
    history
)
history  = result5["history"]
print(f"A: {result5['answer']}")
print(f"Sources: {result5['sources']}")

print("\n✅ Full conversation with memory works!")
print(f"History now has {len(history)} messages")


# ============================================================
# PART 6: History Management
# Important — history grows forever without limits
# ============================================================

print("\n" + "=" * 55)
print("PART 6: History Management")
print("=" * 55)

# Problem: if user asks 100 questions, history gets huge
# LLMs have a context window limit — too much history = error
# Solution: keep only last N messages

def trim_history(history, max_messages=10):
    """
    Keep only the last max_messages in history.
    Always keep pairs (user + assistant) so we don't
    cut in the middle of a conversation turn.
    max_messages=10 means 5 Q&A pairs
    """
    if len(history) <= max_messages:
        return history

    # Keep last max_messages (always even number for pairs)
    trimmed = history[-max_messages:]

    # Make sure we start with a user message not assistant
    if trimmed[0]["role"] == "assistant":
        trimmed = trimmed[1:]

    return trimmed

# Test trimming
long_history = history * 5  # simulate long conversation
print(f"Before trim: {len(long_history)} messages")
trimmed = trim_history(long_history, max_messages=6)
print(f"After trim:  {len(trimmed)} messages")
print("✅ History trimming works!")


# ============================================================
# PART 7: Clean RAGChat Class
# This is what your Flask API will use
# ============================================================

print("\n" + "=" * 55)
print("PART 7: Clean RAGChat Class")
print("=" * 55)

class RAGChat:
    """
    Complete RAG chatbot with conversation memory.
    Drop this into your Flask API on Day 5.
    """

    def __init__(self, pdf_path, max_history=10):
        print(f"Initializing RAGChat for: {pdf_path}")
        self.max_history = max_history
        self.chat_history = []  # starts empty

        # Load PDF
        loader   = PyPDFLoader(pdf_path)
        pages    = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=50
        )
        chunks = splitter.split_documents(pages)

        # Embeddings
        self.embedder  = SentenceTransformer("all-MiniLM-L6-v2")
        texts          = [c.page_content for c in chunks]
        embeddings     = self.embedder.encode(texts)

        # ChromaDB
        self.client = chromadb.PersistentClient(path="./ragchat_db")
        try:
            self.client.delete_collection("ragchat")
        except:
            pass
        self.collection = self.client.create_collection("ragchat")
        self.collection.add(
            documents  = texts,
            embeddings = embeddings.tolist(),
            metadatas  = [
                {"page": c.metadata.get("page", 0) + 1}
                for c in chunks
            ],
            ids = [f"chunk_{i}" for i in range(len(chunks))]
        )

        # Groq
        self.groq = Groq(api_key=GROQ_API_KEY)
        print(f"✅ Ready! {len(chunks)} chunks indexed from {len(pages)} pages")

    def chat(self, question):
        """Ask a question — remembers previous conversation"""

        # Retrieve relevant chunks
        q_emb   = self.embedder.encode([question]).tolist()
        results = self.collection.query(
            query_embeddings = q_emb,
            n_results        = 3,
            include          = ["documents", "metadatas", "distances"]
        )

        # Build context + sources
        context = ""
        sources = []
        for i in range(len(results["documents"][0])):
            page      = results["metadatas"][0][i]["page"]
            text      = results["documents"][0][i]
            relevance = round(1 - results["distances"][0][i], 2)
            context  += f"[Page {page}]: {text}\n\n"
            sources.append({"page": page, "relevance": relevance})

        # Build messages with history
        system_msg = {
            "role": "system",
            "content": f"""You are a helpful document assistant.
Answer using ONLY the context below. Cite page numbers.
Say "I don't have that information" if context is insufficient.

Context:
{context}"""
        }

        messages = [system_msg] + self.chat_history + [
            {"role": "user", "content": question}
        ]

        # Call LLM
        response = self.groq.chat.completions.create(
            model       = "llama-3.3-70b-versatile",
            messages    = messages,
            max_tokens  = 400,
            temperature = 0.1
        )
        answer = response.choices[0].message.content

        # Update + trim history
        self.chat_history.append(
            {"role": "user",      "content": question}
        )
        self.chat_history.append(
            {"role": "assistant", "content": answer}
        )
        if len(self.chat_history) > self.max_history:
            self.chat_history = self.chat_history[-self.max_history:]

        return {
            "answer":        answer,
            "sources":       sources,
            "history_length": len(self.chat_history)
        }

    def reset(self):
        """Clear conversation history — start fresh"""
        self.chat_history = []
        print("✅ Conversation reset!")

    def get_history(self):
        """Return full conversation history"""
        return self.chat_history


# Test the clean class
print("\nTesting RAGChat class:")
chat = RAGChat(PDF_PATH)

# Simulate a real conversation
questions = [
    "What is machine learning?",
    "What are the three main types?",
    "Tell me more about the first one",
    "How does it differ from the second type?",
]

for q in questions:
    print(f"\nQ: {q}")
    result = chat.chat(q)
    print(f"A: {result['answer'][:200]}...")
    print(f"Sources: {[s['page'] for s in result['sources']]}")
    print(f"History length: {result['history_length']} messages")

# Test reset
chat.reset()
print(f"\nAfter reset — history length: {len(chat.get_history())}")


# ============================================================
# DAY 3 CHALLENGE
# ============================================================

print("\n" + "=" * 55)
print("DAY 3 CHALLENGE")
print("=" * 55)
print("""
1. Have a 10 message conversation with your document.
   Use pronouns like "it", "that", "they", "the first one"
   Does the model always understand the reference correctly?
   When does it get confused?

2. Try this: Ask a question, get an answer.
   Then ask "Are you sure about that?"
   Does it correctly refer back to its previous answer?

3. Change max_history from 10 to 2.
   Have a long conversation.
   At what point does it "forget" earlier context?
   What does this tell you about context window limits?

4. Call chat.reset() mid conversation.
   Ask a follow up question that references something
   said before the reset.
   What happens? Why?

5. Most important — explain in your own words:
   Why do we pass the FULL history to the LLM every time
   instead of just storing it somewhere and only sending
   the latest question?
   This is a key architectural concept in LLM apps.
""")

# ─────────────────────────────────────────────
# Day 3 done when you can answer challenge 5
# clearly in your own words.
#
# Come back and say "RAG day 4" — we add
# multi-document support so users can upload
# multiple PDFs and switch between them.
# ─────────────────────────────────────────────
