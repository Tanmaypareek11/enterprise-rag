import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# -----------------------------
# Load Models Once (Global)
# -----------------------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

model.config.tie_word_embeddings = False

# -----------------------------
# 1️⃣ Load PDF (from Streamlit upload)
# -----------------------------
def load_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""

    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted

    return text


# -----------------------------
# 2️⃣ Chunk text
# -----------------------------
def create_chunks(text, chunk_size=400, overlap=50):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks


# -----------------------------
# 3️⃣ Create FAISS Index
# -----------------------------
def build_index(chunks):
    embeddings = embedding_model.encode(chunks)
    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index, chunks


# -----------------------------
# 4️⃣ Generate Answer
# -----------------------------
def generate_answer(context, question):

    prompt = f"""
You are an AI assistant.

Use ONLY the context below to answer the question.
If the answer is not in the context, say "Not found in context."

Context:
{context}

Question:
{question}

Answer:
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=False
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# -----------------------------
# 5️⃣ MAIN FUNCTION (for Streamlit)
# -----------------------------
def generate_answer_logic(query, uploaded_file):

    # Extract text
    text = load_pdf(uploaded_file)

    if not text.strip():
        return "The uploaded PDF contains no readable text."

    # Create chunks
    chunks = create_chunks(text)

    if len(chunks) == 0:
        return "Could not create chunks from the PDF."

    # Build index
    index, chunks = build_index(chunks)

    # Encode query
    query_embedding = embedding_model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    # Retrieve top 5 chunks
    k = 5
    distances, indices = index.search(query_embedding, k)

    retrieved_chunks = [chunks[i] for i in indices[0]]
    context = "\n".join(retrieved_chunks)

    # Generate answer
    answer = generate_answer(context, query)

    return answer
