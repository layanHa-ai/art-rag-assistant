import os
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(
    page_title="Art RAG Assistant",
    page_icon="🎨",
    layout="wide"
)

st.title("🎨 Art RAG Assistant")
st.caption("A local RAG app for asking questions about art documents.")

DOCUMENTS_FOLDER = "documents"


def read_txt_or_md(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def read_pdf(file_path):
    text = ""
    reader = PdfReader(file_path)
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


def load_documents(folder_path):
    documents = []

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if filename.lower().endswith(".pdf"):
            text = read_pdf(file_path)
        elif filename.lower().endswith(".txt") or filename.lower().endswith(".md"):
            text = read_txt_or_md(file_path)
        else:
            continue

        documents.append({"filename": filename, "text": text})

    return documents


def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks


documents = load_documents(DOCUMENTS_FOLDER)

all_chunks = []

for doc in documents:
    chunks = chunk_text(doc["text"])

    for i, chunk in enumerate(chunks):
        all_chunks.append({
            "filename": doc["filename"],
            "chunk_id": i,
            "text": chunk
        })


with st.sidebar:
    st.header("📁 Document Summary")
    st.metric("Documents", len(documents))
    st.metric("Chunks", len(all_chunks))

    st.subheader("Loaded Files")
    for doc in documents:
        st.write(f"📄 {doc['filename']}")

    st.divider()
    st.write("Pipeline:")
    st.write("Documents → Chunks → Embeddings → FAISS → LLM Answer")


col1, col2, col3 = st.columns(3)

with col1:
    st.info(f"📄 Documents: {len(documents)}")

with col2:
    st.info(f"🧩 Chunks: {len(all_chunks)}")

with col3:
    st.info("🔎 Vector Store: FAISS")


if len(all_chunks) > 0:
    model = SentenceTransformer("all-MiniLM-L6-v2")

    chunk_texts = [chunk["text"] for chunk in all_chunks]

    embeddings = model.encode(
        chunk_texts,
        normalize_embeddings=True
    )

    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    st.success(f"✅ FAISS index is ready with {index.ntotal} vectors.")

    st.subheader("💬 Ask Your Documents")

    question = st.text_input(
        "Enter your question:",
        placeholder="Example: Who painted the Mona Lisa?"
    )

    if question:
        with st.spinner("Searching documents and generating answer..."):
            question_embedding = model.encode(
                [question],
                normalize_embeddings=True
            )

            question_embedding = np.array(question_embedding).astype("float32")

            scores, indices = index.search(question_embedding, k=3)

            retrieved_chunks = []

            for idx in indices[0]:
                retrieved_chunks.append(all_chunks[idx]["text"])

            context = "\n\n".join(retrieved_chunks)

            prompt = f"""
You are an art assistant.

Answer the user's question using ONLY the provided context.
If the answer is not in the context, say: I could not find the answer in the provided documents.

Context:
{context}

Question:
{question}
"""

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            answer = response.choices[0].message.content

        st.subheader("🎯 Answer")
        st.write(answer)

        st.subheader("🔍 Sources Used")

        for score, idx in zip(scores[0], indices[0]):
            chunk = all_chunks[idx]

            with st.expander(
                f"📄 {chunk['filename']} | Chunk {chunk['chunk_id']} | Score: {score:.3f}"
            ):
                st.write(chunk["text"][:700])

else:
    st.warning("No documents found. Please add .txt, .md, or .pdf files inside the documents folder.")