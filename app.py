# app.py
import os
import PyPDF2
import faiss
from flask import Flask, request, render_template
from sentence_transformers import SentenceTransformer
from transformers import pipeline

app = Flask(__name__)

# Modelo de embeddings y LLM
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
qa_pipeline = pipeline("text-generation", model="gpt2")

# Variables globales
chunks = []
index = None

def extract_text_from_pdf(pdf_path):
    reader = PyPDF2.PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def create_index(chunks, model):
    embeddings = model.encode(chunks)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def answer_question(query, context, qa_pipeline):
    prompt = f"Contexto:\n{context}\n\nPregunta: {query}\nRespuesta:"
    return qa_pipeline(prompt, max_length=200)[0]['generated_text']

def search_chunks(query, chunks, model, index, k=3):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, k)

    # Si no hay resultados relevantes (ej. distancia muy alta o índice vacío)
    if len(indices[0]) == 0 or all(d > 1.5 for d in distances[0]):  
        return None
    return [chunks[i] for i in indices[0]]

@app.route("/", methods=["GET", "POST"])
def home():
    global chunks, index
    answer = None
    if request.method == "POST":
        if "pdf" in request.files:
            pdf = request.files["pdf"]
            
            
            pdf_path = os.path.join("uploads", pdf.filename)
            os.makedirs("uploads", exist_ok=True)
            pdf.save(pdf_path)

            text = extract_text_from_pdf(pdf_path)
            chunks = chunk_text(text)
            index = create_index(chunks, embed_model)
            answer = "PDF cargado exitosamente. Ahora puedes hacer preguntas."
        elif "question" in request.form and index is not None:
            query = request.form["question"]
            results = search_chunks(query, chunks, embed_model, index)
            if results is None:
                answer = "No se ha encontrado resultados."
            else:
                context = "\n".join(results)
                answer = answer_question(query, context, qa_pipeline)
    return render_template("index.html", answer=answer)

if __name__ == "__main__":
    app.run(debug=True)