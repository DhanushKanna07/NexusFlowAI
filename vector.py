from fastapi import FastAPI, UploadFile, File
import fitz  # PyMuPDF
import pdfplumber
import base64
import os
import tempfile

# LangChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


app = FastAPI()

@app.post("/extract-pdf-data/")
async def extract_pdf_data(file: UploadFile = File(...)):
    try:
        # ---- STEP 1: Save temp PDF file ----
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            temp_pdf_path = tmp.name
            content = await file.read()
            tmp.write(content)

        extracted_tables = []
        extracted_images = []
        extracted_text = ""

        # ---- STEP 2: Extract text and tables ----
        with pdfplumber.open(temp_pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    extracted_text += page_text + "\n"

                tables = page.extract_tables()
                for table in tables:
                    table_dict = []
                    headers = table[0]
                    for row in table[1:]:
                        row_dict = {headers[i]: row[i] for i in range(len(headers))}
                        table_dict.append(row_dict)
                    extracted_tables.append(table_dict)

        # ---- STEP 3: Extract images (base64) ----
        doc = fitz.open(temp_pdf_path)
        for page_index in range(len(doc)):
            for img_index, img in enumerate(doc.get_page_images(page_index)):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                base64_image = base64.b64encode(image_bytes).decode('utf-8')
                extracted_images.append(base64_image)
        doc.close()
        os.remove(temp_pdf_path)

        # ---- STEP 4: Chunk + Embed text using LangChain ----
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_text(extracted_text)

        # Load embedding model
        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # Create vector store
        vectorstore = FAISS.from_texts(texts=chunks, embedding=embedding_model)

        # Save vector DB locally
        vectorstore_path = "vectorstores/pdf_index"
        os.makedirs(vectorstore_path, exist_ok=True)
        vectorstore.save_local(vectorstore_path)

        return {
            "message": "PDF processed and stored in vector DB",
            "num_chunks": len(chunks),
            "tables": extracted_tables,
            "num_images": len(extracted_images),
            "vectorstore_path": vectorstore_path
        }

    except Exception as e:
        return {"error": str(e)}
@app.get("/search/")
async def search_vector(query: str):
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local("vectorstores/pdf_index", embeddings=embedding_model)
    results = vectorstore.similarity_search(query, k=3)
    return {"results": [r.page_content for r in results]}

