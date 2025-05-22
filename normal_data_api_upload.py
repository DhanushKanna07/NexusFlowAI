from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import fitz  # PyMuPDF
import pdfplumber
import base64
import os
import tempfile
import uuid
import io
from PIL import Image

# LangChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Hugging Face - BLIP for image captioning
from transformers import BlipProcessor, BlipForConditionalGeneration

app = FastAPI()

# Load BLIP model only once (at app startup)
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# In-memory cache for session-based vectorstores
VECTOR_STORE_CACHE = {}

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

        # ---- STEP 3: Extract images and generate captions using BLIP ----
        doc = fitz.open(temp_pdf_path)
        for page_index in range(len(doc)):
            for img_index, img in enumerate(doc.get_page_images(page_index)):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image.get("ext", "png")
                base64_str = base64.b64encode(image_bytes).decode('utf-8')

                # Generate caption using BLIP
                image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                inputs = processor(images=image_pil, return_tensors="pt")
                out = model.generate(**inputs)
                caption = processor.decode(out[0], skip_special_tokens=True)

                extracted_images.append({
                    "filename": f"image_{page_index}_{img_index}.{image_ext}",
                    "extension": image_ext,
                    "base64": base64_str,
                    "size_bytes": len(image_bytes),
                    "page": page_index,
                    "metadata": caption
                })
        doc.close()
        os.remove(temp_pdf_path)

        # ---- STEP 4: Chunk + Embed text using LangChain ----
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_text(extracted_text)

        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_texts(texts=chunks, embedding=embedding_model)

        # Generate temporary ID for this session
        session_id = str(uuid.uuid4())
        VECTOR_STORE_CACHE[session_id] = vectorstore

        return {
            "message": "PDF processed and stored in memory with image captions",
            "session_id": session_id,
            "num_chunks": len(chunks),
            "chunks": chunks,
            "tables": extracted_tables,
            "num_images": len(extracted_images),
            "images": extracted_images
        }

    except Exception as e:
        return {"error": str(e)}



    
#This part is commented because it is used to check the output, whether it can be fetched or not from the vectordb and it is working

# Request model for /ask endpoint
"""class AskRequest(BaseModel):
    session_id: str
    question: str

@app.post("/ask/")
async def ask_question(request: AskRequest):
    try:
        session_id = request.session_id
        question = request.question

        if session_id not in VECTOR_STORE_CACHE:
            return {"error": "Invalid or expired session_id"}

        # Retrieve vectorstore and set up retriever
        vectorstore = VECTOR_STORE_CACHE[session_id]
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        # Setup Ollama with LLaMA 2 model
        llm = Ollama(model="llama2", temperature=0)

        # Create RetrievalQA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )

        result = qa_chain({"query": question})

        return {
            "question": question,
            "answer": result["result"],
            "sources": [doc.page_content for doc in result["source_documents"]]
        }

    except Exception as e:
        return {"error": str(e)}"""