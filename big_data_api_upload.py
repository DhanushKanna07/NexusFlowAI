from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
import uuid
import tempfile
import os
import fitz
import pdfplumber
import base64

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

app = FastAPI()

VECTOR_STORE_CACHE = {}
IMAGE_TABLE_CACHE = {}

# ========= Upload PDF, Process, and Store in Memory ========= #
@app.post("/upload-to-memory")
async def upload_to_memory(file: UploadFile = File(...)):
    try:
        if not file.filename.lower().endswith(".pdf"):
            return JSONResponse(status_code=400, content={"error": "Only PDF files are supported"})

        session_id = str(uuid.uuid4())
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            pdf_path = tmp.name
            content = await file.read()
            tmp.write(content)

        extracted_text = ""
        extracted_tables = []
        extracted_images = []

        # Step 1: Extract text and tables
        with pdfplumber.open(pdf_path) as pdf:
            for page_index, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    extracted_text += text + "\n"

                tables = page.extract_tables()
                for t_index, table in enumerate(tables):
                    table_dict = []
                    headers = table[0]
                    for row in table[1:]:
                        row_dict = {headers[i]: row[i] for i in range(len(headers))}
                        table_dict.append(row_dict)
                    extracted_tables.append({
                        "page": page_index,
                        "table_name": f"table_page_{page_index}_{t_index}",
                        "data": table_dict
                    })

        # Step 2: Extract images
        doc = fitz.open(pdf_path)
        for page_index in range(len(doc)):
            for img_index, img in enumerate(doc.get_page_images(page_index)):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                base64_str = base64.b64encode(image_bytes).decode('utf-8')
                image_ext = base_image.get("ext", "png")
                extracted_images.append({
                    "filename": f"image_{page_index}_{img_index}.{image_ext}",
                    "extension": image_ext,
                    "base64": base64_str,
                    "page": page_index
                })
        doc.close()
        os.remove(pdf_path)

        # Step 3: Chunk + Embed
        if not extracted_text.strip():
            return {"error": "No textual content found in the PDF."}

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_text(extracted_text)

        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_texts(texts=chunks, embedding=embedding_model)

        VECTOR_STORE_CACHE[session_id] = vectorstore
        IMAGE_TABLE_CACHE[session_id] = {"images": extracted_images, "tables": extracted_tables}

        return {
            "message": "PDF processed and stored in memory.",
            "session_id": session_id,
            "num_chunks": len(chunks),
            "num_tables": len(extracted_tables),
            "num_images": len(extracted_images),
            "chunks": chunks,
            "tables": extracted_tables,
            "images": extracted_images
        }

    except Exception as e:
        return {"error": str(e)}
    

#This part is commented because it is used to check the output,whether it can be fetched or not from the vectordb can be fetched or not and it is working
    
# ========= Search in Memory ========= #
"""@app.get("/search-memory")
async def search_memory(query: str, session_id: str):
    try:
        if session_id not in VECTOR_STORE_CACHE:
            return {"error": "Invalid or expired session_id"}

        vectorstore = VECTOR_STORE_CACHE[session_id]
        results = vectorstore.similarity_search(query, k=3)
        return {"results": [r.page_content for r in results]}

    except Exception as e:
        return {"error": str(e)}

# ========= Preview Session Data ========= #
@app.get("/preview-session")
async def preview_session(session_id: str, limit: int = 3):
    try:
        if session_id not in VECTOR_STORE_CACHE:
            return {"error": "Invalid or expired session_id"}

        vectorstore = VECTOR_STORE_CACHE[session_id]
        documents = vectorstore.docstore._dict.values()
        chunks = [doc.page_content for doc in documents]

        preview = {
            "session_id": session_id,
            "num_chunks": len(chunks),
            "chunk_preview": chunks[:limit],
            "table_preview": [],
            "image_preview": []
        }

        session_data = IMAGE_TABLE_CACHE.get(session_id, {})
        tables = session_data.get("tables", [])[:limit]
        images = session_data.get("images", [])[:limit]

        for t in tables:
            preview["table_preview"].append({
                "table_name": t.get("table_name"),
                "first_row": t["data"][0] if t.get("data") else {}
            })

        for img in images:
            preview["image_preview"].append({
                "filename": img["filename"],
                "page": img["page"],
                "base64_preview": img["base64"][:100] + "..."
            })

        return preview

    except Exception as e:
        return {"error": str(e)}"""