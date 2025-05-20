from fastapi import FastAPI, UploadFile, File
import fitz  # PyMuPDF
import pdfplumber
import base64
import os
import tempfile

app = FastAPI()

@app.post("/extract-pdf-data/")
async def extract_pdf_data(file: UploadFile = File(...)):
    try:
        # Create temp file safely
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            temp_pdf_path = tmp.name
            content = await file.read()
            tmp.write(content)

        extracted_tables = []
        extracted_images = []
        extracted_text = ""

        # --- Extract text and tables (pdfplumber) ---
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

        # --- Extract images (fitz) ---
        doc = fitz.open(temp_pdf_path)
        for page_index in range(len(doc)):
            for img_index, img in enumerate(doc.get_page_images(page_index)):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                base64_image = base64.b64encode(image_bytes).decode('utf-8')
                extracted_images.append(base64_image)
        doc.close()  # VERY IMPORTANT: Close fitz document before deleting file

        # Delete temp file (safe now)
        os.remove(temp_pdf_path)

        return {
            "tables": extracted_tables,
            "text": extracted_text.strip(),
            "images_base64": extracted_images,
        }

    except Exception as e:
        return {"error": str(e)}