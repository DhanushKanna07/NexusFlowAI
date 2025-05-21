PDF Upload & Semantic Processing API
This FastAPI module allows users to upload PDF files, extract their content (text, tables, and images), and convert the extracted text into a vectorized format for knowledge-based querying.

What the Code Does
1. Upload & Store
Accepts .pdf files via POST request to /upload-to-memory.

Saves the file temporarily on disk for processing.

2. Text and Table Extraction
Uses pdfplumber to read each page and:

Extract raw text

Extract structured tables as row/column dictionaries

Each table is tagged with:

Page number

A generated name (e.g., table_page_2_0)

3. Image Extraction
Uses PyMuPDF (fitz) to detect and extract all embedded images.

Converts each image to a base64-encoded string for preview or rendering.

Captures metadata:

Filename

Extension

Page number

(Optional) Size in bytes

4. Text Chunking and Embedding
Uses LangChain’s RecursiveCharacterTextSplitter to break extracted text into 1000-character chunks with 100-character overlaps.

Embeds each chunk into a vector using HuggingFaceEmbeddings (MiniLM model).

Stores all vectors in an in-memory FAISS vector database, indexed by a unique session_id.

API Response
When a PDF is processed, the API responds with:

json
Copy
Edit
{
  "message": "PDF processed and stored in memory.",
  "session_id": "unique-session-id",
  "num_chunks": 120,
  "num_tables": 5,
  "num_images": 10,
  "chunks": [...],
  "tables": [...],
  "images": [...]
}
This response allows the frontend or other services to:

Query by session_id

Access text, tables, and images

Display extracted content

Run similarity-based searches later using the vector database



The code i commented fully (down) is the part where i checked in GET/Search where it can call the data from the data chunk.It successfully retrived the required chunk where i asked the question from the VectorDB