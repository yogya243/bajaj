import os
import requests
import asyncio
import hashlib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import fitz  # PyMuPDF
import docx2txt
from email import policy
from email.parser import BytesParser
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import AzureChatOpenAI
from langchain.chains import RetrievalQA
from langchain.schema.document import Document
from langchain_openai import AzureOpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from fastapi.responses import Response
import re

# Optional OCR dependencies (keep since you had them)
from pdf2image import convert_from_bytes
import pytesseract

# Load environment variables
load_dotenv()

app = FastAPI(
    title="AI Insurance Analyzer API",
    description="Batch Q&A over insurance documents using LLMs",
    version="2.0.0"
)

# ----------- Input/Output Models -----------
class AnalyzeRequest(BaseModel):
    documents: str  # URL to document
    questions: List[str]  # No limit enforced

class AnalyzeResponse(BaseModel):
    answers: List[str]

# ----------- Cache Settings -----------
INDEX_CACHE = {}
CACHE_DIR = "faiss_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def get_doc_hash(text: str) -> str:
    """Hash document text for caching."""
    return hashlib.md5(text.encode()).hexdigest()

# ----------- File Extraction Helpers -----------
def normalize_text(text: str) -> str:
    """Fix hyphenated line breaks and normalize spaces."""
    text = re.sub(r'-\n', '', text)       # merge hyphenated words
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\n+', '\n', text)     # collapse multiple newlines
    text = re.sub(r'[ \t]+', ' ', text)   # normalize spaces
    return text.strip()

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract PDF text preserving simple table structure and handling scanned PDFs."""
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        all_text_pages = []
        has_text = False

        for page_num, page in enumerate(doc, 1):
            data = page.get_text("dict")
            page_content = []
            for block in data.get("blocks", []):
                if "lines" not in block or not block["lines"]:
                    continue

                # Collect line-level text and X positions (for naive column detection)
                line_texts = []
                x_positions = []
                row_cells = []

                for line in block["lines"]:
                    # combine spans for the line
                    spans = line.get("spans", [])
                    if not spans:
                        continue
                    # use x coord of first span for simple column detection
                    x_positions.append(round(spans[0]["bbox"][0], -1))
                    cell_text = " ".join(span.get("text", "").strip() for span in spans if span.get("text", "").strip())
                    # Normalize cell text
                    cell_text = normalize_text(cell_text)
                    row_cells.append(cell_text)

                # If multiple distinct x positions, assume table-like block
                unique_x = set(x_positions)
                if len(unique_x) > 1:
                    # Emit table rows with pipe separator to preserve row/column relationship
                    page_content.append("[TABLE START]")
                    # create a single "row" representation from the block's row_cells
                    # split into rows heuristically by newline markers (here each 'row_cells' is from a line)
                    # but since we aggregated whole block, we just emit the row as single pipe-joined cell
                    # (the splitter below will further cut into chunks)
                    page_content.append(" | ".join(c for c in row_cells if c))
                    page_content.append("[TABLE END]")
                else:
                    # Normal prose block - join lines into paragraphs
                    paragraph = " ".join(c for c in row_cells if c)
                    if paragraph:
                        page_content.append(paragraph)

            clean_page_text = normalize_text("\n".join(page_content))
            if clean_page_text:
                has_text = True
            all_text_pages.append(f"[Page {page_num}]\n{clean_page_text}")

        # If no extractable text (likely scanned), fallback to OCR per page
        if not has_text:
            images = convert_from_bytes(file_bytes)
            ocr_texts = []
            for i, img in enumerate(images, start=1):
                ocr_text = pytesseract.image_to_string(img)
                ocr_texts.append(f"[Page {i}]\n{normalize_text(ocr_text)}")
            return "\n\n".join(ocr_texts)

        return "\n\n".join(all_text_pages)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {e}")

def extract_text_from_docx(file_bytes: bytes) -> str:
    try:
        with open("temp.docx", "wb") as f:
            f.write(file_bytes)
        return normalize_text(docx2txt.process("temp.docx"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process DOCX: {e}")

def extract_text_from_email(file_bytes: bytes) -> str:
    try:
        msg = BytesParser(policy=policy.default).parsebytes(file_bytes)
        body = msg.get_body(preferencelist=('plain', 'html'))
        return normalize_text(body.get_content()) if body else ""
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process Email: {e}")

def detect_file_type_and_extract(url: str) -> str:
    try:
        session = requests.Session()
        response = session.get(url)
        response.raise_for_status()
        file_bytes = response.content
        content_type = response.headers.get("Content-Type", "").lower()

        if "pdf" in content_type or url.lower().endswith(".pdf"):
            return extract_text_from_pdf(file_bytes)
        elif "wordprocessingml" in content_type or url.lower().endswith(".docx"):
            return extract_text_from_docx(file_bytes)
        elif "message" in content_type or content_type == "application/octet-stream":
            return extract_text_from_email(file_bytes)
        else:
            # Fallback: try PDF extraction anyway
            return extract_text_from_pdf(file_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch document from URL: {e}")

# ----------- Prompt (accuracy-focused) -----------
INSURANCE_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an insurance policy assistant.
Use ONLY the following excerpts from the policy:
{context}

Question: "{question}"

Answer using the exact wording from the context wherever possible.
If the clause contains numbers, reproduce them exactly as in the text (including both words and digits if given).
Do not paraphrase unless the exact wording cannot answer the question.
Do not add details that are not in the provided context.
Provide the answer in one concise sentence.
"""
)

# ----------- FAISS Utilities (use text-embedding-3-large) -----------
EMBEDDING_MODEL_NAME = "text-embedding-3-large"

def build_faiss_from_chunks(chunks: List[Document]) -> FAISS:
    embeddings = AzureOpenAIEmbeddings(
        deployment=os.getenv("AZURE_EMBEDDING_DEPLOYMENT"),
        model=EMBEDDING_MODEL_NAME,
        azure_endpoint=os.getenv("AZURE_API_BASE"),
        openai_api_version=os.getenv("AZURE_API_VERSION"),
        openai_api_key=os.getenv("AZURE_API_KEY")
    )
    return FAISS.from_documents(chunks, embeddings)

def save_faiss_to_disk(faiss_index: FAISS, path: str):
    faiss_index.save_local(path)

def load_faiss_from_disk(path: str) -> FAISS:
    embeddings = AzureOpenAIEmbeddings(
        deployment=os.getenv("AZURE_EMBEDDING_DEPLOYMENT"),
        model=EMBEDDING_MODEL_NAME,
        azure_endpoint=os.getenv("AZURE_API_BASE"),
        openai_api_version=os.getenv("AZURE_API_VERSION"),
        openai_api_key=os.getenv("AZURE_API_KEY")
    )
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)

# ----------- LangChain Chain Creation -----------
def create_langchain_chain_with_prompt(faiss_index: FAISS):
    llm = AzureChatOpenAI(
        temperature=0,
        deployment_name=os.getenv("model"),
        azure_endpoint=os.getenv("AZURE_API_BASE"),
        openai_api_version=os.getenv("AZURE_API_VERSION"),
        openai_api_key=os.getenv("AZURE_API_KEY")
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=faiss_index.as_retriever(search_kwargs={"k": 8, "search_type": "mmr"}),
        return_source_documents=False,
        chain_type_kwargs={"prompt": INSURANCE_PROMPT}
    )

def get_chain_with_cache(document_text: str):
    doc_hash = get_doc_hash(document_text)

    if doc_hash in INDEX_CACHE:
        return INDEX_CACHE[doc_hash]

    cache_path = os.path.join(CACHE_DIR, doc_hash)
    if os.path.exists(cache_path):
        faiss_index = load_faiss_from_disk(cache_path)
        chain = create_langchain_chain_with_prompt(faiss_index)
        INDEX_CACHE[doc_hash] = chain
        return chain

    # Smarter semantic chunking with smaller chunks (more precise retrieval)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " "]
    )
    # Split the document and preserve table-marked segments as documents with metadata
    raw_chunks = splitter.split_text(document_text)
    documents = []
    for idx, chunk_text in enumerate(raw_chunks, start=1):
        # detect if chunk is a table chunk or contains table markers
        is_table = "[TABLE START]" in chunk_text or "[TABLE END]" in chunk_text or "|" in chunk_text
        meta = {"chunk_id": idx, "is_table": bool(is_table)}
        documents.append(Document(page_content=chunk_text, metadata=meta))

    faiss_index = build_faiss_from_chunks(documents)
    save_faiss_to_disk(faiss_index, cache_path)
    chain = create_langchain_chain_with_prompt(faiss_index)
    INDEX_CACHE[doc_hash] = chain
    return chain

# ----------- QA Logic (Parallel, Unlimited) -----------
async def ask_question(query: str, chain) -> str:
    try:
        result = await asyncio.to_thread(chain.run, query)
        return result.strip()
    except Exception as e:
        return f"Error answering question: {e}"

async def process_in_batches(questions: List[str], chain, batch_size: int = 20):
    """Process questions in smaller batches to avoid memory overload."""
    results = []
    for i in range(0, len(questions), batch_size):
        batch = questions[i:i+batch_size]
        batch_results = await asyncio.gather(*[ask_question(q, chain) for q in batch])
        results.extend(batch_results)
    return results

# ----------- API Endpoints (unchanged functionality) -----------
@app.post("/api/v1/hackrx/run", response_model=AnalyzeResponse)
async def analyze_from_url(req: AnalyzeRequest):
    document_text = detect_file_type_and_extract(req.documents)
    chain = get_chain_with_cache(document_text)
    answers = await process_in_batches(req.questions, chain)
    return AnalyzeResponse(answers=answers)

@app.get("/")
def root():
    return {"message": "AI Insurance Document Analyzer is running"}

@app.head("/ping")
async def head_ping():
    return Response(status_code=200)
