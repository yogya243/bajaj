import os
import requests
import asyncio
import hashlib
import logging
import re
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List, Dict, Any
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
from fastapi.responses import JSONResponse
from pdf2image import convert_from_bytes
import pytesseract

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

# ---------- Load env ----------
load_dotenv()

AZURE_API_BASE = os.getenv("AZURE_API_BASE")
AZURE_API_KEY = os.getenv("AZURE_API_KEY")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION")
EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")
LLM_DEPLOYMENT = os.getenv("model")
API_TOKEN = os.getenv("API_TOKEN")

# ---------- FastAPI app ----------
app = FastAPI(title="Multilingual Document QnA", version="1.0.0")

# ---------- Pydantic models ----------
class AnalyzeRequest(BaseModel):
    documents: str
    questions: List[str]

class AnalyzeResponse(BaseModel):
    answers: List[str]

# ---------- Cache ----------
INDEX_CACHE: Dict[str, Any] = {}
CACHE_DIR = "faiss_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# ---------- Utility ----------
def get_doc_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def normalize_text(text: str) -> str:
    text = re.sub(r'-\n', '', text)
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

# ---------- File extraction ----------
def extract_text_from_pdf(file_bytes: bytes) -> str:
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        all_text_pages = []
        has_text = False

        for page_num, page in enumerate(doc, start=1):
            text_dict = page.get_text("dict")
            page_blocks = []
            for block in text_dict.get("blocks", []):
                if "lines" not in block:
                    continue
                block_lines = []
                for line in block["lines"]:
                    spans = [span.get("text", "").strip() for span in line.get("spans", [])]
                    line_text = " ".join([s for s in spans if s])
                    if line_text:
                        block_lines.append(line_text)
                if block_lines:
                    page_blocks.append(" ".join(block_lines))
            page_text = "\n\n".join(page_blocks)
            page_text = normalize_text(page_text)
            if page_text:
                has_text = True
            all_text_pages.append(f"[Page {page_num}]\n{page_text}")

        if not has_text:
            logger.info("No embedded text in PDF; OCR fallback.")
            images = convert_from_bytes(file_bytes)
            ocr_pages = []
            for i, img in enumerate(images, start=1):
                ocr = pytesseract.image_to_string(img)
                ocr_pages.append(f"[Page {i}]\n{normalize_text(ocr)}")
            return "\n\n".join(ocr_pages)

        return "\n\n".join(all_text_pages)
    except Exception as e:
        logger.exception("PDF extraction failed")
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {e}")

def extract_text_from_docx(file_bytes: bytes) -> str:
    try:
        tmp = "temp.docx"
        with open(tmp, "wb") as f:
            f.write(file_bytes)
        text = docx2txt.process(tmp)
        return normalize_text(text)
    except Exception as e:
        logger.exception("DOCX extraction failed")
        raise HTTPException(status_code=500, detail=f"Failed to process DOCX: {e}")

def extract_text_from_email(file_bytes: bytes) -> str:
    try:
        msg = BytesParser(policy=policy.default).parsebytes(file_bytes)
        body = msg.get_body(preferencelist=("plain", "html"))
        return normalize_text(body.get_content()) if body else ""
    except Exception as e:
        logger.exception("Email extraction failed")
        raise HTTPException(status_code=500, detail=f"Failed to process email: {e}")

def detect_file_type_and_extract(url: str) -> str:
    try:
        logger.info(f"Fetching document from URL: {url}")
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        file_bytes = resp.content
        content_type = (resp.headers.get("Content-Type") or "").lower()

        if "pdf" in content_type or url.lower().endswith(".pdf"):
            return extract_text_from_pdf(file_bytes)
        if "wordprocessingml" in content_type or url.lower().endswith(".docx"):
            return extract_text_from_docx(file_bytes)
        if "message" in content_type:
            return extract_text_from_email(file_bytes)
        return extract_text_from_pdf(file_bytes)
    except Exception as e:
        logger.exception("Failed to fetch or extract document")
        raise HTTPException(status_code=500, detail=f"Failed to fetch document from URL: {e}")

# ---------- Language detection ----------
def detect_language(text: str) -> str:
    if re.search(r'[\u0D00-\u0D7F]', text):
        return "ml"
    return "en"

# ---------- LLM translation ----------
async def llm_translate_async(text: str, target_lang: str) -> str:
    llm = AzureChatOpenAI(
        temperature=0,
        deployment_name=LLM_DEPLOYMENT,
        azure_endpoint=AZURE_API_BASE,
        openai_api_version=AZURE_API_VERSION,
        openai_api_key=AZURE_API_KEY,
        max_tokens=500
    )
    prompt = f"Translate the following text to {target_lang} without adding anything else:\n\n{text}"
    return await asyncio.to_thread(llm.predict, prompt)

# ---------- Prompt ----------
GENERIC_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are an expert document analyst.
Answer strictly based on the CONTEXT below. Do not hallucinate. If info missing, say "Information not available in the provided document."

CONTEXT:
{context}

QUESTION:
{question}

INSTRUCTIONS:
- Quote exact policy/paragraph text if relevant, then give a short interpretation.
- For time periods, amounts, or explicit conditions, give exact figures and clause references if present.
- Keep answers concise and factual.

FINAL ANSWER:"""
)

# ---------- Text splitting ----------
def smart_text_splitter(document_text: str) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", "; ", " "],
        length_function=len
    )
    return [
        Document(page_content=chunk.strip(), metadata={"chunk_id": idx})
        for idx, chunk in enumerate(splitter.split_text(document_text))
        if len(chunk.strip()) > 50
    ]

# ---------- FAISS ----------
def build_faiss_from_chunks(chunks: List[Document]) -> FAISS:
    embeddings = AzureOpenAIEmbeddings(
        deployment=EMBEDDING_DEPLOYMENT,
        model=EMBEDDING_DEPLOYMENT,
        azure_endpoint=AZURE_API_BASE,
        openai_api_version=AZURE_API_VERSION,
        openai_api_key=AZURE_API_KEY
    )
    return FAISS.from_documents(chunks, embeddings)

def load_faiss_from_disk(path: str) -> FAISS:
    embeddings = AzureOpenAIEmbeddings(
        deployment=EMBEDDING_DEPLOYMENT,
        model=EMBEDDING_DEPLOYMENT,
        azure_endpoint=AZURE_API_BASE,
        openai_api_version=AZURE_API_VERSION,
        openai_api_key=AZURE_API_KEY
    )
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)

# ---------- Chain with cache ----------
def get_chain_with_cache(document_text: str):
    doc_hash = get_doc_hash(document_text)
    if doc_hash in INDEX_CACHE:
        return INDEX_CACHE[doc_hash]
    cache_path = os.path.join(CACHE_DIR, doc_hash)
    if os.path.exists(cache_path):
        faiss_index = load_faiss_from_disk(cache_path)
    else:
        chunks = smart_text_splitter(document_text)
        faiss_index = build_faiss_from_chunks(chunks)
        faiss_index.save_local(cache_path)
    chain = RetrievalQA.from_chain_type(
        llm=AzureChatOpenAI(
            temperature=0.1,
            deployment_name=LLM_DEPLOYMENT,
            azure_endpoint=AZURE_API_BASE,
            openai_api_version=AZURE_API_VERSION,
            openai_api_key=AZURE_API_KEY,
            max_tokens=500
        ),
        retriever=faiss_index.as_retriever(search_type="mmr", search_kwargs={"k": 10}),
        return_source_documents=False,
        chain_type_kwargs={"prompt": GENERIC_PROMPT}
    )
    INDEX_CACHE[doc_hash] = chain
    return chain

# ---------- QnA ----------
async def ask_question(query: str, chain) -> str:
    lang = detect_language(query)
    translated_q = await llm_translate_async(query, "English") if lang != "en" else query
    res = await asyncio.to_thread(chain.run, translated_q)
    return await llm_translate_async(res, "Malayalam") if lang == "ml" else res

async def process_questions(questions: List[str], chain):
    return await asyncio.gather(*(ask_question(q, chain) for q in questions))

# ---------- Endpoints ----------
@app.post("/api/v1/hackrx/run", response_model=AnalyzeResponse)
async def analyze_from_url(req: AnalyzeRequest, request: Request):
    urls = [u.strip() for u in req.documents.split(",") if u.strip()]
    all_texts = [detect_file_type_and_extract(url) for url in urls]
    combined_text = "\n\n".join(all_texts)
    chain = get_chain_with_cache(combined_text)
    answers_out = await process_questions(req.questions, chain)
    return AnalyzeResponse(answers=answers_out)

@app.get("/ping")
def ping():
    return JSONResponse(content={"status": "ok", "message": "Service is alive"})

@app.get("/")
def root():
    return {"message": "Multilingual Document QnA running"}
