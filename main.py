import os
import requests
import asyncio
import hashlib
import gc
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
from pdf2image import convert_from_bytes
import pytesseract

load_dotenv()

app = FastAPI(
    title="AI Insurance Analyzer API",
    description="Batch Q&A over insurance/legal/HR documents using LLMs",
    version="2.0.0"
)

class AnalyzeRequest(BaseModel):
    documents: str
    questions: List[str]

class AnalyzeResponse(BaseModel):
    answers: List[str]

INDEX_CACHE = {}
CACHE_DIR = "faiss_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def get_doc_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()

def normalize_text(text: str) -> str:
    text = re.sub(r'-\n', '', text)
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

def extract_text_from_pdf_streaming(file_bytes: bytes) -> str:
    """Process PDF page-by-page to avoid memory bloat."""
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        all_pages = []
        for page_num, page in enumerate(doc, 1):
            data = page.get_text("text")
            if not data.strip():
                img = page.get_pixmap()
                ocr_text = pytesseract.image_to_string(img.tobytes(), lang="eng")
                data = ocr_text
            clean_text = normalize_text(data)
            if clean_text:
                all_pages.append(f"[Page {page_num}]\n{clean_text}")
            del data, clean_text
            gc.collect()
        doc.close()
        return "\n\n".join(all_pages)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF extraction failed: {e}")

def extract_text_from_docx(file_bytes: bytes) -> str:
    try:
        with open("temp.docx", "wb") as f:
            f.write(file_bytes)
        return normalize_text(docx2txt.process("temp.docx"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DOCX extraction failed: {e}")

def extract_text_from_email(file_bytes: bytes) -> str:
    try:
        msg = BytesParser(policy=policy.default).parsebytes(file_bytes)
        return normalize_text(msg.get_body(preferencelist=('plain', 'html')).get_content())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Email extraction failed: {e}")

def detect_file_type_and_extract(url: str) -> str:
    try:
        r = requests.get(url, stream=True)
        r.raise_for_status()
        file_bytes = r.content
        content_type = r.headers.get("Content-Type", "")
        if "pdf" in content_type:
            return extract_text_from_pdf_streaming(file_bytes)
        elif "wordprocessingml" in content_type:
            return extract_text_from_docx(file_bytes)
        elif "message" in content_type or content_type == "application/octet-stream":
            return extract_text_from_email(file_bytes)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document fetch failed: {e}")

INSURANCE_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are an intelligent assistant for policy/legal/HR documents.\n"
        "Based ONLY on the following excerpts:\n"
        "{context}\n\n"
        "Question: {question}\n\n"
        "Give a concise, factual answer strictly from the text. "
        "Do not assume or invent. Quote relevant clauses if needed."
    )
)

def build_faiss_index(chunks: List[Document], use_large=True) -> FAISS:
    model_name = "text-embedding-3-large" if use_large else "text-embedding-3-small"
    embeddings = AzureOpenAIEmbeddings(
        deployment=os.getenv("AZURE_EMBEDDING_DEPLOYMENT"),
        model=model_name,
        azure_endpoint=os.getenv("AZURE_API_BASE"),
        openai_api_version=os.getenv("AZURE_API_VERSION"),
        openai_api_key=os.getenv("AZURE_API_KEY")
    )
    return FAISS.from_documents(chunks, embeddings)

def save_faiss(faiss_index: FAISS, path: str):
    faiss_index.save_local(path)
    del faiss_index
    gc.collect()

def load_faiss(path: str, use_large=True) -> FAISS:
    model_name = "text-embedding-3-large" if use_large else "text-embedding-3-small"
    embeddings = AzureOpenAIEmbeddings(
        deployment=os.getenv("AZURE_EMBEDDING_DEPLOYMENT"),
        model=model_name,
        azure_endpoint=os.getenv("AZURE_API_BASE"),
        openai_api_version=os.getenv("AZURE_API_VERSION"),
        openai_api_key=os.getenv("AZURE_API_KEY")
    )
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)

def create_chain(faiss_index: FAISS):
    llm = AzureChatOpenAI(
        temperature=0,
        deployment_name=os.getenv("model"),
        azure_endpoint=os.getenv("AZURE_API_BASE"),
        openai_api_version=os.getenv("AZURE_API_VERSION"),
        openai_api_key=os.getenv("AZURE_API_KEY")
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=faiss_index.as_retriever(search_kwargs={"k": 6}),
        chain_type_kwargs={"prompt": INSURANCE_PROMPT},
        return_source_documents=False
    )

def get_chain_with_cache(doc_text: str):
    doc_hash = get_doc_hash(doc_text)
    if doc_hash in INDEX_CACHE:
        return INDEX_CACHE[doc_hash]

    cache_path = os.path.join(CACHE_DIR, doc_hash)
    use_large = len(doc_text) < 500_000  # heuristic for memory
    if os.path.exists(cache_path):
        chain = create_chain(load_faiss(cache_path, use_large))
        INDEX_CACHE[doc_hash] = chain
        return chain

    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    chunks = [Document(page_content=txt, metadata={"id": i}) 
              for i, txt in enumerate(splitter.split_text(doc_text))]
    faiss_index = build_faiss_index(chunks, use_large)
    save_faiss(faiss_index, cache_path)
    chain = create_chain(load_faiss(cache_path, use_large))
    INDEX_CACHE[doc_hash] = chain
    del chunks
    gc.collect()
    return chain

async def ask_question(q: str, chain) -> str:
    try:
        return (await asyncio.to_thread(chain.run, q)).strip()
    except:
        return "No answer found."

async def process_questions(questions: List[str], chain):
    results = []
    for q in questions:
        results.append(await ask_question(q, chain))
    return results

@app.post("/api/v1/hackrx/run", response_model=AnalyzeResponse)
async def analyze_from_url(req: AnalyzeRequest):
    doc_text = detect_file_type_and_extract(req.documents)
    chain = get_chain_with_cache(doc_text)
    answers = await process_questions(req.questions, chain)
    return AnalyzeResponse(answers=answers)

@app.get("/")
def root():
    return {"message": "LLM Query–Retrieval API running"}

@app.head("/ping")
async def head_ping():
    return Response(status_code=200)
