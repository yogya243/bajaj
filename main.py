import os
import requests
import asyncio
import hashlib
import logging
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
from fastapi.responses import Response
import re
from pdf2image import convert_from_bytes
import pytesseract

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

load_dotenv()

AZURE_API_BASE = os.getenv("AZURE_API_BASE")
AZURE_API_KEY = os.getenv("AZURE_API_KEY")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION")
EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")
LLM_DEPLOYMENT = os.getenv("model")
API_TOKEN = os.getenv("API_TOKEN")

app = FastAPI(title="Multilingual Document QnA", version="1.0.0")

class AnalyzeRequest(BaseModel):
    documents: str
    questions: List[str]

INDEX_CACHE: Dict[str, Any] = {}
CACHE_DIR = "faiss_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def get_doc_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def normalize_text(text: str) -> str:
    text = re.sub(r'-\n', '', text)
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

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
            logger.info("No embedded text found in PDF; performing OCR fallback.")
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

def azure_translate(texts: List[str], to_lang: str = "en") -> List[Dict[str, Any]]:
    if not AZURE_API_BASE or not AZURE_API_KEY:
        raise RuntimeError("AZURE_API_BASE or AZURE_API_KEY not set for Translator")
    endpoint = AZURE_API_BASE.rstrip("/") + "/translate"
    params = {"api-version": "3.0", "to": to_lang}
    headers = {
        "Ocp-Apim-Subscription-Key": AZURE_API_KEY,
        "Content-Type": "application/json"
    }
    body = [{"text": t} for t in texts]
    resp = requests.post(endpoint, params=params, headers=headers, json=body, timeout=15)
    resp.raise_for_status()
    return resp.json()

def translate_to_english(text: str) -> (str, str):
    try:
        resp = azure_translate([text], to_lang="en")
        detected = resp[0].get("detectedLanguage", {}).get("language", "en")
        translated = resp[0]["translations"][0]["text"]
        return translated, detected
    except Exception as e:
        logger.warning(f"Translation to English failed: {e}")
        return text, "en"

def translate_from_english(text: str, target_lang: str) -> str:
    if target_lang == "en":
        return text
    try:
        resp = azure_translate([text], to_lang=target_lang)
        return resp[0]["translations"][0]["text"]
    except Exception as e:
        logger.warning(f"Translation from English failed: {e}")
        return text

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
- Keep answers concise and factual.

FINAL ANSWER:"""
)

def smart_text_splitter(document_text: str) -> List[Document]:
    section_pattern = r'\n(?=\d+\.?\s+[A-Z][^.]*[:\n])'
    major_sections = re.split(section_pattern, document_text)
    docs = []
    chunk_id = 0
    for s_idx, sec in enumerate(major_sections):
        if not sec.strip():
            continue
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", "; ", ", ", " "],
            length_function=len
        )
        for chunk in splitter.split_text(sec):
            if len(chunk.strip()) > 50:
                docs.append(Document(page_content=chunk.strip(), metadata={"chunk_id": chunk_id, "section": s_idx}))
                chunk_id += 1
    return docs

def build_faiss_from_chunks(chunks: List[Document]) -> FAISS:
    embeddings = AzureOpenAIEmbeddings(
        deployment=EMBEDDING_DEPLOYMENT,
        model=EMBEDDING_DEPLOYMENT,
        azure_endpoint=AZURE_API_BASE,
        openai_api_version=AZURE_API_VERSION,
        openai_api_key=AZURE_API_KEY
    )
    return FAISS.from_documents(chunks, embeddings)

def save_faiss_to_disk(faiss_index: FAISS, path: str):
    faiss_index.save_local(path)

def load_faiss_from_disk(path: str) -> FAISS:
    embeddings = AzureOpenAIEmbeddings(
        deployment=EMBEDDING_DEPLOYMENT,
        model=EMBEDDING_DEPLOYMENT,
        azure_endpoint=AZURE_API_BASE,
        openai_api_version=AZURE_API_VERSION,
        openai_api_key=AZURE_API_KEY
    )
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)

def create_retriever(faiss_index: FAISS):
    return faiss_index.as_retriever(search_type="mmr", search_kwargs={"k": 10, "lambda_mult": 0.7})

def create_chain_from_faiss(faiss_index: FAISS):
    llm = AzureChatOpenAI(
        temperature=0.1,
        deployment_name=LLM_DEPLOYMENT,
        azure_endpoint=AZURE_API_BASE,
        openai_api_version=AZURE_API_VERSION,
        openai_api_key=AZURE_API_KEY,
        max_tokens=500
    )
    retriever = create_retriever(faiss_index)
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=False,
                                       chain_type_kwargs={"prompt": GENERIC_PROMPT})

def get_chain_with_cache(document_text: str):
    doc_hash = get_doc_hash(document_text)
    if doc_hash in INDEX_CACHE:
        return INDEX_CACHE[doc_hash]
    cache_path = os.path.join(CACHE_DIR, doc_hash)
    if os.path.exists(cache_path):
        logger.info("Loading FAISS index from cache")
        faiss_index = load_faiss_from_disk(cache_path)
        chain = create_chain_from_faiss(faiss_index)
        INDEX_CACHE[doc_hash] = chain
        return chain
    chunks = smart_text_splitter(document_text)
    logger.info(f"Building FAISS from {len(chunks)} chunks")
    faiss_index = build_faiss_from_chunks(chunks)
    save_faiss_to_disk(faiss_index, cache_path)
    chain = create_chain_from_faiss(faiss_index)
    INDEX_CACHE[doc_hash] = chain
    return chain

async def ask_question_with_optional_translation(query: str, chain) -> str:
    try:
        translated_q, detected_lang = translate_to_english(query)
        use_query = translated_q if detected_lang != "en" else query
        res = await asyncio.to_thread(chain.run, use_query)
        final_answer = res
        if detected_lang != "en":
            final_answer = translate_from_english(res, detected_lang)
        return final_answer.strip()
    except Exception as e:
        logger.error(f"Error answering question '{query}': {e}")
        return "Information not available in the provided document."

async def process_in_batches(questions: List[str], chain) -> List[str]:
    results = []
    for i in range(0, len(questions), 12):
        batch = questions[i:i + 12]
        tasks = [ask_question_with_optional_translation(q, chain) for q in batch]
        batch_results = await asyncio.gather(*tasks, return_exceptions=False)
        results.extend(batch_results)
        if i + 12 < len(questions):
            await asyncio.sleep(0.4)
    return results

@app.post("/api/v1/hackrx/run")
async def analyze_from_url(req: AnalyzeRequest, request: Request):
    logger.info(f"Incoming request from {request.client.host if request.client else 'unknown'}")
    urls = [u.strip() for u in req.documents.split(",") if u.strip()]
    logger.info(f"📄 Received {len(urls)} document URLs")
    for idx, u in enumerate(urls, start=1):
        logger.info(f"   Doc{idx}: {u}")
    logger.info(f"📝 Received {len(req.questions)} questions")
    for idx, q in enumerate(req.questions, start=1):
        logger.info(f"   Q{idx}: {q}")
    all_texts = [detect_file_type_and_extract(url) for url in urls]
    if not any(all_texts):
        raise HTTPException(status_code=400, detail="No extractable text found")
    combined_text = "\n\n".join(all_texts)
    chain = get_chain_with_cache(combined_text)
    answers_out = await process_in_batches(req.questions, chain)
    logger.info(f"✅ Answers: {answers_out}")
    return {"answers": answers_out}

@app.get("/")
def root():
    return {"message": "Multilingual Document QnA running"}

@app.head("/ping")
def ping():
    return Response(status_code=200)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
