import os
import requests
import asyncio
import hashlib
import logging
from fastapi import FastAPI, HTTPException, Request
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
from fastapi.responses import PlainTextResponse
from bs4 import BeautifulSoup
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

load_dotenv()

AZURE_API_BASE = os.getenv("AZURE_API_BASE")
AZURE_API_KEY = os.getenv("AZURE_API_KEY")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION")
EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")
LLM_DEPLOYMENT = os.getenv("model")

app = FastAPI()

class AnalyzeRequest(BaseModel):
    documents: str
    questions: List[str]

class AnalyzeResponse(BaseModel):
    answers: List[str]

INDEX_CACHE = {}
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
            text = page.get_text().strip()
            if text:
                has_text = True
                all_text_pages.append(text)
        if not has_text:
            images = convert_from_bytes(file_bytes)
            for img in images:
                ocr_text = pytesseract.image_to_string(img)
                all_text_pages.append(ocr_text)
        return normalize_text("\n\n".join(all_text_pages))
    except Exception as e:
        logger.exception("PDF extraction failed")
        raise HTTPException(status_code=500, detail=f"PDF processing failed: {e}")

def extract_text_from_docx(file_bytes: bytes) -> str:
    tmp = "temp.docx"
    with open(tmp, "wb") as f:
        f.write(file_bytes)
    text = docx2txt.process(tmp)
    return normalize_text(text)

def extract_text_from_email(file_bytes: bytes) -> str:
    msg = BytesParser(policy=policy.default).parsebytes(file_bytes)
    body = msg.get_body(preferencelist=("plain", "html"))
    return normalize_text(body.get_content()) if body else ""

def detect_file_type_and_extract(url: str) -> str:
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

def create_llm(temp=0.1, max_tokens=500):
    return AzureChatOpenAI(
        temperature=temp,
        deployment_name=LLM_DEPLOYMENT,
        azure_endpoint=AZURE_API_BASE,
        openai_api_version=AZURE_API_VERSION,
        openai_api_key=AZURE_API_KEY,
        max_tokens=max_tokens
    )

def translate_text(text: str, target_lang: str = "en") -> str:
    if not text.strip():
        return text
    if target_lang == "en":
        prompt = f"Translate the following text to English preserving meaning:\n{text}"
    else:
        prompt = f"Translate the following English text to {target_lang} preserving meaning:\n{text}"
    llm = create_llm()
    try:
        result = llm.predict(prompt)
        return result.strip()
    except Exception as e:
        logger.warning(f"Translation failed: {e}")
        return text

def detect_language(text: str) -> str:
    prompt = f"Detect the language code (like 'en', 'ml', 'hi') of the following text and return ONLY the code:\n{text[:200]}"
    llm = create_llm()
    try:
        result = llm.predict(prompt)
        return result.strip().lower()
    except:
        return "en"

GENERIC_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are an expert document analyst.
Answer strictly based on the CONTEXT below...
FINAL ANSWER:"""
)

def smart_text_splitter(document_text: str) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    return [Document(page_content=chunk) for chunk in splitter.split_text(document_text) if chunk.strip()]

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

def get_chain_with_cache(document_text: str):
    doc_hash = get_doc_hash(document_text)
    if doc_hash in INDEX_CACHE:
        return INDEX_CACHE[doc_hash]
    cache_path = os.path.join(CACHE_DIR, doc_hash)
    if os.path.exists(cache_path):
        faiss_index = load_faiss_from_disk(cache_path)
        chain = RetrievalQA.from_chain_type(llm=create_llm(), retriever=faiss_index.as_retriever(), chain_type_kwargs={"prompt": GENERIC_PROMPT})
        INDEX_CACHE[doc_hash] = chain
        return chain
    chunks = smart_text_splitter(document_text)
    faiss_index = build_faiss_from_chunks(chunks)
    faiss_index.save_local(cache_path)
    chain = RetrievalQA.from_chain_type(llm=create_llm(), retriever=faiss_index.as_retriever(), chain_type_kwargs={"prompt": GENERIC_PROMPT})
    INDEX_CACHE[doc_hash] = chain
    return chain

async def ask_question(query: str, chain) -> str:
    lang = detect_language(query)
    eng_query = translate_text(query, target_lang="en") if lang != "en" else query
    try:
        result = await asyncio.to_thread(chain.run, eng_query)
    except Exception as e:
        result = f"Error answering question: {e}"
    if lang != "en":
        result = translate_text(result, target_lang=lang)
    return result.strip()

# --- city/landmark parsing ---
def parse_city_landmark_pairs(doc_text: str) -> dict:
    mapping = {}
    lines = doc_text.splitlines()
    for line in lines:
        cleaned = re.sub(r'^[^\w]+', '', line).strip()
        if not cleaned:
            continue
        if re.search(r'Landmark Current Location', cleaned, re.I):
            continue
        m = re.match(r'(.+?)\s+([A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*)*)\s*$', cleaned)
        if m:
            landmark = m.group(1).strip()
            city = m.group(2).strip()
            mapping[city.lower()] = landmark
    return mapping

def parse_landmark_endpoints(doc_text: str) -> (dict, str):
    landmark_to_endpoint = {}
    pattern = re.compile(r'If\s+landmark.*?is\s*(?:"|“)?([^"”\n,]+?)(?:"|”)?\s*[,:\n].*?GET\s*(https?://\S+)', re.I | re.S)
    for lm, url in pattern.findall(doc_text):
        landmark_to_endpoint[lm.strip()] = url.strip()
    m_def = re.search(r'For\s+all\s+other\s+landmarks.*?GET\s*(https?://\S+)', doc_text, re.I | re.S)
    if m_def:
        landmark_to_endpoint["DEFAULT"] = m_def.group(1).strip()
    fav_m = re.search(r'GET\s*(https?://register\.hackrx\.in/submissions/myFavouriteCity)', doc_text, re.I)
    fav_url = fav_m.group(1).strip() if fav_m else "https://register.hackrx.in/submissions/myFavouriteCity"
    if not landmark_to_endpoint:
        for match in re.findall(r'(https?://register\.hackrx\.in/teams/public/flights/\S+)', doc_text, re.I):
            landmark_to_endpoint.setdefault("DEFAULT", match)
    return landmark_to_endpoint, fav_url

def extract_flightnumber_from_response(resp: requests.Response) -> str:
    text = resp.text.strip()
    try:
        parsed = resp.json()
        if isinstance(parsed, dict):
            for key in ("flightNumber", "flight_number", "flight", "number"):
                if key in parsed:
                    val = parsed[key]
                    if isinstance(val, (str, int)):
                        return str(val).strip()
            data = parsed.get("data")
            if isinstance(data, dict):
                for key in ("flightNumber", "flight_number", "flight", "number"):
                    if key in data:
                        val = data[key]
                        if isinstance(val, (str, int)):
                            return str(val).strip()
            if isinstance(parsed, (str, int)):
                return str(parsed).strip()
    except:
        pass
    m = re.search(r'([A-Z]{1,3}\-?\d{1,5}|[A-Z0-9\-]{2,20})', text)
    if m:
        return m.group(1).strip()
    return text

def _get_flight_number_via_api_sequence(doc_text: str) -> str:
    try:
        city_to_landmark = parse_city_landmark_pairs(doc_text)
        landmark_to_endpoint, fav_url = parse_landmark_endpoints(doc_text)
        fav_resp = requests.get(fav_url, timeout=10)
        fav_resp.raise_for_status()
        city_name = None
        try:
            fav_json = fav_resp.json()
            if isinstance(fav_json, dict):
                if "data" in fav_json and isinstance(fav_json["data"], dict) and "city" in fav_json["data"]:
                    city_name = fav_json["data"]["city"]
                elif "city" in fav_json:
                    city_name = fav_json["city"]
        except:
            pass
        if not city_name:
            city_name = fav_resp.text.strip()
        city_norm = re.sub(r'[^A-Za-z\s]', '', (city_name or "")).strip().lower()
        landmark = city_to_landmark.get(city_norm)
        if not landmark:
            city_title = " ".join([w.capitalize() for w in city_norm.split()])
            m = re.search(r'([A-Za-z][A-Za-z\s\'\-]{2,60}?)\s+' + re.escape(city_title) + r'\b', doc_text)
            if m:
                landmark = m.group(1).strip()
        endpoint = None
        if landmark:
            for lm_key, url in landmark_to_endpoint.items():
                if lm_key.lower() == landmark.lower():
                    endpoint = url
                    break
            if not endpoint:
                for lm_key, url in landmark_to_endpoint.items():
                    if lm_key.lower() in landmark.lower() or landmark.lower() in lm_key.lower():
                        endpoint = url
                        break
        if not endpoint:
            endpoint = landmark_to_endpoint.get("DEFAULT")
        if not endpoint:
            return None
        flight_resp = requests.get(endpoint, timeout=10)
        flight_resp.raise_for_status()
        return extract_flightnumber_from_response(flight_resp)
    except:
        return None

async def get_raw_content_if_api(url: str):
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        content_type = (resp.headers.get("Content-Type") or "").lower()
        text = resp.text
        if "text/html" in content_type:
            soup = BeautifulSoup(text, "html.parser")
            token_tag = soup.find("div", id="token")
            if token_tag:
                token = token_tag.get_text(strip=True)
                if token:
                    return token
            return text
        if "application/json" in content_type or "text/plain" in content_type or "text/" in content_type:
            return text
        return None
    except:
        return None

@app.post("/api/v1/hackrx/run", response_model=AnalyzeResponse)
async def analyze_from_url(req: AnalyzeRequest, request: Request):
    urls = [u.strip() for u in req.documents.split(",") if u.strip()]
    all_texts = []
    for url in urls:
        raw_text = await get_raw_content_if_api(url)
        if raw_text:
            all_texts.append(raw_text)
            continue
        text = await asyncio.to_thread(detect_file_type_and_extract, url)
        if text.strip():
            all_texts.append(text)
    if not all_texts:
        raise HTTPException(status_code=400, detail="No extractable text found in provided URLs")
    combined_text = "\n\n".join(all_texts)
    doc_lang = detect_language(combined_text)
    if doc_lang != "en":
        combined_text = translate_text(combined_text, target_lang="en")
    chain = get_chain_with_cache(combined_text)

    answers = []
    for q in req.questions:
        q_lower = q.lower()

        # Secret token flow
        if "secret token" in q_lower:
            try:
                m = re.search(r'GET\s+(https?://[^\s]+/getSecretToken[^\s]*)', combined_text, re.I)
                token_url = m.group(1) if m else "https://register.hackrx.in/submissions/getSecretToken"
                token_resp = requests.get(token_url, timeout=10)
                token_resp.raise_for_status()
                token_val = token_resp.text.strip()
                try:
                    token_json = token_resp.json()
                    if isinstance(token_json, dict):
                        token_val = token_json.get("token", token_val)
                except:
                    pass
                answers.append(str(token_val))
                continue
            except:
                answers.append("Information not available in the provided document.")
                continue

        # Flight number flow
        if "flight number" in q_lower or ("flight" in q_lower and "number" in q_lower):
            flight_val = _get_flight_number_via_api_sequence(combined_text)
            if flight_val:
                answers.append(flight_val)
            else:
                ans = await ask_question(q, chain)
                answers.append(ans)
            continue

        ans = await ask_question(q, chain)
        answers.append(ans)

    out = {"answers": answers}
    logger.info(f"📤 Response JSON returned to user: {json.dumps(out, ensure_ascii=False)}")
    return AnalyzeResponse(answers=answers)

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
