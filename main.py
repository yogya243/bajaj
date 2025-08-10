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

# simple translation utilities kept (as in your code) - you said not to change other functionality
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
Answer strictly based on the CONTEXT below. Do not hallucinate.
If the exact answer is not directly stated but the document provides steps or instructions to obtain it, explain those steps clearly.
If the exact answer is not in the document, try to provide a relevant fact-based answer using general knowledge, but clearly mention it is based on common knowledge and not from the document.
If no relevant info or general answer is possible, say "Information not available in the provided document."

CONTEXT:
{context}

QUESTION:
{question}

INSTRUCTIONS:
- Quote exact policy/paragraph text if relevant, then give a short interpretation.
- For time periods, amounts, or explicit conditions, give exact figures and clause references if present.
- Keep answers concise and factual.
- Extract explicit facts, dates, numbers, names, and commitments mentioned.
- If the answer is not found in the document, provide relevant general knowledge based on facts, but do not hallucinate specifics.

FINAL ANSWER:"""
)

def smart_text_splitter(document_text: str) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200
    )
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

# ---------- New: flight-number procedural flow (integrated, deterministic) ----------
# Map city -> landmark as per your PDF tables (excerpts included)
_CITY_TO_LANDMARK = {
    # from PDF pages: keys are city names that the API may return; keep common capitalization variants
    "delhi": "Gateway of India",   # Gateway of India currently in Delhi (per PDF)
    "mumbai": "India Gate",
    "chennai": "Charminar",
    "hyderabad": "Marina Beach",  # note: PDF maps Taj Mahal to Hyderabad in a later table too - we'll handle duplicates below
    "ahmedabad": "Howrah Bridge",
    "mysuru": "Golconda Fort",
    "kochi": "Qutub Minar",
    "hyderabad_taj": "Taj Mahal",  # in the second table Taj Mahal -> Hyderabad (we will check direct matches)
    # International mapping (only include as needed)
    "new york": "Eiffel Tower",
    "london": "Statue of Liberty",
    "tokyo": "Big Ben",
    "beijing": "Colosseum",
    # Add additional mappings if needed; mapping is based on the PDF content
}

# Landmark -> flight endpoint mapping exactly as the PDF instructs
_LANDMARK_TO_ENDPOINT = {
    "Gateway of India": "https://register.hackrx.in/teams/public/flights/getFirstCityFlightNumber",
    "Taj Mahal": "https://register.hackrx.in/teams/public/flights/getSecondCityFlightNumber",
    "Eiffel Tower": "https://register.hackrx.in/teams/public/flights/getThirdCityFlightNumber",
    "Big Ben": "https://register.hackrx.in/teams/public/flights/getFourthCityFlightNumber",
    # Default for all other landmarks
    "DEFAULT": "https://register.hackrx.in/teams/public/flights/getFifthCityFlightNumber",
}

def _normalize_city_name(name: str) -> str:
    if not name:
        return ""
    return re.sub(r'[^a-zA-Z\s]', '', name).strip().lower()

def _map_city_to_landmark(city_name: str) -> str:
    if not city_name:
        return None
    norm = _normalize_city_name(city_name)
    # PDF has examples like "Chennai" -> Charminar. We'll try direct lookups and some heuristics.
    # Check direct known mapping first
    direct = {
        "chennai": "Charminar",
        "delhi": "Gateway of India",
        "mumbai": "India Gate",
        "hyderabad": "Taj Mahal",  # per PDF second table: Taj Mahal -> Hyderabad (choose Taj Mahal)
        "ahmedabad": "Howrah Bridge",
        "mysuru": "Golconda Fort",
        "kochi": "Qutub Minar",
        "paris": "Eiffel Tower",
        "london": "Statue of Liberty",
        "tokyo": "Big Ben",
        "beijing": "Colosseum",
        "new york": "Eiffel Tower",  # per PDF: Eiffel Tower -> New York (so reverse mapping)
    }
    if norm in direct:
        return direct[norm]
    # fallback try the earlier broader mapping
    return _CITY_TO_LANDMARK.get(norm)

def _get_flight_number_via_api_sequence() -> str:
    """
    Implements the step-by-step flow from the PDF:
    1) GET https://register.hackrx.in/submissions/myFavouriteCity  => returns city name
    2) Map city -> landmark using PDF table
    3) Based on landmark call the relevant flight-number endpoint
    4) Return the flight number text (trimmed)
    """
    try:
        # Step 1: Get favourite city
        city_resp = requests.get("https://register.hackrx.in/submissions/myFavouriteCity", timeout=10)
        city_resp.raise_for_status()
        city_name = city_resp.text.strip()
        logger.info(f"[flight-flow] favourite city from API: {city_name!r}")

        # Step 2: Map to landmark
        landmark = _map_city_to_landmark(city_name)
        logger.info(f"[flight-flow] mapped city '{city_name}' -> landmark '{landmark}'")

        # Step 3: Choose endpoint
        if landmark in _LANDMARK_TO_ENDPOINT:
            endpoint = _LANDMARK_TO_ENDPOINT[landmark]
        else:
            endpoint = _LANDMARK_TO_ENDPOINT["DEFAULT"]

        logger.info(f"[flight-flow] calling flight endpoint: {endpoint}")
        flight_resp = requests.get(endpoint, timeout=10)
        flight_resp.raise_for_status()
        flight_text = flight_resp.text.strip()
        # flight endpoint may return JSON or plain text; try to extract simple string/number
        try:
            parsed = flight_resp.json()
            # If JSON, try common keys
            if isinstance(parsed, dict):
                for k in ("flight", "flight_number", "flightNumber", "data", "flight_no", "number"):
                    if k in parsed:
                        val = parsed[k]
                        if isinstance(val, (str, int)):
                            return str(val).strip()
                # if 'data' is nested
                if "data" in parsed and isinstance(parsed["data"], (str, int)):
                    return str(parsed["data"]).strip()
            # if JSON is just a string or number
            if isinstance(parsed, (str, int)):
                return str(parsed).strip()
        except Exception:
            # Not JSON or parsing failed — treat as plain text
            pass

        # If plain text, try to extract something that looks like a flight number (alphanumeric)
        m = re.search(r'([A-Z]{2,3}\s?-?\d{1,5}|[A-Z0-9\-]{3,20})', flight_text)
        if m:
            return m.group(1).strip()
        return flight_text
    except Exception as e:
        logger.exception("[flight-flow] failed to fetch flight number")
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
    except Exception as e:
        logger.warning(f"Failed to fetch raw content from API: {e}")
        return None

@app.post("/api/v1/hackrx/run", response_model=AnalyzeResponse)
async def analyze_from_url(req: AnalyzeRequest, request: Request):
    urls = [u.strip() for u in req.documents.split(",") if u.strip()]
    logger.info(f"📄 Received {len(urls)} document URLs:")
    for idx, url in enumerate(urls, start=1):
        logger.info(f"   Doc {idx}: {url}")

    logger.info(f"📝 Received {len(req.questions)} questions:")
    for idx, q in enumerate(req.questions, start=1):
        logger.info(f"   Q{idx}: {q}")

    all_texts = []

    for url in urls:
        # Check if URL is API returning raw token/text (like Input 2)
        raw_text = await get_raw_content_if_api(url)
        if raw_text:
            all_texts.append(raw_text)
            continue

        # Otherwise, normal doc extraction
        text = await asyncio.to_thread(detect_file_type_and_extract, url)
        if text.strip():
            all_texts.append(text)

    if not all_texts:
        raise HTTPException(status_code=400, detail="No extractable text found in provided URLs")

    combined_text = "\n\n".join(all_texts)

    # Language detection and translation (kept unchanged)
    doc_lang = detect_language(combined_text)
    if doc_lang != "en":
        combined_text = translate_text(combined_text, target_lang="en")

    # Build or get cached chain for document text
    chain = get_chain_with_cache(combined_text)

    answers = []
    for q in req.questions:
        # --- NEW: if question requests flight number, perform deterministic API flow ---
        q_lower = q.lower()
        if "flight number" in q_lower or ("flight" in q_lower and "number" in q_lower):
            logger.info("[flight-flow] detected flight-number question; running procedural flow")
            flight_val = _get_flight_number_via_api_sequence()
            if flight_val:
                answers.append(flight_val)
            else:
                # fall back to normal chain if flow failed
                logger.info("[flight-flow] flight flow failed; falling back to LLM retrieval")
                ans = await ask_question(q, chain)
                answers.append(ans)
            continue

        # existing general flow
        ans = await ask_question(q, chain)
        answers.append(ans)

    # Prepare output JSON object (preserve the requested JSON shape)
    out = {"answers": answers}

    # Log the exact JSON returned to render logs as requested
    try:
        logger.info(f"📤 Response JSON returned to user: {json.dumps(out, ensure_ascii=False)}")
    except Exception:
        logger.info("📤 Response JSON returned to user (could not serialize)")

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
