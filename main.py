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
    return hashlib.md5(text.encode()).hexdigest()

# ----------- Text Normalization -----------
def normalize_text(text: str) -> str:
    text = re.sub(r'-\n', '', text)
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

# ----------- File Extraction -----------
def extract_text_from_pdf(file_bytes: bytes) -> str:
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        all_text = []
        has_text = False

        for page_num, page in enumerate(doc, 1):
            data = page.get_text("dict")
            page_content = []
            table_mode = False

            for block in data["blocks"]:
                if "lines" not in block:
                    continue
                column_positions = {round(line["spans"][0]["bbox"][0], -1)
                                    for line in block["lines"] if line["spans"]}
                if len(column_positions) > 1 and not table_mode:
                    page_content.append("[TABLE START]")
                    table_mode = True

                block_text = " ".join(span["text"].strip()
                                      for line in block["lines"]
                                      for span in line["spans"] if span["text"].strip())
                if block_text:
                    page_content.append(block_text)

                if table_mode and len(column_positions) <= 1:
                    page_content.append("[TABLE END]")
                    table_mode = False

            clean_page_text = normalize_text("\n".join(page_content))
            if clean_page_text.strip():
                has_text = True
            all_text.append(f"[Page {page_num}]\n{clean_page_text}")

        if not has_text:
            images = convert_from_bytes(file_bytes)
            ocr_texts = []
            for i, img in enumerate(images, start=1):
                ocr_text = pytesseract.image_to_string(img)
                ocr_texts.append(f"[Page {i}]\n{normalize_text(ocr_text)}")
            return "\n\n".join(ocr_texts)

        return "\n\n".join(all_text)
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
        return normalize_text(msg.get_body(preferencelist=('plain', 'html')).get_content())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process Email: {e}")

def detect_file_type_and_extract(url: str) -> str:
    try:
        response = requests.get(url, stream=True)
        file_bytes = response.content
        content_type = response.headers.get("Content-Type", "")

        if "pdf" in content_type:
            return extract_text_from_pdf(file_bytes)
        elif "wordprocessingml" in content_type:
            return extract_text_from_docx(file_bytes)
        elif "message" in content_type or content_type == "application/octet-stream":
            return extract_text_from_email(file_bytes)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {content_type}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch document: {e}")

# ----------- Prompt -----------
INSURANCE_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a smart insurance policy assistant.
Based only on the following excerpts from the policy:
{context}

Question: "{question}"

Provide a concise, direct answer strictly based on the clauses above.
Avoid assumptions. Mention specific clauses if helpful.
"""
)

# ----------- FAISS Utilities -----------
def build_faiss_from_chunks(chunks: List[Document]) -> FAISS:
    embeddings = AzureOpenAIEmbeddings(
        deployment=os.getenv("AZURE_EMBEDDING_DEPLOYMENT"),
        model="text-embedding-3-large",
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
        model="text-embedding-3-large",
        azure_endpoint=os.getenv("AZURE_API_BASE"),
        openai_api_version=os.getenv("AZURE_API_VERSION"),
        openai_api_key=os.getenv("AZURE_API_KEY")
    )
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)

# ----------- Chain Creation -----------
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

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " "]
    )
    chunks = [Document(page_content=text, metadata={"chunk_id": i})
              for i, text in enumerate(splitter.split_text(document_text), start=1)]

    faiss_index = build_faiss_from_chunks(chunks)
    save_faiss_to_disk(faiss_index, cache_path)
    chain = create_langchain_chain_with_prompt(faiss_index)
    INDEX_CACHE[doc_hash] = chain
    return chain

# ----------- Async Parallel Processing -----------
async def ask_question(q: str, chain, semaphore: asyncio.Semaphore) -> str:
    async with semaphore:
        try:
            return await asyncio.to_thread(chain.run, q)
        except Exception as e:
            return f"Error: {e}"

async def process_questions_parallel(questions: List[str], chain, max_concurrency: int = 5):
    semaphore = asyncio.Semaphore(max_concurrency)
    tasks = [ask_question(q, chain, semaphore) for q in questions]
    return await asyncio.gather(*tasks)

# ----------- API Endpoints -----------
@app.post("/api/v1/hackrx/run", response_model=AnalyzeResponse)
async def analyze_from_url(req: AnalyzeRequest):
    document_text = detect_file_type_and_extract(req.documents)
    chain = get_chain_with_cache(document_text)
    answers = await process_questions_parallel(req.questions, chain, max_concurrency=5)
    return AnalyzeResponse(answers=[a.strip() for a in answers])

@app.get("/")
def root():
    return {"message": "AI Insurance Document Analyzer is running"}

@app.head("/ping")
async def head_ping():
    return Response(status_code=200)
