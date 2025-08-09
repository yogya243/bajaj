import os
import requests
import asyncio
import hashlib
import logging
from fastapi import FastAPI, HTTPException
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
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# OCR fallback
from pdf2image import convert_from_bytes
import pytesseract

# Enable logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI(
    title="AI Insurance Analyzer API",
    description="Batch Q&A over insurance documents using LLMs",
    version="3.0.0"
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
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

def extract_text_from_pdf(file_bytes: bytes) -> str:
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        all_text_pages = []
        has_text = False

        for page_num, page in enumerate(doc, 1):
            # Extract text with better structure preservation
            text_dict = page.get_text("dict")
            page_text = []
            
            # Process blocks to maintain structure
            for block in text_dict.get("blocks", []):
                if "lines" not in block:
                    continue
                    
                block_text = []
                for line in block["lines"]:
                    line_text = ""
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        if text:
                            line_text += text + " "
                    if line_text.strip():
                        block_text.append(line_text.strip())
                
                if block_text:
                    # Join lines in block with space, blocks with newlines
                    page_text.append(" ".join(block_text))
            
            # Join blocks with double newlines to preserve structure
            clean_page_text = "\n\n".join(page_text)
            clean_page_text = normalize_text(clean_page_text)
            
            if clean_page_text:
                has_text = True
            all_text_pages.append(f"[Page {page_num}]\n{clean_page_text}")

        if not has_text:
            logger.warning("No extractable text found, using OCR fallback...")
            images = convert_from_bytes(file_bytes)
            ocr_texts = [
                f"[Page {i}]\n{normalize_text(pytesseract.image_to_string(img))}"
                for i, img in enumerate(images, start=1)
            ]
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
        response = requests.get(url)
        response.raise_for_status()
        file_bytes = response.content
        content_type = response.headers.get("Content-Type", "").lower()

        if "pdf" in content_type or url.lower().endswith(".pdf"):
            return extract_text_from_pdf(file_bytes)
        elif "wordprocessingml" in content_type or url.lower().endswith(".docx"):
            return extract_text_from_docx(file_bytes)
        elif "message" in content_type:
            return extract_text_from_email(file_bytes)
        else:
            return extract_text_from_pdf(file_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch document from URL: {e}")

# Enhanced prompt template for insurance documents
ENHANCED_INSURANCE_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are an expert insurance policy analyst. Your task is to answer questions about insurance policies with precision and accuracy.

CONTEXT FROM POLICY DOCUMENT:
{context}

QUESTION: {question}

INSTRUCTIONS:
1. Answer based ONLY on the information provided in the context above
2. Quote exact text from the policy when possible
3. If the question asks about specific periods, amounts, or conditions, provide the exact details
4. If the information is not found in the context, form the closest statement accordingly.
5. Be concise but complete - include all relevant details from the policy
6. For waiting periods, grace periods, or coverage limits, always specify the exact duration/amount
7. If there are conditions or exceptions, mention them

ANSWER:"""
)

EMBEDDING_MODEL_NAME = "text-embedding-3-large"

def enhance_query(question: str) -> List[str]:
    """Generate multiple query variations to improve retrieval"""
    base_query = question.lower()
    
    # Insurance-specific keyword mappings
    keyword_mappings = {
        'grace period': ['grace period', 'premium payment grace', 'renewal grace'],
        'waiting period': ['waiting period', 'exclusion period', 'moratorium'],
        'pre-existing': ['pre-existing', 'PED', 'pre existing disease'],
        'maternity': ['maternity', 'childbirth', 'pregnancy', 'delivery'],
        'cataract': ['cataract', 'eye surgery', 'ophthalmology'],
        'organ donor': ['organ donor', 'transplant', 'harvesting'],
        'no claim discount': ['no claim discount', 'NCD', 'claim free bonus'],
        'health check': ['health check', 'preventive health', 'medical checkup'],
        'hospital definition': ['hospital', 'medical institution', 'healthcare facility'],
        'ayush': ['AYUSH', 'Ayurveda', 'Yoga', 'Naturopathy', 'Unani', 'Siddha', 'Homeopathy'],
        'room rent': ['room rent', 'accommodation charges', 'daily room charges'],
        'icu charges': ['ICU', 'intensive care', 'critical care']
    }
    
    # Generate enhanced queries
    queries = [question]
    
    for key, variations in keyword_mappings.items():
        if any(term in base_query for term in [key] + variations):
            for variation in variations:
                if variation != key and variation not in base_query:
                    enhanced_q = question.replace(key, variation) if key in question.lower() else question + f" {variation}"
                    queries.append(enhanced_q)
    
    return list(set(queries))

def smart_text_splitter(document_text: str) -> List[Document]:
    """Enhanced text splitting strategy for insurance documents"""
    
    # First, split by major sections (identified by section numbers or headers)
    section_pattern = r'\n(?=\d+\.?\s+[A-Z][^.]*[:\n])'
    major_sections = re.split(section_pattern, document_text)
    
    documents = []
    chunk_id = 0
    
    for section_idx, section in enumerate(major_sections):
        if not section.strip():
            continue
            
        # For each major section, use recursive splitting with insurance-specific separators
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,  # Larger chunks for insurance context
            chunk_overlap=200,  # More overlap to maintain context
            separators=[
                "\n\n",  # Paragraph breaks
                "\n",    # Line breaks
                ". ",    # Sentence ends
                "; ",    # Clause separators
                ", ",    # Sub-clause separators
                " "      # Word breaks
            ],
            length_function=len,
            is_separator_regex=False,
        )
        
        section_chunks = splitter.split_text(section)
        
        for chunk_text in section_chunks:
            if len(chunk_text.strip()) > 50:  # Skip very short chunks
                metadata = {
                    "chunk_id": chunk_id,
                    "section_id": section_idx,
                    "char_count": len(chunk_text)
                }
                documents.append(Document(page_content=chunk_text.strip(), metadata=metadata))
                chunk_id += 1
    
    return documents

def build_faiss_from_chunks(chunks: List[Document]) -> FAISS:
    embeddings = AzureOpenAIEmbeddings(
        deployment=os.getenv("AZURE_EMBEDDING_DEPLOYMENT"),
        model=EMBEDDING_MODEL_NAME,
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
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
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        openai_api_version=os.getenv("AZURE_API_VERSION"),
        openai_api_key=os.getenv("AZURE_API_KEY")
    )
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)

def create_enhanced_retriever(faiss_index: FAISS):
    """Create an enhanced retriever with multiple search strategies"""
    
    # Base retriever with higher k value
    base_retriever = faiss_index.as_retriever(
        search_type="mmr",  # Maximum Marginal Relevance for diversity
        search_kwargs={
            "k": 12,  # Retrieve more documents initially
            "lambda_mult": 0.7,  # Balance between relevance and diversity
        }
    )
    
    return base_retriever

def create_langchain_chain_with_enhanced_prompt(faiss_index: FAISS):
    llm = AzureChatOpenAI(
        temperature=0.1,  # Slightly higher for nuanced responses
        deployment_name=os.getenv("AZURE_MODEL"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        openai_api_version=os.getenv("AZURE_API_VERSION"),
        openai_api_key=os.getenv("AZURE_API_KEY"),
        max_tokens=500  # Ensure adequate response length
    )
    
    retriever = create_enhanced_retriever(faiss_index)
    
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs={"prompt": ENHANCED_INSURANCE_PROMPT}
    )

def get_chain_with_cache(document_text: str):
    doc_hash = get_doc_hash(document_text)
    if doc_hash in INDEX_CACHE:
        return INDEX_CACHE[doc_hash]

    cache_path = os.path.join(CACHE_DIR, doc_hash)
    if os.path.exists(cache_path):
        logger.info("Loading FAISS index from cache...")
        faiss_index = load_faiss_from_disk(cache_path)
        chain = create_langchain_chain_with_enhanced_prompt(faiss_index)
        INDEX_CACHE[doc_hash] = chain
        return chain

    # Use enhanced text splitting
    documents = smart_text_splitter(document_text)
    logger.info(f"Created {len(documents)} document chunks")

    faiss_index = build_faiss_from_chunks(documents)
    save_faiss_to_disk(faiss_index, cache_path)
    chain = create_langchain_chain_with_enhanced_prompt(faiss_index)
    INDEX_CACHE[doc_hash] = chain
    return chain

async def ask_question_enhanced(query: str, chain) -> str:
    """Enhanced question answering with query expansion"""
    try:
        # Generate multiple query variations
        enhanced_queries = enhance_query(query)
        
        # Try primary query first
        result = await asyncio.to_thread(chain.run, query)
        
        # If result seems insufficient, try enhanced queries
        if len(result.strip()) < 50 or "not available" in result.lower():
            for enhanced_query in enhanced_queries[1:3]:  # Try up to 2 variations
                try:
                    enhanced_result = await asyncio.to_thread(chain.run, enhanced_query)
                    if len(enhanced_result.strip()) > len(result.strip()) and "not available" not in enhanced_result.lower():
                        result = enhanced_result
                        break
                except Exception:
                    continue
        
        return result.strip()
    except Exception as e:
        logger.error(f"Error answering question '{query}': {e}")
        return f"Error answering question: {e}"

async def process_in_batches(questions: List[str], chain, batch_size: int = 15):
    """Process questions in smaller batches for better resource management"""
    results = []
    for i in range(0, len(questions), batch_size):
        batch = questions[i:i+batch_size]
        batch_results = await asyncio.gather(*[ask_question_enhanced(q, chain) for q in batch])
        results.extend(batch_results)
        
        # Small delay between batches to prevent rate limiting
        if i + batch_size < len(questions):
            await asyncio.sleep(0.5)
    
    return results

@app.post("/api/v1/hackrx/run", response_model=AnalyzeResponse)
async def analyze_from_url(req: AnalyzeRequest):
    logger.info(f"Processing document: {req.documents}")
    logger.info(f"Number of questions: {len(req.questions)}")
    
    # Extract and process document
    document_text = detect_file_type_and_extract(req.documents)
    logger.info(f"Extracted document length: {len(document_text)} characters")
    
    # Get or create chain
    chain = get_chain_with_cache(document_text)
    
    # Process questions
    answers = await process_in_batches(req.questions, chain)
    
    logger.info("Successfully processed all questions")
    return AnalyzeResponse(answers=answers)

@app.get("/")
def root():
    return {"message": "Enhanced AI Insurance Document Analyzer is running"}

@app.head("/ping")
async def head_ping():
    return Response(status_code=200)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
