import os
import requests
import asyncio
import hashlib
import logging
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Azure Translator API config
AZURE_TRANSLATOR_KEY = os.getenv("AZURE_TRANSLATOR_KEY")
AZURE_TRANSLATOR_ENDPOINT = os.getenv("AZURE_TRANSLATOR_ENDPOINT")
AZURE_TRANSLATOR_REGION = os.getenv("AZURE_TRANSLATOR_REGION")

def azure_translate(text: str, target_lang: str = "en") -> str:
    """Translate text to target_lang using Azure Translator."""
    if not text.strip():
        return text
    url = f"{AZURE_TRANSLATOR_ENDPOINT}/translate?api-version=3.0&to={target_lang}"
    headers = {
        "Ocp-Apim-Subscription-Key": AZURE_TRANSLATOR_KEY,
        "Ocp-Apim-Subscription-Region": AZURE_TRANSLATOR_REGION,
        "Content-Type": "application/json"
    }
    body = [{"text": text}]
    try:
        resp = requests.post(url, headers=headers, json=body, timeout=10)
        resp.raise_for_status()
        return resp.json()[0]["translations"][0]["text"]
    except Exception as e:
        logger.error(f"Translation error: {e}")
        return text

def azure_detect_language(text: str) -> str:
    """Detect language code for given text."""
    if not text.strip():
        return "en"
    url = f"{AZURE_TRANSLATOR_ENDPOINT}/detect?api-version=3.0"
    headers = {
        "Ocp-Apim-Subscription-Key": AZURE_TRANSLATOR_KEY,
        "Ocp-Apim-Subscription-Region": AZURE_TRANSLATOR_REGION,
        "Content-Type": "application/json"
    }
    body = [{"text": text}]
    try:
        resp = requests.post(url, headers=headers, json=body, timeout=10)
        resp.raise_for_status()
        return resp.json()[0]["language"]
    except Exception as e:
        logger.error(f"Language detection error: {e}")
        return "en"

# ----------------- rest of your unchanged code up to ask_question_enhanced -----------------

async def ask_question_enhanced(query: str, chain) -> dict:
    try:
        # Detect language of the question
        detected_lang = azure_detect_language(query)

        # Translate to English for processing if needed
        query_for_llm = query if detected_lang == "en" else azure_translate(query, "en")

        # Run through LLM
        enhanced_queries = enhance_query(query_for_llm)
        result = await asyncio.to_thread(chain.run, query_for_llm)
        if len(result.strip()) < 50 or "not available" in result.lower():
            for enhanced_query in enhanced_queries[1:3]:
                try:
                    enhanced_result = await asyncio.to_thread(chain.run, enhanced_query)
                    if len(enhanced_result.strip()) > len(result.strip()) and "not available" not in enhanced_result.lower():
                        result = enhanced_result
                        break
                except Exception:
                    continue

        # Translate back to original language if needed
        final_answer = result if detected_lang == "en" else azure_translate(result, detected_lang)

        # Log both Q & A in Render logs
        logger.info(f"Q ({detected_lang}): {query}")
        logger.info(f"A ({detected_lang}): {final_answer}")

        return {
            "question": query,
            "language": detected_lang,
            "answer": final_answer.strip()
        }

    except Exception as e:
        logger.error(f"Error answering question '{query}': {e}")
        return {
            "question": query,
            "language": "en",
            "answer": f"Error answering question: {e}"
        }

async def process_in_batches(questions: List[str], chain, batch_size: int = 15):
    results = []
    for i in range(0, len(questions), batch_size):
        batch = questions[i:i+batch_size]
        batch_results = await asyncio.gather(*[ask_question_enhanced(q, chain) for q in batch])
        results.extend(batch_results)
        if i + batch_size < len(questions):
            await asyncio.sleep(0.5)
    return results

# ----------------- in your route -----------------
@app.post("/api/v1/hackrx/run")
async def analyze_from_url(req: AnalyzeRequest):
    urls = [u.strip() for u in req.documents.split(",") if u.strip()]
    logger.info(f"📄 Received {len(urls)} document URLs:")
    for i, url in enumerate(urls, 1):
        logger.info(f"   Doc {i}: {url}")

    logger.info(f"📝 Received {len(req.questions)} questions:")

    all_texts = []
    for url in urls:
        text = detect_file_type_and_extract(url)
        if text.strip():
            all_texts.append(text)

    if not all_texts:
        raise HTTPException(status_code=400, detail="No extractable text found in provided URLs")

    combined_text = "\n\n".join(all_texts)
    chain = get_chain_with_cache(combined_text)
    answers = await process_in_batches(req.questions, chain)

    # Wrap in your preserved format
    return {
        "domain": "unknown",  # still placeholder if domain detection not implemented
        "answers": answers
    }
