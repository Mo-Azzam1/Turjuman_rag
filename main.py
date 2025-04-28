# main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import os
from dotenv import load_dotenv
import nltk
from nltk.tokenize.punkt import PunktSentenceTokenizer # استخدام استدعاء أكثر تحديدًا
from langchain.text_splitter import NLTKTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.chains.Youtubeing import load_qa_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import LLMChain
from langchain_core.runnables import RunnableMap, RunnableLambda
from langchain_core.documents import Document
import uvicorn # إضافة uvicorn
import time # للملاحظة حول time.sleep

# --- NLTK Download (Run once on startup) ---
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    print("NLTK 'punkt' tokenizer not found. Downloading...")
    nltk.download('punkt')
except LookupError:
     print("NLTK 'punkt' tokenizer not found. Downloading...")
     nltk.download('punkt')


# --- Environment Setup & API Key Loading ---
load_dotenv()
# استخدم نفس اسم متغير البيئة الذي ستضعه في Railway
google_api_key = os.getenv("GOOGLE_API_KEY")

if not google_api_key:
    print("Error: GOOGLE_API_KEY environment variable is not set.")
    raise ValueError("GOOGLE_API_KEY environment variable is missing.")

# --- Global Resource Initialization ---
# ملاحظة: في تطبيق حقيقي يتعامل مع مستندات متغيرة،
# يجب إعادة بناء vector_db لكل مستند جديد أو استخدام حلول تخزين متقدمة.
# هنا نستخدم محتوى ثابت للمستند كمثال بناءً على السكريبت الأصلي.

# قم بتغيير هذا النص للمستند الفعلي الذي تريد معالجته عند النشر
document_content = """
This is the first paragraph of the example document. It talks about the importance of technology in modern life. Technology has revolutionized how we communicate, work, and learn.

The second paragraph discusses artificial intelligence. AI is transforming industries by enabling automation and data analysis on an unprecedented scale. Machine learning, a subset of AI, powers many applications we use daily.

The third paragraph focuses on renewable energy. The shift towards renewable sources like solar and wind power is crucial for combating climate change. Investing in green technologies is essential for a sustainable future.

Finally, the fourth paragraph highlights global collaboration. Addressing complex global challenges requires international cooperation and shared knowledge. Working together across borders is key to finding effective solutions.
"""


# Initializing Text Splitter
text_splitter = NLTKTextSplitter(
    chunk_size=600, chunk_overlap=30
)

# Splitting the document into chunks
token_chunks = text_splitter.create_documents([document_content])
print(f"Document split into {len(token_chunks)} chunks.")


# Initializing Embedding Model
try:
    embedding_llm = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        task_type="retrieval_document",
        google_api_key=google_api_key # Use loaded key
    )
    print("Embedding model initialized.")
except Exception as e:
    print(f"Error initializing embedding model: {e}")
    raise # Fail fast if model initialization fails


# Building the FAISS Vector Store (This will run once on startup)
try:
    texts = [doc.page_content for doc in token_chunks]
    # Note: FAISS.from_texts handles embedding internally if an embedding model is passed
    vector_db = FAISS.from_texts(texts, embedding_llm)
    print("FAISS vector store built.")
except Exception as e:
    print(f"Error building vector store: {e}")
    raise # Fail fast if vector store building fails


# Initializing Chat LLM
try:
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-1.5-flash",
        temperature=0.5,
        google_api_key=google_api_key # Use loaded key
    )
    print("Chat model initialized.")
except Exception as e:
    print(f"Error initializing chat model: {e}")
    raise # Fail fast if chat model initialization fails


# --- Langchain Chain Definitions ---

# Chain for Question Answering (RAG)
qa_qna_template = "\n".join([
    "Answer the next question using ONLY the provided context.",
    "If the answer is not contained in the context, say 'NO ANSWER IS AVAILABLE'.",
    "### Context:",
    "{context}",
    "",
    "### Question:",
    "{question}",
    "",
    "### Answer:",
])

qa_prompt = PromptTemplate(
    template=qa_qna_template,
    input_variables=['context', 'question'],
)

qa_combine_docs_chain = create_stuff_documents_chain(llm, qa_prompt)

# This RAG chain takes a question, retrieves docs, and answers
qa_rag_chain = RunnableMap({
    "context": lambda x: vector_db.similarity_search(x["question"], k=3), # Retrieve more docs for QA
    "question": lambda x: x["question"],
}) | qa_combine_docs_chain


# Chain for generating context *description* for translation (based on user's script logic)
# NOTE: The logic in the original script for translation context seems unusual.
# It processes the current paragraph using the LLM and a prompt "ما هو السياق؟"
# We replicate this, but a typical approach might be to retrieve related chunks
# from the vector store as context for translation.
context_prompt_template = PromptTemplate.from_template(
    """
Use the following text:

Text:
{context}

Based on the text, what is the main context or topic being discussed? Provide a brief description.
"""
)
# This chain takes a paragraph and uses the LLM to describe its context/topic
context_retriever_chain = LLMChain(llm=llm, prompt=context_prompt_template)


# Chain for Translation
translation_prompt_template = PromptTemplate.from_template(
    """
You are an expert Arabic translator. Use the following context description to guide your translation:

Context Description:
{context_description}

Translate the following English text to Modern Standard Arabic (MSA):

{english_paragraph}

Provide only the translated Arabic text. Do not add any explanations or extra text.
"""
)

translation_chain = LLMChain(llm=llm, prompt=translation_prompt_template)


# --- FastAPI App Setup ---

app = FastAPI(
    title="Document Processing API",
    description="API for Q&A and Translation of a pre-loaded document using Gemini and Langchain.",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all for now (can be restricted)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---

class AnswerRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str

class TranslateResponse(BaseModel):
    translated_chunks: List[str]


# --- FastAPI Endpoints ---

@app.get("/")
async def read_root():
    """Root endpoint returning basic API info."""
    return {"message": "Document Processing API is running.", "version": "1.0.0", "chunks_loaded": len(token_chunks)}

@app.post("/answer", response_model=AnswerResponse)
async def get_answer(request: AnswerRequest):
    """
    Answers a question based on the pre-loaded document using RAG.

    Expects a JSON body with 'question'.
    Returns a JSON object with the answer.
    """
    try:
        answer = qa_rag_chain.invoke({"question": request.question})
        return AnswerResponse(answer=answer)
    except Exception as e:
        print(f"Error during Q&A: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during Q&A: {e}")


@app.post("/translate_document", response_model=TranslateResponse)
async def translate_full_document():
    """
    Translates the entire pre-loaded document chunk by chunk.

    WARNING: This endpoint can be very slow for large documents and may
    hit request timeouts due to sequential processing and potential sleeps.
    A better approach for production might be chunk-by-chunk translation endpoints
    or background processing.
    """
    all_translations = []

    # Iterate through globally available chunks
    for i, chunk in enumerate(token_chunks):
        paragraph = chunk.page_content
        try:
            # Get context description for the current paragraph (replicating user's script logic)
            # Note: The chain originally used for this was somewhat unconventional for context retrieval
            context_output = await context_retriever_chain.ainvoke({"context": paragraph}) # Using async invoke

            # Perform translation for the current paragraph
            translation = await translation_chain.ainvoke({ # Using async invoke
                "context_description": context_output, # Pass the generated context description
                "english_paragraph": paragraph
            })
            all_translations.append(translation)

            # WARNING: This sleep is from the original script and blocks the server.
            # It's included here to replicate the script's behavior but is bad for API responsiveness.
            print(f"Translated chunk {i+1}/{len(token_chunks)}. Sleeping...")
            time.sleep(7) # Blocking sleep!

        except Exception as e:
            print(f"\nError translating chunk {i+1}: {e}")
            # Decide how to handle errors for individual chunks - here we just print and continue
            all_translations.append(f"ERROR: Translation failed for chunk {i+1}. Details: {e}") # Add error marker

    return TranslateResponse(translated_chunks=all_translations)


# --- Server Runner (for local testing and Railway) ---
# هذا الجزء يجعل السكريبت قابلاً للتشغيل مباشرة بـ `python main.py`
# ويتأكد من أن uvicorn يستمع على العنوان والمنفذ الصحيحين لـ Railway.

if __name__ == "__main__":
    # تأكد من استيراد uvicorn في بداية الملف لو هتستخدم الجزء ده
    # if uvicorn غير مستورد في البداية، استورد هنا:
    # import uvicorn

    # حاول تقرأ البورت من متغير البيئة PORT، لو مش موجود استخدم 8080
    port = int(os.environ.get("PORT", 8080))

    # شغل السيرفر باستخدام uvicorn
    # host="0.0.0.0" مهم جداً عشان Railway يقدر يوصل للتطبيق
    print(f"Starting server on {os.environ.get('HOST', '0.0.0.0')}:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)