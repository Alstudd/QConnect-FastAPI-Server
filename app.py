import os
import uuid
import shutil
import boto3
import nltk
from pinecone import Pinecone
from typing import List, Dict
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from tempfile import NamedTemporaryFile
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.schema.runnable import Runnable
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI as LlamaOpenAI

nltk.download('punkt')
load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET = os.getenv("S3_BUCKET")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("qconnect")
vector_store = PineconeVectorStore(pinecone_index=index)

Settings.llm = LlamaOpenAI(model="gpt-3.5-turbo", temperature=0.7)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
query_engine = VectorStoreIndex.from_vector_store(vector_store=vector_store).as_query_engine()

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set")

s3_client = boto3.client(
    "s3",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

loaded_indexes: Dict[str, FAISS] = {}


def extract_text_chunks(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    return splitter.split_documents(pages)


def build_vector_index(docs, embeddings, index_dir):
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(index_dir)
    return db


def upload_index_to_s3(index_dir, s3_key_prefix):
    for file_name in os.listdir(index_dir):
        full_path = os.path.join(index_dir, file_name)
        s3_key = f"{s3_key_prefix}/{file_name}"
        s3_client.upload_file(full_path, S3_BUCKET, s3_key)


def get_cached_db(index_dir, embeddings):
    if index_dir in loaded_indexes:
        return loaded_indexes[index_dir]

    local_index_dir = f"/tmp/{index_dir}"
    os.makedirs(local_index_dir, exist_ok=True)

    for file_name in ["index.faiss", "index.pkl"]:
        s3_key = f"{index_dir}/{file_name}"
        local_path = os.path.join(local_index_dir, file_name)

        if not os.path.exists(local_path):
            try:
                s3_client.download_file(S3_BUCKET, s3_key, local_path)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error downloading {file_name} from S3: {str(e)}")

    db = FAISS.load_local(local_index_dir, embeddings, allow_dangerous_deserialization=True)
    loaded_indexes[index_dir] = db
    return db

def index_attempts(attempts: List[str]):
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    attempt_embeddings = embeddings.embed_documents(attempts)
    upserts = []
    for i, (attempt, embedding) in enumerate(zip(attempts, attempt_embeddings)):
        attempt_id = str(uuid.uuid4())
        upserts.append({
            "id": attempt_id,
            "values": embedding,
            "metadata": {"text": attempt}
        })
    index.upsert(vectors=upserts)


def query_attempts(query: str, top_k: int = 5):
    response = query_engine.query(query)
    return str(response)


class AttemptsRequest(BaseModel):
    attempts: List[str]

class QueryRequest(BaseModel):
    query: str


@app.post("/upload_attempts/")
async def upload_user_attempts(request: AttemptsRequest):
    try:
        index_attempts(request.attempts)
        return {"status": "success", "message": "User attempts indexed successfully!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query_attempts/")
async def query_user_attempts(request: QueryRequest):
    try:
        response = query_attempts(request.query)
        return JSONResponse(content=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload/", response_model=List[str])
async def upload_documents(
    files: List[UploadFile] = File(...),
    id: str = Form(...)
):
    if any(not file.filename.endswith(".pdf") for file in files):
        raise HTTPException(status_code=400, detail="All files must be PDF format.")

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    all_docs = []
    temp_paths = []

    try:
        for file in files:
            with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                contents = await file.read()
                tmp.write(contents)
                temp_paths.append(tmp.name)

                s3_key = f"{id}/original_pdfs/{file.filename}"
                s3_client.put_object(Bucket=S3_BUCKET, Key=s3_key, Body=contents)

        for path in temp_paths:
            all_docs.extend(extract_text_chunks(path))

        index_dir = f"faiss_index_{id}"
        build_vector_index(all_docs, embeddings, index_dir)
        upload_index_to_s3(index_dir, s3_key_prefix=index_dir)

        result = await query_documents(id=id, query="Give me all important topics that can be used as states in Q-learning for MCQ generation")
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        for path in temp_paths:
            os.remove(path)


@app.post("/query/", response_model=List[str])
async def query_documents(
    id: str = Form(...),
    query: str = Form(...)
):
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    index_dir = f"faiss_index_{id}"
    db = get_cached_db(index_dir, embeddings)

    retriever = db.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(openai_api_key=OPENAI_API_KEY),
        retriever=retriever,
        return_source_documents=False
    )

    response = qa_chain.run(query)
    return [line.strip() for line in response.split("\n") if line.strip()]
