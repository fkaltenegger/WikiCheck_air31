from fastapi import FastAPI
from typing import List
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import os
import pickle
import torch.nn.functional as F
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel




MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
#MODEL_NAME = "bert-base-multilingual-cased"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 256
TOP_K = 5

tokenizer = None
model = None

paragraph_embeddings = None
paragraphs = None
para_titles = None
para_urls = None

### Helper Functions ###

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    pooled = torch.sum(token_embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
    return pooled

@torch.no_grad()
def encode_texts(texts, batch_size=16):
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i + batch_size]
        encoded = tokenizer(batch, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt").to(DEVICE)
        model_output = model(**encoded)
        emb = mean_pooling(model_output, encoded["attention_mask"])
        emb = F.normalize(emb, p=2, dim=1)
        embeddings.append(emb.cpu().numpy())
        
    return np.vstack(embeddings)

def split_into_paragraphs(text, min_length=200, max_length=1200):
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    paragraphs = [p for p in paragraphs if len(p) >= min_length]
    paragraphs = [p[:max_length] for p in paragraphs]
    return paragraphs

def save_encodings(path="wikicheck_index"):
    os.makedirs(path, exist_ok=True)
    np.save(os.path.join(path, "embeddings.npy"), paragraph_embeddings)

    with open(os.path.join(path, "meta.pkl"), "wb") as f:
        pickle.dump(
            {
                "paragraphs": paragraphs,
                "para_titles": para_titles,
                "para_urls": para_urls,
                "model_name": MODEL_NAME,
                "max_length": MAX_LENGTH,
            },
            f,
        )

    print(f"Encodings saved to '{path}/'")

def load_encodings(path="wikicheck_index"):
    global paragraph_embeddings, paragraphs, para_titles, para_urls

    paragraph_embeddings = np.load(os.path.join(path, "embeddings.npy"))

    with open(os.path.join(path, "meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    if meta.get("model_name") != MODEL_NAME:
        print(f"WARNING: saved model_name={meta.get('model_name')} but current MODEL_NAME={MODEL_NAME}")

    paragraphs = meta["paragraphs"]
    para_titles = meta["para_titles"]
    para_urls = meta["para_urls"]

    print(f"Encodings loaded from '{path}/'")


def startup(subset_size=2000):
    global paragraph_embeddings, paragraphs, para_titles, para_urls
    dataset = load_dataset(
        "wikimedia/wikipedia",
        "20231101.en",
        split=f"train[:{subset_size}]"
    )

    articles = [a["text"] for a in dataset]
    titles = [a["title"] for a in dataset]
    urls = [a["url"] for a in dataset]

    paragraphs = []
    para_titles = []
    para_urls = []

    for t, u, art in zip(titles, urls, articles):
        paras = split_into_paragraphs(art)
        for p in paras:
            paragraphs.append(p)
            para_titles.append(t)
            para_urls.append(u)

    safe_model_tag = MODEL_NAME.replace("/", "_")
    INDEX_PATH = f"wikicheck_para_index_{subset_size}_{safe_model_tag}"

    if os.path.exists(os.path.join(INDEX_PATH, "embeddings.npy")):
        load_encodings(INDEX_PATH)
    else:
        print("Encoding Wikipedia paragraphs...")
        paragraph_embeddings = encode_texts(paragraphs)
        save_encodings(INDEX_PATH)

def switch_model(new_model_name, subset_size=2000):
    print(f"Switching model to {new_model_name}...")
    global MODEL_NAME, tokenizer, model
    old_model = model
    old_tokenizer = tokenizer
    
    MODEL_NAME = new_model_name
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()
    
    del old_model
    del old_tokenizer

    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    startup(subset_size=subset_size)



@asynccontextmanager
async def lifespan(app: FastAPI):
    global tokenizer, model
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
        model.eval()

        startup(subset_size=2000)

        print("Startup completed successfully")
        yield

    except Exception as e:
        print(f"Error during startup: {e}")
        raise e

### Query Answering ###

def answer_query(query, top_k=TOP_K):
    query_emb = encode_texts([query])
    scores = cosine_similarity(query_emb, paragraph_embeddings)[0]

    top_indices = scores.argsort()[::-1][:top_k]
    best_idx = int(top_indices[0])

    results = []
    for idx in top_indices:
        results.append({
            "title": para_titles[idx],
            "url": para_urls[idx],
            "score": float(scores[idx]),
            "paragraph": paragraphs[idx],
        })

    return {
        "query": query,
        "answer": paragraphs[best_idx],
        "best_article": {
            "title": para_titles[best_idx],
            "url": para_urls[best_idx],
            "score": float(scores[best_idx]),
        },
        "top_k_results": results,
    }

### FastAPI App ###

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



class RequestBody(BaseModel):
    method: str
    query: str

@app.post("/check")
def health_check(r: RequestBody):
    if r.method == "sbert":
        if MODEL_NAME != "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2":
            switch_model("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    elif r.method == "mbert":
        if MODEL_NAME != "bert-base-multilingual-cased":
            switch_model("bert-base-multilingual-cased")
    elif r.method == "tfidf":
        pass  # keep current model
    
    print(r)
    return answer_query(r.query)

@app.post("/checkmultiple")
def create_items(items: List[str]):
    return {
        i: answer_query(i) for i in items
    }