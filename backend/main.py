from fastapi import FastAPI
from typing import List
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, MarianMTModel, MarianTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import CrossEncoder
import json
import numpy as np
import torch
import os
import pickle
import torch.nn.functional as F

models = {"sbert": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "mbert": "bert-base-multilingual-cased", "tf-idf": "tf-idf"}
MODEL_NAME = models["sbert"]
MODEL_RERANKER = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
MODEL_EVAL = "facebook/bart-large-mnli"
MODEL_TRANSLATE_DE = "Helsinki-NLP/opus-mt-en-de"
MODEL_TRANSLATE_ES = "Helsinki-NLP/opus-mt-en-es"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 256
TOP_K = 5

tokenizer = None
model = None
eval_model = None
eval_tokenizer = None
reranker = None

tfidf_vectorizer = None
tfidf_matrix = None

paragraph_embeddings = None
paragraphs = None
para_titles = None
para_urls = None

translation_models = {
    "de": {
        model: None,
        tokenizer: None
    },
    "es": {
        model: None,
        tokenizer: None
    }
}

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

def save_tfidf(path):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "tfidf.pkl"), "wb") as f:
        pickle.dump(
            {
                "vectorizer": tfidf_vectorizer,
                "matrix": tfidf_matrix,
            },
            f,
        )

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

def load_tfidf(path):
    global tfidf_vectorizer, tfidf_matrix

    with open(os.path.join(path, "tfidf.pkl"), "rb") as f:
        data = pickle.load(f)
    
    tfidf_vectorizer = data["vectorizer"]
    tfidf_matrix = data["matrix"]

def build_tfidf_index():
    global tfidf_vectorizer, tfidf_matrix

    tfidf_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_df=0.95,
        min_df=2,
        stop_words="english"
    )
    tfidf_matrix = tfidf_vectorizer.fit_transform(paragraphs)

def translate(texts, language):
    tokenizer = translation_models[language]["tokenizer"]
    model = translation_models[language]["model"]

    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    translated = model.generate(**inputs)

    return tokenizer.batch_decode(translated, skip_special_tokens=True)

def startup(subset_size=2000):
    global paragraph_embeddings, paragraphs, para_titles, para_urls, tokenizer, model
    
    # TODO: TEST if streaming works correctly
    
    dataset = load_dataset(
        "wikimedia/wikipedia",
        "20231101.en",
        split=f"train",
        streaming=True
    ).take(subset_size)
    
    # dataset = load_dataset(
    #     "wikimedia/wikipedia",
    #     "20231101.en",
    #     split=f"train[:{subset_size}]"
    # )

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

    if MODEL_NAME == "tf-idf":
        if os.path.exists(os.path.join(INDEX_PATH, "tfidf.pkl")):
            load_tfidf(INDEX_PATH)
        else:
            print("Building TF-IDF index...")
            build_tfidf_index()
            save_tfidf(INDEX_PATH)
    else:
        old_model = model
        old_tokenizer = tokenizer
        
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
        model.eval()

        del old_model
        del old_tokenizer

        if os.path.exists(os.path.join(INDEX_PATH, "embeddings.npy")):
            load_encodings(INDEX_PATH)
        else:
            print("Encoding Wikipedia paragraphs...")
            paragraph_embeddings = encode_texts(paragraphs)
            save_encodings(INDEX_PATH)

def switch_model(new_model_name, subset_size=2000):
    print(f"Switching model to {new_model_name}...")
    global MODEL_NAME
    
    MODEL_NAME = new_model_name

    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    startup(subset_size=subset_size)



@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        global reranker, eval_model, eval_tokenizer, translation_models
        reranker = CrossEncoder(MODEL_RERANKER, device=DEVICE)
        eval_tokenizer = AutoTokenizer.from_pretrained(MODEL_EVAL)
        eval_model = AutoModelForSequenceClassification.from_pretrained(MODEL_EVAL).to(DEVICE)
        eval_model.eval()
        translation_models["de"]["tokenizer"] = MarianTokenizer.from_pretrained(MODEL_TRANSLATE_DE)
        translation_models["de"]["model"] = MarianMTModel.from_pretrained(MODEL_TRANSLATE_DE).to(DEVICE)
        translation_models["es"]["tokenizer"] = MarianTokenizer.from_pretrained(MODEL_TRANSLATE_ES)
        translation_models["es"]["model"] = MarianMTModel.from_pretrained(MODEL_TRANSLATE_ES).to(DEVICE)
        startup(subset_size=2000)
        print("Startup completed successfully")
        yield

    except Exception as e:
        print(f"Error during startup: {e}")
        raise e

### Query Answering ###

def answer_query(query, top_k, rerank, lang="en"):
    tfidf = MODEL_NAME == "tf-idf"
    query_emb = tfidf_vectorizer.transform([query]) if tfidf else encode_texts([query])
    scores = cosine_similarity(query_emb, tfidf_matrix if tfidf else paragraph_embeddings)[0]

    top_indices = scores.argsort()[::-1][:top_k + 100]
    
    if rerank:
        pairs = [(query, paragraphs[idx]) for idx in top_indices]
        ce_scores = reranker.predict(pairs)

        for idx, new_score in zip(top_indices, ce_scores):
            scores[idx] = new_score

        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
    
    translations = []

    results = []
    for idx in top_indices[:top_k]:
        results.append({
            "title": para_titles[idx],
            "url": para_urls[idx],
            "score": float(scores[idx]),
            "paragraph": paragraphs[idx],
            "eval": eval(query, paragraphs[idx])
        })
        if lang != "en":
            translations.append(paragraphs[idx])
            translations.append(para_titles[idx])

    if len(translations) > 0:
        translations = translate(translations, lang)
        for i, result in enumerate(results):
            result["paragraph"] = translations[i * 2]
            result["title"] = translations[i * 2 + 1]

    return results

### Evalutaion ###

def eval(query, paragraph):
    inputs = eval_tokenizer(
        paragraph,
        query,
        return_tensors="pt",
        truncation=True
    ).to(DEVICE)
    
    with torch.no_grad():
        logits = eval_model(**inputs).logits
        
    prediction = logits.argmax(dim=1).item()

    label_map = {
        0: "CONTRADICTS",
        1: "NOT MENTIONED",
        2: "SUPPORTS"
    }

    return label_map[prediction]
    
    
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
    ce: bool = True
    query: str
    response_language: str

@app.post("/check")
def check(r: RequestBody):
    key = r.method.lower()
    if key in models:
        requestedModel = models[key]
        if MODEL_NAME != requestedModel:
            switch_model(requestedModel)

    print(r)
    return answer_query(r.query, TOP_K, r.ce, r.response_language)

@app.post("/checkmultiple")
def checkmultiple(items: List[str]):
    return {
        i: answer_query(i, TOP_K, False) for i in items
    }

@app.post("/evaluation")
def evaluation():
    with open('eval.json', 'r') as f:
        eval_data = f.read()

    eval_data = json.loads(eval_data)

    languages = ["en", "de", "es"]
    
    eval_results = {model: { f"ce_{ce}": {} for ce in [True, False] } for model in models.keys()}

    model_list = list(models.keys())
    n = len(eval_data)

    for model in model_list:
        switch_model(models[model])
        for ce in [True, False]:
            for lang in languages:
                mrr = 0
                hit_rate = 0
                accuracy = 0
                query_answers = []
                for data in eval_data:
                    results = answer_query(data["claims"][lang], TOP_K, ce)
                    query_answers.append({
                        "query": data["claims"][lang],
                        "results": results
                    })
                    if results[0]["url"] == data["url"]:
                        accuracy += 1
                    if results[0]["eval"] != "NOT MENTIONED":
                        hit_rate += 1
                    for i, result in enumerate(results, 1):
                        if result["url"] == data["url"]:
                            mrr += 1 / i
                            break
                mrr /= n
                hit_rate /= n
                accuracy /= n
                eval_results[model][f"ce_{ce}"][lang] = {
                    "mrr": mrr,
                    "hit_rate": hit_rate,
                    "accuracy": accuracy,
                    "accurate_hit_rate": hit_rate / accuracy if accuracy > 0 else 0,
                    "results": query_answers
                }
                
    # with open('eval_results.json', 'w') as f:
    #     json.dump(eval_results, f, indent=4)

    return eval_results
