# scripts/06_hybrid_search.py
import os
import math
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# Обов'язкове завантаження змінних оточення на початку файлу
load_dotenv()

INDEX_NAME = "arxiv-papers"
MODEL_NAME = "allenai/specter2_base"
TOP_K = 5  

# Ініціалізація клієнтів Pinecone та моделі ембеддінгів
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(INDEX_NAME)
model = SentenceTransformer(MODEL_NAME)
df = pd.read_parquet("data/arxiv_subset.parquet").reset_index(drop=True)

# ============================================================================
# КРОК 1: ПОБУДОВА ЛОКАЛЬНОГО BM25 ІНДЕКСУ
# ============================================================================
print("Побудова локального текстового індексу BM25...")

corpus = (df["title"] + " " + df["abstract"]).str.lower().str.split().tolist()
bm25 = BM25Okapi(corpus)

# ============================================================================
# КРОК 4: РЕАЛІЗАЦІЯ ФУНКЦІЙ ПОШУКУ
# ============================================================================

def search_bm25(query: str, top_k=5):
    """Повнотекстовий пошук за алгоритмом BM25 Okapi."""
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        results.append({
            "id": f"paper_{idx}", 
            "score": scores[idx], 
            "metadata": {
                "title": df.iloc[idx]["title"], 
                "year": int(df.iloc[idx]["year"]), 
                "category": df.iloc[idx]["category"]
            }
        })
    return results

def search_vector(query: str, top_k=5):
    """Семантичний векторний пошук через індекс Pinecone."""
    q_vec = model.encode(query, normalize_embeddings=True).tolist()
    res = index.query(vector=q_vec, top_k=top_k, include_metadata=True)
    return res['matches']

# ============================================================================
# КРОК 3: РЕАЛІЗАЦІЯ RECIPROCAL RANK FUSION (RRF)
# ============================================================================
def reciprocal_rank_fusion(vec_results, bm25_results, k=60, top_k=5):
    """Об'єднання ранжованих списків за алгоритмом Reciprocal Rank Fusion."""
    rrf_scores = {}
    
    ranks_vec = {item.id: rank for rank, item in enumerate(vec_results, start=1)}
    ranks_bm25 = {item["id"]: rank for rank, item in enumerate(bm25_results, start=1)}
    
    # Об'єднуємо всі унікальні ID документів з обох списків
    all_ids = set(ranks_vec.keys()).union(set(ranks_bm25.keys()))
    
    for doc_id in all_ids:
        score = 0.0
        if doc_id in ranks_vec:
            score += 1.0 / (k + ranks_vec[doc_id])
        if doc_id in ranks_bm25:
            score += 1.0 / (k + ranks_bm25[doc_id])
        rrf_scores[doc_id] = score
        
    # Сортуємо документи за фінальним RRF скором
    sorted_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    # Відновлюємо метадані для фінальної видачі
    final_results = []
    for doc_id, score in sorted_ids:
        # Парсимо індекс з id вигляду paper_XXXX
        idx = int(doc_id.split("_")[1])
        final_results.append({
            "id": doc_id,
            "rrf_score": score,
            "metadata": {
                "title": df.iloc[idx]["title"], 
                "year": int(df.iloc[idx]["year"]), 
                "category": df.iloc[idx]["category"]
            }
        })
    return final_results

def hybrid_search(query: str, top_k=5):
    """Гібридний пошук: збір кандидатів з обох систем та їх RRF ранжування."""
    # Беремо по 30 кандидатів з кожного методу
    vec_res = search_vector(query, top_k=30)
    bm25_res = search_bm25(query, top_k=30)
    return reciprocal_rank_fusion(vec_res, bm25_res, k=60, top_k=top_k)

# ============================================================================
# КРОК 5: ДЕМОНСТРАЦІЯ ТА ТЕСТОВІ ЗАПИТИ
# ============================================================================
queries = [
    "Quantum teleportation using GHZ states",
    "Edward Witten Witten-Reshetikhin-Turaev invariants",
    "reasons behind the accelerating expansion of the universe after the big bang"
]

for q in queries:
    print(f"\n{'='*80}\n ТЕСТОВИЙ ЗАПИТ: '{q}'\n{'='*80}")
    
    # 1. Вивід Топ-5 для класичного BM25
    print("\n ТОП-5 ЗА АЛГОРИТМОМ BM25 (Повнотекстовий пошук):")
    print("-" * 50)
    for i, r in enumerate(search_bm25(q, top_k=TOP_K), 1):
        print(f"  {i}. [{r['metadata']['category']}] {r['metadata']['title'][:65]}... (Скоринг: {r['score']:.4f})")
        
    # 2. Вивід Топ-5 для семантичного векторного пошуку
    print("\n ТОП-5 ЗА ВЕКТОРНИМ ПОШУКОМ (Pinecone Семантика):")
    print("-" * 50)
    for i, r in enumerate(search_vector(q, top_k=TOP_K), 1):
        print(f"  {i}. [{r.metadata['category']}] {r.metadata['title'][:65]}... (Схожість: {r.score:.4f})")
        
    # 3. Вивід Топ-5 для гібридного пошуку через RRF слияние
    print("\n ТОП-5 ГІБРИДНИЙ ПОШУК (BM25 + Вектор через RRF):")
    print("-" * 50)
    for i, r in enumerate(hybrid_search(q, top_k=TOP_K), 1):
        print(f"  {i}. [{r['metadata']['category']}] {r['metadata']['title'][:65]}... (RRF Скор: {r['rrf_score']:.6f})")

print("\n" + "="*80)
print("Гібридний аналіз успішно завершено!")
print("="*80)
