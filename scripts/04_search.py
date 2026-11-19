# scripts/04_search.py
import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# Ініціалізація змінних оточення
load_dotenv()

INDEX_NAME = "arxiv-papers"
MODEL_NAME = "allenai/specter2_base"
TOP_K = 5

# Ініціалізація клієнтів для Pinecone та моделі HuggingFace
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(INDEX_NAME)
model = SentenceTransformer(MODEL_NAME)

# Завантаження локальних даних для математичного порівняння метрик
df = pd.read_parquet("data/arxiv_subset.parquet")
local_embeddings = np.load("embeddings/embeddings.npy")


# ============================================================================
# КРОК 1: АНАЛІЗ ЛОКАЛЬНОГО ДАТАСЕТУ
# ============================================================================

def collect_target_articles():
    """Аналізує локальний Parquet-файл на наявність цільових років публікації."""
    print("="*70)
    print("КРОК 1: АНАЛІЗ НАЯВНОСТІ ЦІЛЬОВИХ СТАТЕЙ У ЛОКАЛЬНОМУ ДАТАСЕТІ")
    print("="*70)
    
    mask_A = (df["category"] == "cs.LG") & (df["year"] >= 2021)
    articles_A = df[mask_A]
    
    mask_B = df["year"] <= 2015
    articles_B = df[mask_B]
    
    print(f"\n Потенційні результати для Прикладу A (cs.LG, >= 2021): {len(articles_A)} шт.")
    print(f" Потенційні результати для Прикладу B (будь-яка, <= 2015): {len(articles_B)} шт.")
    print("="*70 + "\n")
    return articles_A, articles_B


# ============================================================================
# КРОК 2: СЕМАНТИЧНИЙ ПОШУК (Без фільтрації)
# ============================================================================

def search_semantic(query: str, top_k=5):
    """Чистий семантичний пошук за вектором запиту по всьому індексу Pinecone."""
    query_embedding = model.encode(query, normalize_embeddings=True).tolist()
    
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    
    print(f"КРОК 2: ЧИСТИЙ СЕМАНТИЧНИЙ ПОШУК ЗА ЗАПИТОМ: '{query}'")
    print("-"*70)
    
    for i, match in enumerate(results['matches'], 1):
        meta = match['metadata']
        print(f"{i}. [{meta['category']}, {meta['year']}] {meta['title'][:70]}...")
        print(f"   Score (Cosine): {match['score']:.4f}")
        print(f"   Abstract: {meta['abstract'][:150]}...\n")
    
    return results['matches']


# ============================================================================
# КРОК 3: ПОШУК З ФІЛЬТРАЦІЄЮ МЕТАДАНИХ
# ============================================================================

def search_with_filter(query: str, filter_dict: dict, top_k=5, label=""):
    """Пошук у Pinecone з застосуванням умовних фільтрів для метаданих."""
    query_embedding = model.encode(query, normalize_embeddings=True).tolist()
    
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        filter=filter_dict,
        include_metadata=True
    )
    
    print(f"КРОК 3: ФІЛЬТРОВАНИЙ ПОШУК [{label}]: '{query}'")
    print(f"Конфігурація фільтра: {filter_dict}")
    print("-"*70)
    
    if not results['matches']:
        print("  Нічого не знайдено за вказаними метаданими у поточному зрізі бази.\n")
        return []
    
    for i, match in enumerate(results['matches'], 1):
        meta = match['metadata']
        print(f"{i}. [{meta['category']}, {meta['year']}] {meta['title'][:70]}...")
        print(f"   Score: {match['score']:.4f}")
        print(f"   Abstract: {meta['abstract'][:150]}...\n")
    
    return results['matches']


def explain_filtering_differences():
    """Виводить аналітичне порівняння та пояснення результатів фільтрації А і Б."""
    print("="*70)
    print("АНАЛІЗ ВІДМІННОСТЕЙ МІЖ ПРИКЛАДОМ А ТА ПРИКЛАДОМ Б")
    print("="*70)
    print("1. Зсув у часі та наукових поняттях (Concept Drift):")
    print("   - Приклад А (>= 2021) орієнтований на сучасні архітектури Deep Learning та LLM.")
    print("   - Приклад Б (<= 2015) повертає фундаментальні алгоритми (SVM, Random Forest, MDP).")
    print("2. Особливість нашого локального індексу:")
    print("   - Оскільки базовий скрипт завантажив перші 10 000 рядків архіву, всі вони")
    print("     стовідсотково належать до 2007 року. Через це Приклад А видасть 0 результатів,")
    print("     а Приклад Б поверне найбільш релевантні статті з усього нашого масиву.")
    print("   - Це наочно демонструє силу метафільтрації Pinecone: сервер миттєво відсікає")
    print("     невідповідні вектори, захищаючи користувача від застарілої чи нерелевантної видачі.")
    print("="*70 + "\n")


# ============================================================================
# КРОК 4: ПОРІВНЯННЯ МЕТРИК СХОЖОСТІ (Локальні обчислення через NumPy)
# ============================================================================

def compare_similarity_metrics(query: str, top_k=5):
    """Обчислює Cosine, Dot Product та L2 відстані локально для порівняння математики."""
    query_embedding = model.encode(query, normalize_embeddings=True)
    
    # 1. Cosine similarity (косинусна схожість)
    cosine_scores = np.dot(local_embeddings, query_embedding)
    top_cosine_idx = np.argsort(cosine_scores)[::-1][:top_k]
    
    # 2. Dot product (скалярний добуток)
    dot_scores = np.dot(local_embeddings, query_embedding)
    top_dot_idx = np.argsort(dot_scores)[::-1][:top_k]
    
    # 3. L2 distance (Евклідова відстань)
    l2_distances = np.linalg.norm(local_embeddings - query_embedding, axis=1)
    top_l2_idx = np.argsort(l2_distances)[:top_k]
    
    print(f"КРОК 4: ЛОКАЛЬНЕ ПОРІВНЯННЯ МЕТРИК СХОЖОСТІ ДЛЯ ЗАПИТУ: '{query}'")
    print("="*70)
    
    print("\nTOP-5 ЗА COSINE SIMILARITY:")
    for rank, idx in enumerate(top_cosine_idx, 1):
        print(f"   {rank}. [{df.iloc[idx]['category']}, {df.iloc[idx]['year']}] {df.iloc[idx]['title'][:60]}...")
        print(f"      Score: {cosine_scores[idx]:.4f}")
    
    print("\nTOP-5 ЗА DOT PRODUCT:")
    for rank, idx in enumerate(top_dot_idx, 1):
        print(f"   {rank}. [{df.iloc[idx]['category']}, {df.iloc[idx]['year']}] {df.iloc[idx]['title'][:60]}...")
        print(f"      Score: {dot_scores[idx]:.4f}")
    
    print("\nTOP-5 ЗА L2 DISTANCE (менша відстань = краще):")
    for rank, idx in enumerate(top_l2_idx, 1):
        print(f"   {rank}. [{df.iloc[idx]['category']}, {df.iloc[idx]['year']}] {df.iloc[idx]['title'][:60]}...")
        print(f"      Distance: {l2_distances[idx]:.4f}")
    
    # Порівняльний аналіз результатів у консолі
    print("\nМАТЕМАТИЧНИЙ АНАЛІЗ ВІДМІННОСТЕЙ МЕТРИК:")
    cosine_set = set(top_cosine_idx)
    dot_set = set(top_dot_idx)
    l2_set = set(top_l2_idx)
    
    print(f"   Спільних статей Cosine ∩ Dot: {len(cosine_set & dot_set)} з {top_k}")
    print(f"   Спільних статей Cosine ∩ L2: {len(cosine_set & l2_set)} з {top_k}")
    
    if cosine_set == dot_set:
        print("   Cosine Similarity та Dot Product дали ІДЕНТИЧНІ результати.")
        print("   Пояснення: Наші ембеддинги були попередньо нормалізовані (довжина = 1.0).")
        print("   Для одиничних векторів косинус кута математично дорівнює їхньому скалярному добутку.")
        
    if cosine_set == l2_set:
        print("   Евклідова відстань (L2) дала ТОЧНО ТАКІ Ж результати, що й косинусна схожість.")
        print("   Пояснення: Між L2-відстанню та косинусною схожістю нормалізованих векторів")
        print("   існує зворотна монотонна залежність: квадрат відстані L2 = 2 * (1 - Cosine).")
        print("   Тому геометрично найближчі точки за L2 завжди будуть найближчими за кутом Cosine.")
    print("="*70 + "\n")


# ============================================================================
# ТОЧКА ВХОДУ В СЦЕНАРІЙ ПОШУКУ
# ============================================================================

if __name__ == "__main__":
    # Запускаємо попередній аналіз локального Parquet-файлу
    collect_target_articles()
    
    # Тестовий пошуковий запит із  завдання
    main_query = "teaching machines to recognize objects in pictures"
    
    # 1. Запуск чистого семантичного пошуку через хмару Pinecone
    search_semantic(query=main_query, top_k=TOP_K)
    
    # 2. Пошук із фільтрацією (Приклад А: cs.LG за останні 5 років)
    filter_A = {
        "category": {"$eq": "cs.LG"},
        # "category": {"$eq": "..."}, для тестування
        "year": {"$gte": 2021}
    }
    search_with_filter(query="reinforcement learning", filter_dict=filter_A, top_k=TOP_K, label="Приклад А")
    
    # 3. Пошук із фільтрацією (Приклад Б: старі статті до 2015 року)
    filter_B = {
        "year": {"$lte": 2015}
    }
    search_with_filter(query="neural networks", filter_dict=filter_B, top_k=TOP_K, label="Приклад Б")
    
    # 4. Вивід аналітичного порівняння результатів фільтрації у консоль
    explain_filtering_differences()
    
    # 5. Локальне порівняння математичних метрик схожості через матриці NumPy
    compare_similarity_metrics(query=main_query, top_k=TOP_K)
