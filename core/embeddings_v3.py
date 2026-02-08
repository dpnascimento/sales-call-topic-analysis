"""
Utilitários para manipulação de embeddings
"""
import numpy as np
from typing import List, Optional, Dict


def to_pgvector(vec: np.ndarray) -> str:
    """Converte numpy array para literal pgvector"""
    return "[" + ",".join(f"{float(x):.8f}" for x in vec.tolist()) + "]"


def from_pgvector(s: str) -> np.ndarray:
    """Parse string pgvector para numpy array"""
    if not s:
        return np.array([], dtype=np.float32)
    s = s.strip()
    if not (s.startswith("[") and s.endswith("]")):
        return np.array([], dtype=np.float32)
    parts = s[1:-1].strip()
    if not parts:
        return np.array([], dtype=np.float32)
    vals = [float(x) for x in parts.split(",")]
    return np.array(vals, dtype=np.float32)


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    """
    Normalização L2 de vetor(es)
    
    Args:
        vec: Array numpy (1D ou 2D)
        
    Returns:
        Array normalizado
    """
    vec = np.asarray(vec, dtype=np.float32)
    if vec.ndim == 1:
        norm = float(np.linalg.norm(vec))
        return vec / (norm if norm > 0 else 1.0)
    # 2D case
    norms = np.linalg.norm(vec, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vec / norms


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Calcula similaridade cosseno entre dois vetores
    
    Args:
        vec_a: Primeiro vetor
        vec_b: Segundo vetor
        
    Returns:
        Similaridade cosseno (-1 a 1)
    """
    vec_a = np.asarray(vec_a, dtype=np.float32)
    vec_b = np.asarray(vec_b, dtype=np.float32)
    
    if vec_a.size == 0 or vec_b.size == 0:
        return 0.0
    
    # Normaliza vetores
    vec_a = l2_normalize(vec_a)
    vec_b = l2_normalize(vec_b)
    
    # Produto escalar
    return float(np.dot(vec_a, vec_b))


def cosine_distance(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Calcula distância cosseno entre dois vetores (1 - similaridade)
    
    Args:
        vec_a: Primeiro vetor
        vec_b: Segundo vetor
        
    Returns:
        Distância cosseno (0 a 2)
    """
    return 1.0 - cosine_similarity(vec_a, vec_b)


def centroid(vectors: List[np.ndarray]) -> Optional[np.ndarray]:
    """
    Calcula centróide (média) de uma lista de vetores
    
    Args:
        vectors: Lista de arrays numpy
        
    Returns:
        Centróide normalizado ou None se lista vazia
    """
    if not vectors:
        return None
    
    # Empilha vetores
    matrix = np.vstack(vectors).astype(np.float32)
    
    # Calcula média
    mean_vec = np.mean(matrix, axis=0)
    
    # Normaliza
    return l2_normalize(mean_vec)


def weighted_centroid(vectors: List[np.ndarray], weights: List[float]) -> Optional[np.ndarray]:
    """
    Calcula centróide ponderado de vetores
    
    Args:
        vectors: Lista de arrays numpy
        weights: Lista de pesos (mesmo tamanho que vectors)
        
    Returns:
        Centróide ponderado normalizado ou None
    """
    if not vectors or not weights or len(vectors) != len(weights):
        return None
    
    w = np.array(weights, dtype=np.float32).reshape(-1, 1)
    M = np.vstack(vectors).astype(np.float32)
    
    weight_sum = float(w.sum())
    if weight_sum <= 0:
        return None
    
    # Média ponderada
    weighted_mean = (M * w).sum(axis=0) / weight_sum
    
    # Normaliza
    return l2_normalize(weighted_mean)


def pairwise_cosine_similarity(vectors: List[np.ndarray]) -> np.ndarray:
    """
    Calcula matriz de similaridade cosseno par-a-par
    
    Args:
        vectors: Lista de vetores
        
    Returns:
        Matriz NxN de similaridades
    """
    if not vectors:
        return np.array([])
    
    # Empilha e normaliza
    matrix = np.vstack(vectors).astype(np.float32)
    matrix = l2_normalize(matrix)
    
    # Produto matricial (similaridade cosseno)
    return np.dot(matrix, matrix.T)


def intra_cluster_cohesion(vectors: List[np.ndarray]) -> float:
    """
    Calcula coesão intra-cluster (média de similaridades)
    
    Args:
        vectors: Lista de vetores do cluster
        
    Returns:
        Coesão média (0 a 1)
    """
    if len(vectors) < 2:
        return 1.0
    
    sim_matrix = pairwise_cosine_similarity(vectors)
    
    # Pega apenas triângulo superior (sem diagonal)
    n = len(vectors)
    similarities = []
    for i in range(n):
        for j in range(i+1, n):
            similarities.append(sim_matrix[i, j])
    
    return float(np.mean(similarities)) if similarities else 0.0


def inter_cluster_separation(
    cluster_a: List[np.ndarray],
    cluster_b: List[np.ndarray]
) -> float:
    """
    Calcula separação entre dois clusters (distância entre centróides)
    
    Args:
        cluster_a: Vetores do cluster A
        cluster_b: Vetores do cluster B
        
    Returns:
        Distância cosseno entre centróides (0 a 2)
    """
    centroid_a = centroid(cluster_a)
    centroid_b = centroid(cluster_b)
    
    if centroid_a is None or centroid_b is None:
        return 0.0
    
    return cosine_distance(centroid_a, centroid_b)


def silhouette_coefficient(
    vector: np.ndarray,
    own_cluster: List[np.ndarray],
    other_cluster: List[np.ndarray]
) -> float:
    """
    Calcula coeficiente de silhueta simplificado para um vetor
    
    Args:
        vector: Vetor a avaliar
        own_cluster: Vetores do próprio cluster
        other_cluster: Vetores do outro cluster
        
    Returns:
        Coeficiente de silhueta (-1 a 1)
    """
    if not own_cluster or not other_cluster:
        return 0.0
    
    # Distância média intra-cluster (a)
    a = np.mean([cosine_distance(vector, v) for v in own_cluster])
    
    # Distância média inter-cluster (b)
    b = np.mean([cosine_distance(vector, v) for v in other_cluster])
    
    # Silhueta
    return (b - a) / max(a, b) if max(a, b) > 0 else 0.0


def average_silhouette(
    cluster_a: List[np.ndarray],
    cluster_b: List[np.ndarray]
) -> Dict[str, float]:
    """
    Calcula silhueta média para separação entre dois clusters
    
    Args:
        cluster_a: Vetores do cluster A
        cluster_b: Vetores do cluster B
        
    Returns:
        Dict com silhueta média de cada cluster e geral
    """
    from typing import Dict
    
    if not cluster_a or not cluster_b:
        return {"cluster_a": 0.0, "cluster_b": 0.0, "overall": 0.0}
    
    # Silhueta para cada vetor do cluster A
    silhouettes_a = [
        silhouette_coefficient(v, cluster_a, cluster_b)
        for v in cluster_a
    ]
    
    # Silhueta para cada vetor do cluster B
    silhouettes_b = [
        silhouette_coefficient(v, cluster_b, cluster_a)
        for v in cluster_b
    ]
    
    return {
        "cluster_a": float(np.mean(silhouettes_a)),
        "cluster_b": float(np.mean(silhouettes_b)),
        "overall": float(np.mean(silhouettes_a + silhouettes_b))
    }

