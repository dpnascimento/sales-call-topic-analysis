"""
Análise de protótipos
Suporta múltiplas visões de embedding (full, agent, client)
"""
import logging
from typing import Dict, List, Optional
import numpy as np

from core.database_v3 import DatabaseManagerV3
from core.embeddings_v3 import (
    from_pgvector, l2_normalize, centroid, 
    cosine_similarity, cosine_distance,
    intra_cluster_cohesion, inter_cluster_separation,
    average_silhouette
)
from config import settings_v3

log = logging.getLogger(__name__)


class PrototypeAnalyzerV3:
    """
    Analisa protótipos para diferentes visões de embedding
    Calcula protótipos globais, por produto e por produto+status
    """
    
    def __init__(self, db: DatabaseManagerV3):
        self.db = db
        self.prototypes = {}  # Cache de protótipos
    
    def compute_global_prototypes(
        self, 
        embedding_view: str = "full"
    ) -> Dict[str, Dict]:
        """
        Calcula protótipos globais (ganha vs perdida)
        
        Args:
            embedding_view: 'full', 'agent' ou 'client'
            
        Returns:
            Dict com protótipos e estatísticas
        """
        log.info(f"Computando protótipos globais - visão: {embedding_view}")
        
        results = {}
        
        for outcome in ["ganha", "perdida"]:
            # Busca embeddings
            rows = self.db.get_all_embeddings_by_view(
                embedding_view=embedding_view,
                outcome=outcome
            )
            
            if not rows:
                log.warning(f"Nenhum embedding encontrado para outcome={outcome}")
                continue
            
            # Converte embeddings
            vectors = []
            for row in rows:
                vec = from_pgvector(row['embedding_text'])
                if vec.size > 0:
                    vectors.append(l2_normalize(vec))
            
            if not vectors:
                log.warning(f"Nenhum vetor válido para outcome={outcome}")
                continue
            
            # Calcula protótipo (centróide)
            proto = centroid(vectors)
            
            if proto is None:
                continue
            
            # Calcula coesão intra-cluster
            cohesion = intra_cluster_cohesion(vectors)
            
            # Calcula distâncias ao protótipo
            distances = [cosine_distance(v, proto) for v in vectors]
            
            results[outcome] = {
                "prototype": proto,
                "n_samples": len(vectors),
                "cohesion": cohesion,
                "mean_distance": float(np.mean(distances)),
                "std_distance": float(np.std(distances)),
                "median_distance": float(np.median(distances))
            }
            
            log.info(f"Protótipo '{outcome}': {len(vectors)} amostras, coesão={cohesion:.4f}")
        
        # Análise de separação entre protótipos
        if "ganha" in results and "perdida" in results:
            separation_metrics = self._compute_separation_metrics(
                results["ganha"]["prototype"],
                results["perdida"]["prototype"],
                [from_pgvector(r['embedding_text']) for r in self.db.get_all_embeddings_by_view(embedding_view, "ganha")],
                [from_pgvector(r['embedding_text']) for r in self.db.get_all_embeddings_by_view(embedding_view, "perdida")]
            )
            results["separation"] = separation_metrics
            
            log.info(f"Separação entre protótipos: {separation_metrics['distance']:.4f}")
            log.info(f"Silhueta média: {separation_metrics['silhouette']['overall']:.4f}")
        
        # Cacheia resultados
        cache_key = f"global_{embedding_view}"
        self.prototypes[cache_key] = results
        
        return results
    
    def compute_product_prototypes(
        self,
        product_name: str,
        embedding_view: str = "full"
    ) -> Dict[str, Dict]:
        """
        Calcula protótipos para um produto específico
        
        Args:
            product_name: Nome do produto
            embedding_view: Visão de embedding
            
        Returns:
            Dict com protótipos por outcome
        """
        log.info(f"Calcula protótipos - produto: {product_name}, visão: {embedding_view}")
        
        results = {}
        
        for outcome in ["ganha", "perdida"]:
            rows = self.db.get_all_embeddings_by_view(
                embedding_view=embedding_view,
                outcome=outcome,
                product_name=product_name
            )
            
            if not rows:
                continue
            
            vectors = []
            for row in rows:
                vec = from_pgvector(row['embedding_text'])
                if vec.size > 0:
                    vectors.append(l2_normalize(vec))
            
            if len(vectors) < settings_v3.MIN_CALLS_PER_PRODUCT_STATUS:
                log.warning(f"Produto '{product_name}' - outcome '{outcome}': amostras insuficientes ({len(vectors)})")
                continue
            
            proto = centroid(vectors)
            if proto is None:
                continue
            
            cohesion = intra_cluster_cohesion(vectors)
            distances = [cosine_distance(v, proto) for v in vectors]
            
            results[outcome] = {
                "prototype": proto,
                "n_samples": len(vectors),
                "cohesion": cohesion,
                "mean_distance": float(np.mean(distances)),
                "std_distance": float(np.std(distances))
            }
        
        # Separação
        if "ganha" in results and "perdida" in results:
            ganha_vecs = [from_pgvector(r['embedding_text']) for r in self.db.get_all_embeddings_by_view(embedding_view, "ganha", product_name)]
            perdida_vecs = [from_pgvector(r['embedding_text']) for r in self.db.get_all_embeddings_by_view(embedding_view, "perdida", product_name)]
            
            separation_metrics = self._compute_separation_metrics(
                results["ganha"]["prototype"],
                results["perdida"]["prototype"],
                ganha_vecs,
                perdida_vecs
            )
            results["separation"] = separation_metrics
        
        # Cacheia
        cache_key = f"product_{product_name}_{embedding_view}"
        self.prototypes[cache_key] = results
        
        return results
    
    def compute_all_products_prototypes(
        self,
        embedding_view: str = "full"
    ) -> Dict[str, Dict]:
        """
        Calcula protótipos para todos os produtos
        
        Args:
            embedding_view: Visão de embedding
            
        Returns:
            Dict mapeando produto -> protótipos
        """
        log.info(f"Calcula protótipos para todos os produtos - visão: {embedding_view}")
        
        # Busca produtos elegíveis
        products = self.db.get_products(min_calls=settings_v3.MIN_CALLS_PER_PRODUCT)
        
        results = {}
        
        for product in products:
            product_name = product['product_name']
            
            try:
                product_protos = self.compute_product_prototypes(product_name, embedding_view)
                if product_protos:
                    results[product_name] = product_protos
            except Exception as e:
                log.error(f"Erro ao processar produto '{product_name}': {e}")
        
        log.info(f"Protótipos computados para {len(results)} produtos")
        
        return results
    
    def compare_products_separation(
        self,
        embedding_view: str = "full"
    ) -> List[Dict]:
        """
        Compara separação ganha/perdida entre produtos
        
        Args:
            embedding_view: Visão de embedding
            
        Returns:
            Lista ordenada por qualidade de separação
        """
        log.info(f"Comparando separação entre produtos - visão: {embedding_view}")
        
        products_protos = self.compute_all_products_prototypes(embedding_view)
        
        comparisons = []
        
        for product_name, protos in products_protos.items():
            if "separation" not in protos:
                continue
            
            sep = protos["separation"]
            
            comparison = {
                "product_name": product_name,
                "n_ganha": protos["ganha"]["n_samples"],
                "n_perdida": protos["perdida"]["n_samples"],
                "separation_distance": sep["distance"],
                "silhouette": sep["silhouette"]["overall"],
                "cohesion_ganha": protos["ganha"]["cohesion"],
                "cohesion_perdida": protos["perdida"]["cohesion"],
                "embedding_view": embedding_view
            }
            
            comparisons.append(comparison)
        
        # Ordena por silhueta (melhor separação)
        comparisons.sort(key=lambda x: x["silhouette"], reverse=True)
        
        # Log top 5
        log.info(f"\nTop 5 produtos com melhor separação ({embedding_view}):")
        for i, comp in enumerate(comparisons[:5], 1):
            log.info(f"  {i}. {comp['product_name']}: silhueta={comp['silhouette']:.4f}, dist={comp['separation_distance']:.4f}")
        
        return comparisons
    
    def _compute_separation_metrics(
        self,
        proto_ganha: np.ndarray,
        proto_perdida: np.ndarray,
        vecs_ganha: List[np.ndarray],
        vecs_perdida: List[np.ndarray]
    ) -> Dict:
        """Calcula métricas de separação entre dois clusters"""
        
        # Distância entre protótipos
        dist = cosine_distance(proto_ganha, proto_perdida)
        
        # Silhueta (com amostragem para performance)
        if settings_v3.COMPUTE_SILHOUETTE:
            sample_size = settings_v3.SILHOUETTE_SAMPLE_SIZE
            
            # Amostragem se necessário
            if sample_size and (len(vecs_ganha) > sample_size or len(vecs_perdida) > sample_size):
                import random
                random.seed(42)  # Reproducibilidade
                
                sampled_ganha = random.sample(vecs_ganha, min(len(vecs_ganha), sample_size))
                sampled_perdida = random.sample(vecs_perdida, min(len(vecs_perdida), sample_size))
                
                log.debug(f"Amostragem silhueta: {len(sampled_ganha)} ganha, {len(sampled_perdida)} perdida")
                silhouette = average_silhouette(sampled_ganha, sampled_perdida)
            else:
                silhouette = average_silhouette(vecs_ganha, vecs_perdida)
        else:
            silhouette = {"cluster_a": 0.0, "cluster_b": 0.0, "overall": 0.0}
            log.debug("Cálculo de silhueta desabilitado (COMPUTE_SILHOUETTE=False)")
        
        # Similaridade entre protótipos
        similarity = cosine_similarity(proto_ganha, proto_perdida)
        
        return {
            "distance": float(dist),
            "similarity": float(similarity),
            "silhouette": silhouette,
            "inter_cluster_separation": inter_cluster_separation(vecs_ganha, vecs_perdida)
        }
    
    def get_prototype(
        self, 
        outcome: str,
        embedding_view: str = "full",
        product_name: str = None
    ) -> Optional[np.ndarray]:
        """
        Recupera protótipo do cache
        
        Args:
            outcome: 'ganha' ou 'perdida'
            embedding_view: Visão de embedding
            product_name: Nome do produto (opcional, se None = global)
            
        Returns:
            Protótipo ou None
        """
        if product_name:
            cache_key = f"product_{product_name}_{embedding_view}"
        else:
            cache_key = f"global_{embedding_view}"
        
        if cache_key not in self.prototypes:
            # Computa se não existe
            if product_name:
                self.compute_product_prototypes(product_name, embedding_view)
            else:
                self.compute_global_prototypes(embedding_view)
        
        protos = self.prototypes.get(cache_key, {})
        return protos.get(outcome, {}).get("prototype")
    
    def export_results(self, output_path: str):
        """
        Exporta resultados para arquivo JSON
        
        Args:
            output_path: Caminho do arquivo de saída
        """
        import json
        
        # Converte arrays numpy para listas para JSON
        export_data = {}
        
        for key, value in self.prototypes.items():
            export_data[key] = {}
            for outcome, metrics in value.items():
                if outcome == "separation":
                    export_data[key][outcome] = {
                        "distance": metrics["distance"],
                        "similarity": metrics["similarity"],
                        "silhouette": metrics["silhouette"],
                        "inter_cluster_separation": metrics["inter_cluster_separation"]
                    }
                elif isinstance(metrics, dict) and "prototype" in metrics:
                    export_data[key][outcome] = {
                        k: (v.tolist() if isinstance(v, np.ndarray) else v)
                        for k, v in metrics.items()
                        if k != "prototype"  # Não exporta protótipo completo (muito grande)
                    }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        log.info(f"Resultados exportados para: {output_path}")

