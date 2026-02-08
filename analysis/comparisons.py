"""
Módulo de comparação entre visões de embedding (full, agent, client)
Analisa qual visão produz melhor separação ganha/perdida
"""
import logging
from typing import Dict, List
import numpy as np

from core.database_v3 import DatabaseManagerV3
from core.embeddings_v3 import (
    from_pgvector, cosine_similarity, cosine_distance,
    average_silhouette, inter_cluster_separation
)
from analysis.prototypes_v3 import PrototypeAnalyzerV3
from config import settings_v3

log = logging.getLogger(__name__)


class EmbeddingViewComparator:
    """
    Compara performance de diferentes visões de embedding
    Identifica qual visão produz melhor separação semântica
    """
    
    def __init__(self, db: DatabaseManagerV3):
        self.db = db
        self.prototype_analyzer = PrototypeAnalyzerV3(db)
    
    def compare_views_global(self) -> Dict:
        """
        Compara visões de embedding em nível global
        
        Returns:
            Dict com métricas de cada visão
        """
        log.info("Comparando visões de embedding - nível global")
        
        results = {}
        
        for view in settings_v3.EMBEDDING_VIEWS:
            try:
                # Calcula protótipos para esta visão
                prototypes = self.prototype_analyzer.compute_global_prototypes(view)
                
                if "separation" in prototypes:
                    sep = prototypes["separation"]
                    
                    results[view] = {
                        "n_ganha": prototypes["ganha"]["n_samples"],
                        "n_perdida": prototypes["perdida"]["n_samples"],
                        "cohesion_ganha": prototypes["ganha"]["cohesion"],
                        "cohesion_perdida": prototypes["perdida"]["cohesion"],
                        "separation_distance": sep["distance"],
                        "separation_similarity": sep["similarity"],
                        "silhouette_overall": sep["silhouette"]["overall"],
                        "silhouette_ganha": sep["silhouette"]["cluster_a"],
                        "silhouette_perdida": sep["silhouette"]["cluster_b"]
                    }
                    
                    log.info(f"Visão '{view}': silhueta={sep['silhouette']['overall']:.4f}, dist={sep['distance']:.4f}")
                
            except Exception as e:
                log.error(f"Erro ao processar visão '{view}': {e}")
        
        # Rankeia visões por silhueta
        ranked = sorted(results.items(), key=lambda x: x[1]["silhouette_overall"], reverse=True)
        
        log.info(f"\nRanking de visões (por silhueta):")
        for i, (view, metrics) in enumerate(ranked, 1):
            log.info(f"  {i}. {view}: {metrics['silhouette_overall']:.4f}")
        
        return {
            "by_view": results,
            "ranking": [{"view": v, "silhouette": m["silhouette_overall"]} for v, m in ranked],
            "best_view": ranked[0][0] if ranked else None
        }
    
    def compare_views_by_product(self, product_name: str) -> Dict:
        """
        Compara visões para um produto específico
        
        Args:
            product_name: Nome do produto
            
        Returns:
            Dict com métricas por visão
        """
        log.info(f"Comparando visões - produto: {product_name}")
        
        results = {}
        
        for view in settings_v3.EMBEDDING_VIEWS:
            try:
                # Calcula protótipos do produto
                prototypes = self.prototype_analyzer.compute_product_prototypes(product_name, view)
                
                if "separation" in prototypes:
                    sep = prototypes["separation"]
                    
                    results[view] = {
                        "n_ganha": prototypes["ganha"]["n_samples"],
                        "n_perdida": prototypes["perdida"]["n_samples"],
                        "cohesion_ganha": prototypes["ganha"]["cohesion"],
                        "cohesion_perdida": prototypes["perdida"]["cohesion"],
                        "separation_distance": sep["distance"],
                        "silhouette_overall": sep["silhouette"]["overall"]
                    }
                
            except Exception as e:
                log.error(f"Erro ao processar visão '{view}' para produto '{product_name}': {e}")
        
        # Rankeia
        ranked = sorted(results.items(), key=lambda x: x[1]["silhouette_overall"], reverse=True)
        
        return {
            "product_name": product_name,
            "by_view": results,
            "ranking": [{"view": v, "silhouette": m["silhouette_overall"]} for v, m in ranked],
            "best_view": ranked[0][0] if ranked else None
        }
    
    def compare_views_all_products(self) -> Dict:
        """
        Compara visões para todos os produtos
        
        Returns:
            Dict agregado com análise por produto
        """
        log.info("Comparando visões para todos os produtos")
        
        # Busca produtos
        products = self.db.get_products(min_calls=settings_v3.MIN_CALLS_PER_PRODUCT)
        
        product_comparisons = {}
        
        for product in products:
            product_name = product['product_name']
            
            try:
                comparison = self.compare_views_by_product(product_name)
                if comparison and comparison.get("best_view"):
                    product_comparisons[product_name] = comparison
            except Exception as e:
                log.error(f"Erro ao comparar produto '{product_name}': {e}")
        
        # Agrega estatísticas
        best_view_counts = {}
        for comp in product_comparisons.values():
            best_view = comp["best_view"]
            best_view_counts[best_view] = best_view_counts.get(best_view, 0) + 1
        
        # Calcula win rate de cada visão por produto
        view_performance = {view: [] for view in settings_v3.EMBEDDING_VIEWS}
        
        for product_name, comp in product_comparisons.items():
            for view, metrics in comp["by_view"].items():
                view_performance[view].append({
                    "product": product_name,
                    "silhouette": metrics["silhouette_overall"],
                    "separation": metrics["separation_distance"]
                })
        
        # Médias por visão
        avg_performance = {}
        for view, performances in view_performance.items():
            if performances:
                avg_performance[view] = {
                    "avg_silhouette": float(np.mean([p["silhouette"] for p in performances])),
                    "avg_separation": float(np.mean([p["separation"] for p in performances])),
                    "std_silhouette": float(np.std([p["silhouette"] for p in performances])),
                    "n_products": len(performances)
                }
        
        return {
            "n_products_analyzed": len(product_comparisons),
            "product_comparisons": product_comparisons,
            "best_view_counts": best_view_counts,
            "avg_performance_by_view": avg_performance,
            "overall_best_view": max(best_view_counts.items(), key=lambda x: x[1])[0] if best_view_counts else None
        }
    
    def generate_view_recommendations(self) -> Dict:
        """
        Gera recomendações sobre qual visão usar
        
        Returns:
            Dict com recomendações estratégicas
        """
        log.info("Gerando recomendações sobre visões de embedding")
        
        # Comparação global
        global_comp = self.compare_views_global()
        
        # Comparação por produtos
        products_comp = self.compare_views_all_products()
        
        recommendations = {
            "global": {
                "best_view": global_comp["best_view"],
                "reason": self._explain_best_view(global_comp["by_view"])
            },
            "by_product_type": {
                "most_consistent": products_comp["overall_best_view"],
                "counts": products_comp["best_view_counts"],
                "avg_performance": products_comp["avg_performance_by_view"]
            },
            "general_recommendations": []
        }
        
        # Gera recomendações gerais
        best_global = global_comp["best_view"]
        best_products = products_comp["overall_best_view"]
        
        if best_global == best_products:
            recommendations["general_recommendations"].append(
                f"✅ Visão '{best_global}' é consistentemente a melhor (global e por produtos)"
            )
        else:
            recommendations["general_recommendations"].append(
                f"⚠️ Visão varia: '{best_global}' globalmente, mas '{best_products}' domina em produtos"
            )
        
        # Análise de variância
        for view, perf in products_comp["avg_performance_by_view"].items():
            std = perf["std_silhouette"]
            if std < 0.1:
                recommendations["general_recommendations"].append(
                    f"✅ Visão '{view}': performance consistente entre produtos (std={std:.4f})"
                )
            elif std > 0.2:
                recommendations["general_recommendations"].append(
                    f"⚠️ Visão '{view}': alta variância entre produtos (std={std:.4f})"
                )
        
        # Recomendação final
        avg_silhouettes = {
            view: perf["avg_silhouette"] 
            for view, perf in products_comp["avg_performance_by_view"].items()
        }
        
        if avg_silhouettes:
            best_avg_view = max(avg_silhouettes.items(), key=lambda x: x[1])
            recommendations["final_recommendation"] = {
                "recommended_view": best_avg_view[0],
                "avg_silhouette": best_avg_view[1],
                "justification": f"Melhor performance média através de {products_comp['n_products_analyzed']} produtos"
            }
        
        return recommendations
    
    def _explain_best_view(self, view_metrics: Dict) -> str:
        """Explica por que uma visão é melhor"""
        if not view_metrics:
            return "Sem dados suficientes"
        
        # Ordena por silhueta
        ranked = sorted(view_metrics.items(), key=lambda x: x[1]["silhouette_overall"], reverse=True)
        
        best_view, best_metrics = ranked[0]
        
        explanation = f"Melhor silhueta ({best_metrics['silhouette_overall']:.4f}) "
        explanation += f"com separação de {best_metrics['separation_distance']:.4f}. "
        explanation += f"Coesão interna: ganha={best_metrics['cohesion_ganha']:.4f}, "
        explanation += f"perdida={best_metrics['cohesion_perdida']:.4f}"
        
        return explanation
    
    def export_comparison_results(self, output_path: str):
        """
        Exporta comparação completa para arquivo
        
        Args:
            output_path: Caminho do arquivo de saída
        """
        log.info("Exportando resultados de comparação")
        
        results = {
            "global_comparison": self.compare_views_global(),
            "products_comparison": self.compare_views_all_products(),
            "recommendations": self.generate_view_recommendations()
        }
        
        import json
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        log.info(f"Resultados exportados para: {output_path}")
        
        return results

