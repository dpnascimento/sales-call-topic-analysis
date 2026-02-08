"""
Análises Geométricas de Embeddings
Inclui PCA interpretável, análise de subespaços discriminativos e métricas geométricas
"""
import logging
from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass

from core.database_v3 import DatabaseManagerV3
from core.embeddings_v3 import from_pgvector, cosine_similarity
from config import settings_v3

log = logging.getLogger(__name__)


@dataclass
class PCAComponent:
    """Representa um componente principal interpretado"""
    component_id: int
    variance_explained: float
    cumulative_variance: float
    top_dimensions: List[Tuple[int, float]]  # (dim_index, loading)
    correlated_patterns: List[Dict]  # Padrões correlacionados
    interpretation: str


class EmbeddingGeometryAnalyzer:
    """
    Análises geométricas e matemáticas no espaço de embeddings
    """
    
    def __init__(self, db: DatabaseManagerV3):
        self.db = db
        self.pca_cache = {}
        
    def analyze_interpretable_pca(
        self,
        embeddings: np.ndarray,
        metadata: List[Dict],
        n_components: int = 10,
        pattern_features: Optional[np.ndarray] = None,
        pattern_names: Optional[List[str]] = None
    ) -> Dict:
        """
        Análise PCA com interpretação semântica
        
        Args:
            embeddings: Matriz de embeddings (n_samples, 768)
            metadata: Lista de metadados (outcome, product, call_id)
            n_components: Número de componentes principais
            pattern_features: Matriz de contagem de padrões (n_samples, n_patterns)
            pattern_names: Nomes dos padrões
            
        Returns:
            Dict com componentes, interpretações e projeções
        """
        from sklearn.decomposition import PCA
        
        log.info(f"Executando PCA interpretável ({n_components} componentes)")
        
        # PCA
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(embeddings)  # (n_samples, n_components)
        
        log.info(f"Variância explicada (total): {pca.explained_variance_ratio_.sum():.1%}")
        
        # Analisa cada componente
        components = []
        cumulative_var = 0
        
        for i in range(n_components):
            cumulative_var += pca.explained_variance_ratio_[i]
            
            # Top dimensões que contribuem para este PC
            loadings = pca.components_[i]  # (768,)
            top_dim_indices = np.argsort(np.abs(loadings))[-20:][::-1]
            top_dimensions = [
                (int(idx), float(loadings[idx]))
                for idx in top_dim_indices
            ]
            
            # Correlação com padrões (se disponível)
            correlated_patterns = []
            interpretation = f"Componente {i+1}"
            
            if pattern_features is not None and pattern_names is not None:
                correlations = self._correlate_pc_with_patterns(
                    X_pca[:, i],
                    pattern_features,
                    pattern_names
                )
                correlated_patterns = correlations
                interpretation = self._interpret_pc(correlations)
            
            component = PCAComponent(
                component_id=i+1,
                variance_explained=float(pca.explained_variance_ratio_[i]),
                cumulative_variance=float(cumulative_var),
                top_dimensions=top_dimensions[:10],  # Top 10
                correlated_patterns=correlated_patterns,
                interpretation=interpretation
            )
            
            components.append(component)
            
            log.info(
                f"  PC{i+1}: {pca.explained_variance_ratio_[i]:.1%} variância "
                f"(acum: {cumulative_var:.1%}) - {interpretation}"
            )
        
        # Analisa separação por outcome nos primeiros 2 PCs
        separation_analysis = self._analyze_pc_separation(
            X_pca[:, :2],
            [m['outcome'] for m in metadata]
        )
        
        return {
            "n_components": n_components,
            "n_samples": embeddings.shape[0],
            "n_dimensions": embeddings.shape[1],
            "components": [self._component_to_dict(c) for c in components],
            "pca_projections": X_pca.tolist(),  # Para visualizações
            "separation_analysis": separation_analysis,
            "pca_model": pca  # Para transformar novos dados
        }
    
    def _correlate_pc_with_patterns(
        self,
        pc_scores: np.ndarray,
        pattern_features: np.ndarray,
        pattern_names: List[str],
        min_corr: float = 0.15,
        max_pvalue: float = 0.05
    ) -> List[Dict]:
        """
        Correlaciona scores de um PC com features de padrões
        """
        from scipy.stats import spearmanr
        
        correlations = []
        
        for j, pattern_name in enumerate(pattern_names):
            pattern_count = pattern_features[:, j]
            
            # Ignora padrões com zero ocorrências
            if pattern_count.sum() == 0:
                continue
            
            corr, pval = spearmanr(pc_scores, pattern_count)
            
            if abs(corr) >= min_corr and pval < max_pvalue:
                correlations.append({
                    "pattern": pattern_name,
                    "correlation": float(corr),
                    "p_value": float(pval),
                    "direction": "positive" if corr > 0 else "negative"
                })
        
        # Ordena por correlação absoluta
        correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        return correlations
    
    def _interpret_pc(self, correlations: List[Dict], top_k: int = 3) -> str:
        """
        Gera interpretação textual de um componente
        """
        if not correlations:
            return "Não interpretável (sem correlações significativas)"
        
        top_corr = correlations[:top_k]
        
        pos = [c['pattern'] for c in top_corr if c['correlation'] > 0]
        neg = [c['pattern'] for c in top_corr if c['correlation'] < 0]
        
        parts = []
        if pos:
            parts.append(" + ".join(pos[:2]))
        if neg:
            parts.append(" vs ".join(neg[:2]))
        
        if parts:
            return "Eixo: " + " ↔ ".join(parts)
        
        return "Eixo: " + pos[0] if pos else correlations[0]['pattern']
    
    def _analyze_pc_separation(
        self,
        pc_scores: np.ndarray,
        outcomes: List[str]
    ) -> Dict:
        """
        Analisa separação entre ganhas/perdidas nos PCs
        """
        outcomes_array = np.array(outcomes)
        
        ganha_mask = outcomes_array == 'ganha'
        perdida_mask = outcomes_array == 'perdida'
        
        pc1_ganha = pc_scores[ganha_mask, 0]
        pc1_perdida = pc_scores[perdida_mask, 0]
        
        pc2_ganha = pc_scores[ganha_mask, 1]
        pc2_perdida = pc_scores[perdida_mask, 1]
        
        from scipy.stats import mannwhitneyu
        
        # Teste estatístico
        stat_pc1, pval_pc1 = mannwhitneyu(pc1_ganha, pc1_perdida)
        stat_pc2, pval_pc2 = mannwhitneyu(pc2_ganha, pc2_perdida)
        
        return {
            "pc1": {
                "mean_ganha": float(np.mean(pc1_ganha)),
                "mean_perdida": float(np.mean(pc1_perdida)),
                "std_ganha": float(np.std(pc1_ganha)),
                "std_perdida": float(np.std(pc1_perdida)),
                "mannwhitney_u": float(stat_pc1),
                "p_value": float(pval_pc1),
                "significant": pval_pc1 < 0.05
            },
            "pc2": {
                "mean_ganha": float(np.mean(pc2_ganha)),
                "mean_perdida": float(np.mean(pc2_perdida)),
                "std_ganha": float(np.std(pc2_ganha)),
                "std_perdida": float(np.std(pc2_perdida)),
                "mannwhitney_u": float(stat_pc2),
                "p_value": float(pval_pc2),
                "significant": pval_pc2 < 0.05
            }
        }
    
    def _component_to_dict(self, component: PCAComponent) -> Dict:
        """Converte PCAComponent para dict"""
        return {
            "component_id": component.component_id,
            "variance_explained": component.variance_explained,
            "cumulative_variance": component.cumulative_variance,
            "top_dimensions": component.top_dimensions,
            "correlated_patterns": component.correlated_patterns,
            "interpretation": component.interpretation
        }
    
    def analyze_discriminative_subspace(
        self,
        embeddings_ganha: np.ndarray,
        embeddings_perdida: np.ndarray,
        top_k: int = 20
    ) -> Dict:
        """
        Identifica dimensões mais discriminativas usando LDA
        
        Args:
            embeddings_ganha: Embeddings de chamadas ganhas (n_ganha, 768)
            embeddings_perdida: Embeddings de chamadas perdidas (n_perdida, 768)
            top_k: Número de dimensões top
            
        Returns:
            Dict com dimensões discriminativas e scores
        """
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        
        log.info("Analisando subespaço discriminativo (LDA)")
        
        # Prepara dados
        X = np.vstack([embeddings_ganha, embeddings_perdida])
        y = np.array([1]*len(embeddings_ganha) + [0]*len(embeddings_perdida))
        
        # LDA
        lda = LinearDiscriminantAnalysis()
        lda.fit(X, y)
        
        # Importância de cada dimensão
        coef = np.abs(lda.coef_[0])  # (768,)
        top_dims = np.argsort(coef)[-top_k:][::-1]
        
        discriminative_power = coef[top_dims] / coef.sum()  # Normalizado
        
        log.info(f"Top {top_k} dimensões explicam {discriminative_power.sum():.1%} da discriminação")
        
        return {
            "top_dimensions": top_dims.tolist(),
            "importance_scores": coef[top_dims].tolist(),
            "normalized_power": discriminative_power.tolist(),
            "explained_discrimination": float(discriminative_power.sum()),
            "lda_accuracy": float(lda.score(X, y))
        }
    
    def analyze_view_alignment(
        self,
        embeddings_full: List[np.ndarray],
        embeddings_agent: List[np.ndarray],
        embeddings_client: List[np.ndarray],
        outcomes: List[str]
    ) -> Dict:
        """
        Analisa alinhamento entre visões (full/agent/client)
        
        Args:
            embeddings_*: Listas de embeddings para cada visão
            outcomes: Lista de outcomes correspondentes
            
        Returns:
            Dict com métricas de alinhamento
        """
        log.info("Analisando alinhamento entre visões")
        
        alignments = []
        
        for i in range(len(embeddings_full)):
            sim_agent_client = cosine_similarity(embeddings_agent[i], embeddings_client[i])
            sim_full_agent = cosine_similarity(embeddings_full[i], embeddings_agent[i])
            sim_full_client = cosine_similarity(embeddings_full[i], embeddings_client[i])
            
            alignment_score = (sim_agent_client + sim_full_agent + sim_full_client) / 3
            
            alignments.append({
                "outcome": outcomes[i],
                "alignment_score": float(alignment_score),
                "agent_client_similarity": float(sim_agent_client),
                "full_agent_similarity": float(sim_full_agent),
                "full_client_similarity": float(sim_full_client)
            })
        
        # Agregação por outcome
        outcomes_array = np.array(outcomes)
        ganha_mask = outcomes_array == 'ganha'
        
        align_scores = np.array([a['alignment_score'] for a in alignments])
        ac_sim = np.array([a['agent_client_similarity'] for a in alignments])
        
        from scipy.stats import mannwhitneyu
        
        stat, pval = mannwhitneyu(
            align_scores[ganha_mask],
            align_scores[~ganha_mask]
        )
        
        return {
            "n_samples": len(alignments),
            "alignment_mean_ganha": float(np.mean(align_scores[ganha_mask])),
            "alignment_mean_perdida": float(np.mean(align_scores[~ganha_mask])),
            "alignment_std_ganha": float(np.std(align_scores[ganha_mask])),
            "alignment_std_perdida": float(np.std(align_scores[~ganha_mask])),
            "agent_client_sync_ganha": float(np.mean(ac_sim[ganha_mask])),
            "agent_client_sync_perdida": float(np.mean(ac_sim[~ganha_mask])),
            "statistical_test": {
                "test": "Mann-Whitney U",
                "statistic": float(stat),
                "p_value": float(pval),
                "significant": pval < 0.05
            },
            "individual_alignments": alignments
        }
    
    def detect_outliers(
        self,
        embeddings: np.ndarray,
        metadata: List[Dict],
        contamination: float = 0.05
    ) -> Dict:
        """
        Detecta outliers no espaço de embeddings
        
        Args:
            embeddings: Matriz de embeddings
            metadata: Metadados (call_id, outcome)
            contamination: Proporção esperada de outliers
            
        Returns:
            Dict com outliers identificados
        """
        from sklearn.neighbors import LocalOutlierFactor
        
        log.info(f"Detectando outliers (contamination={contamination:.1%})")
        
        # LOF
        lof = LocalOutlierFactor(
            n_neighbors=20,
            contamination=contamination,
            metric='cosine'
        )
        outlier_labels = lof.fit_predict(embeddings)  # -1 = outlier
        outlier_scores = -lof.negative_outlier_factor_
        
        outlier_mask = outlier_labels == -1
        outlier_indices = np.where(outlier_mask)[0]
        
        # Ordena por score
        sorted_indices = outlier_indices[np.argsort(outlier_scores[outlier_mask])[::-1]]
        
        outliers = []
        for rank, idx in enumerate(sorted_indices, 1):
            outliers.append({
                "rank": rank,
                "call_id": metadata[idx]['call_id'],
                "outcome": metadata[idx]['outcome'],
                "product_name": metadata[idx].get('product_name', 'Unknown'),
                "outlier_score": float(outlier_scores[idx])
            })
        
        # Estatísticas por outcome
        outcomes = np.array([m['outcome'] for m in metadata])
        n_outliers_ganha = np.sum(outlier_mask & (outcomes == 'ganha'))
        n_outliers_perdida = np.sum(outlier_mask & (outcomes == 'perdida'))
        
        log.info(f"Outliers detectados: {len(outliers)} ({len(outliers)/len(embeddings):.1%})")
        log.info(f"  Ganhas: {n_outliers_ganha}, Perdidas: {n_outliers_perdida}")
        
        return {
            "n_outliers": len(outliers),
            "contamination_rate": float(len(outliers) / len(embeddings)),
            "outliers_by_outcome": {
                "ganha": int(n_outliers_ganha),
                "perdida": int(n_outliers_perdida)
            },
            "top_outliers": outliers[:50]  # Top 50
        }
    
    def run_complete_analysis(
        self,
        embedding_view: str = "full",
        include_patterns: bool = True,
        sample_size: int = None
    ) -> Dict:
        """
        Executa análise geométrica completa para uma visão
        
        Args:
            embedding_view: Visão a analisar ('full', 'agent', 'client')
            include_patterns: Se True, correlaciona PCs com padrões
            sample_size: Limite de amostras
            
        Returns:
            Dict com todas as análises
        """
        log.info(f"Executando análise geométrica completa - visão: {embedding_view}")
        
        if sample_size is None:
            sample_size = settings_v3.UMAP_SAMPLE_SIZE
        
        # Busca embeddings
        rows = self.db.get_all_embeddings_by_view(
            embedding_view=embedding_view,
            limit=sample_size
        )
        
        if not rows or len(rows) < 50:
            log.warning(f"Dados insuficientes: {len(rows)} amostras")
            return {}
        
        # Converte embeddings e metadados
        embeddings_list = []
        metadata = []
        
        for row in rows:
            vec = from_pgvector(row['embedding_text'])
            if vec.size > 0:
                embeddings_list.append(vec)
                metadata.append({
                    'call_id': row['call_id'],
                    'outcome': row['outcome'],
                    'product_name': row['product_name']
                })
        
        embeddings = np.vstack(embeddings_list)
        
        log.info(f"Analisando {len(embeddings)} embeddings ({embeddings.shape[1]} dimensões)")
        
        # Carrega padrões (se disponível)
        pattern_features = None
        pattern_names = None
        
        if include_patterns:
            try:
                pattern_data = self._load_pattern_features(
                    [m['call_id'] for m in metadata]
                )
                if pattern_data:
                    pattern_features = pattern_data['features']
                    pattern_names = pattern_data['names']
                    log.info(f"Carregados {len(pattern_names)} padrões para correlação")
            except Exception as e:
                log.warning(f"Não foi possível carregar padrões: {e}")
        
        results = {
            "embedding_view": embedding_view,
            "n_samples": len(embeddings),
            "n_dimensions": embeddings.shape[1]
        }
        
        # 1. PCA Interpretável
        pca_analysis = self.analyze_interpretable_pca(
            embeddings=embeddings,
            metadata=metadata,
            n_components=10,
            pattern_features=pattern_features,
            pattern_names=pattern_names
        )
        results["pca_analysis"] = pca_analysis
        
        # 2. Subespaço Discriminativo (LDA)
        outcomes = np.array([m['outcome'] for m in metadata])
        ganha_mask = outcomes == 'ganha'
        
        if ganha_mask.sum() >= 10 and (~ganha_mask).sum() >= 10:
            lda_analysis = self.analyze_discriminative_subspace(
                embeddings_ganha=embeddings[ganha_mask],
                embeddings_perdida=embeddings[~ganha_mask],
                top_k=20
            )
            results["lda_analysis"] = lda_analysis
        
        # 3. Detecção de Outliers
        outlier_analysis = self.detect_outliers(
            embeddings=embeddings,
            metadata=metadata,
            contamination=0.05
        )
        results["outlier_analysis"] = outlier_analysis
        
        return results
    
    def _load_pattern_features(self, call_ids: List[str]) -> Optional[Dict]:
        """
        Carrega features de padrões para as calls
        Placeholder - implementar baseado nos enhanced_patterns
        """
        # TODO: Implementar carregamento de padrões
        # Por enquanto retorna None
        return None


