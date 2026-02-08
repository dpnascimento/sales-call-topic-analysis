"""
Análise de Tópicos com BERTopic
Usa embeddings V2 (call_embeddings_v2) para identificar tópicos semânticos
"""
import logging
import os
import json
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime

from core.database_v3 import DatabaseManagerV3
from core.embeddings_v3 import from_pgvector, l2_normalize
from config import settings_v3

log = logging.getLogger(__name__)

# Importações opcionais do BERTopic
HAS_BERTOPIC = True
HAS_WORDCLOUD = True
try:
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    from bertopic.vectorizers import ClassTfidfTransformer
    from bertopic.representation import KeyBERTInspired
    from umap import UMAP
    from wordcloud import WordCloud
except ImportError as e:
    HAS_BERTOPIC = False
    IMPORT_ERR = e
    try:
        from wordcloud import WordCloud
    except ImportError:
        HAS_WORDCLOUD = False

class TopicAnalyzerV3:
    """Analisa tópicos usando BERTopic com embeddings V2"""
    
    def __init__(self, db: DatabaseManagerV3):
        self.db = db
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def analyze_topics(self, embedding_view: str = "full") -> Dict:
        """
        Analisa tópicos usando BERTopic
        
        Args:
            embedding_view: Visão de embedding ('full', 'agent', 'client')
            
        Returns:
            Estatísticas da análise de tópicos e paths dos outputs
        """
        log.info(f"Iniciando análise de tópicos - visão: {embedding_view}")
        
        if not settings_v3.DO_TOPICS:
            log.info("Análise de tópicos desativada por configuração")
            return {"status": "disabled"}
        
        if not HAS_BERTOPIC:
            log.error(f"BERTopic não disponível: {IMPORT_ERR}")
            return {"status": "error", "error": str(IMPORT_ERR)}
        
        # Carrega dados
        docs, embeddings, metadata = self._load_topic_data(embedding_view)
        if not docs:
            log.warning("Nenhum documento elegível para análise de tópicos")
            return {"status": "no_data"}
        
        # Configura e treina BERTopic (passa n_docs para ajuste automático)
        topic_model = self._setup_bertopic(n_docs=len(docs))
        
        log.info(f"Treinando BERTopic em {len(docs)} documentos...")
        topics, probs = topic_model.fit_transform(docs, embeddings=embeddings)
        
        # Redução de outliers (alinhado com V2)
        outliers_before = np.sum(np.array(topics) == -1)
        if outliers_before > 0:
            log.info(f"🔧 Reduzindo outliers ({outliers_before} documentos)...")
            topics = topic_model.reduce_outliers(
                docs,
                topics,
                strategy="distributions",
                probabilities=probs
            )
            outliers_after = np.sum(np.array(topics) == -1)
            reduced = outliers_before - outliers_after
            log.info(f"✓ {reduced} outliers reassignados ({outliers_after} restantes)")
        
        # Calcula estatísticas
        stats = self._compute_topic_stats(topic_model, topics, metadata, embedding_view)
        
        # Salva resultados
        output_paths = self._save_results(
            topic_model, topics, probs, docs, metadata, embedding_view, stats
        )
        
        # Gera visualizações
        self._generate_visualizations(
            topic_model, topics, probs, embeddings, docs, metadata, embedding_view
        )
        
        # Gera word clouds
        if settings_v3.TOPICS_GENERATE_PLOTS:
            self._generate_wordclouds(
                topic_model, topics, metadata, embedding_view
            )
        
        # Gera análise temporal (Topics over Time)
        if settings_v3.TOPICS_GENERATE_OVERTIME:
            self._generate_topics_over_time(
                topic_model, topics, docs, metadata, embedding_view
            )
        
        # Gera scatter plots PCA
        if settings_v3.TOPICS_GENERATE_PLOTS:
            self._generate_pca_scatter_plots(
                embeddings, topics, metadata, embedding_view
            )
        
        log.info(f"Análise de tópicos concluída: {stats['n_topics']} tópicos identificados")
        
        return {
            "status": "success",
            "embedding_view": embedding_view,
            "stats": stats,
            "output_paths": output_paths
        }
    
    def analyze_topics_by_product(
        self,
        product_name: str,
        embedding_view: str = "full"
    ) -> Dict:
        """
        Analisa tópicos para um produto específico usando BERTopic
        
        Args:
            product_name: Nome do produto
            embedding_view: Visão de embedding ('full', 'agent', 'client')
            
        Returns:
            Estatísticas da análise de tópicos e paths dos outputs
        """
        log.info(f"Iniciando análise de tópicos por produto - {product_name} - visão: {embedding_view}")
        
        if not settings_v3.DO_TOPICS:
            log.info("Análise de tópicos desativada por configuração")
            return {"status": "disabled"}
        
        if not HAS_BERTOPIC:
            log.error(f"BERTopic não disponível: {IMPORT_ERR}")
            return {"status": "error", "error": str(IMPORT_ERR)}
        
        # Carrega dados DO PRODUTO ESPECÍFICO
        docs, embeddings, metadata = self._load_topic_data(embedding_view, product_name)
        if not docs:
            log.warning(f"Nenhum documento elegível para produto '{product_name}'")
            return {"status": "no_data", "product_name": product_name}
        
        log.info(f"Produto '{product_name}': {len(docs)} documentos encontrados")
        
        # Configura e treina BERTopic (passa n_docs para ajuste automático)
        topic_model = self._setup_bertopic(n_docs=len(docs))
        
        log.info(f"Treinando BERTopic em {len(docs)} documentos do produto '{product_name}'...")
        topics, probs = topic_model.fit_transform(docs, embeddings=embeddings)
        
        # Redução de outliers
        outliers_before = np.sum(np.array(topics) == -1)
        if outliers_before > 0:
            log.info(f"🔧 Reduzindo outliers ({outliers_before} documentos)...")
            topics = topic_model.reduce_outliers(
                docs,
                topics,
                strategy="distributions",
                probabilities=probs
            )
            outliers_after = np.sum(np.array(topics) == -1)
            reduced = outliers_before - outliers_after
            log.info(f"✓ {reduced} outliers reassignados ({outliers_after} restantes)")
        
        # Calcula estatísticas
        stats = self._compute_topic_stats(topic_model, topics, metadata, embedding_view)
        stats['product_name'] = product_name
        
        # Salva resultados COM SUFIXO DO PRODUTO
        output_paths = self._save_results(
            topic_model, topics, probs, docs, metadata, embedding_view, stats, product_name
        )
        
        # Gera visualizações
        self._generate_visualizations(
            topic_model, topics, probs, embeddings, docs, metadata, embedding_view, product_name
        )
        
        # Gera word clouds
        if settings_v3.TOPICS_GENERATE_PLOTS:
            self._generate_wordclouds(
                topic_model, topics, metadata, embedding_view, product_name
            )
        
        # Gera análise temporal
        if settings_v3.TOPICS_GENERATE_OVERTIME:
            self._generate_topics_over_time(
                topic_model, topics, docs, metadata, embedding_view, product_name
            )
        
        # Gera scatter plots PCA
        if settings_v3.TOPICS_GENERATE_PLOTS:
            self._generate_pca_scatter_plots(
                embeddings, topics, metadata, embedding_view, product_name
            )
        
        log.info(f"Análise de tópicos concluída para '{product_name}': {stats['n_topics']} tópicos")
        
        return {
            "status": "success",
            "product_name": product_name,
            "embedding_view": embedding_view,
            "stats": stats,
            "output_paths": output_paths
        }
    
    def analyze_all_products(
        self,
        embedding_view: str = "full"
    ) -> Dict[str, Dict]:
        """
        Analisa tópicos para todos os produtos
        
        Args:
            embedding_view: Visão de embedding ('full', 'agent', 'client')
            
        Returns:
            Dict mapeando produto -> resultado da análise
        """
        log.info(f"Iniciando análise de tópicos para todos os produtos - visão: {embedding_view}")
        
        # Busca produtos elegíveis
        products = self.db.get_products(min_calls=settings_v3.MIN_CALLS_PER_PRODUCT)
        
        if not products:
            log.warning("Nenhum produto encontrado")
            return {}
        
        log.info(f"Encontrados {len(products)} produtos para análise")
        
        results = {}
        
        for product in products:
            product_name = product['product_name']
            
            try:
                result = self.analyze_topics_by_product(product_name, embedding_view)
                
                if result.get("status") == "success":
                    results[product_name] = result
                    log.info(f"✓ '{product_name}' concluído: {result['stats']['n_topics']} tópicos")
                elif result.get("status") == "no_data":
                    log.warning(f"⚠️  '{product_name}': sem dados suficientes")
                else:
                    log.error(f"❌ '{product_name}': erro na análise")
                    
            except Exception as e:
                log.error(f"❌ Erro ao processar produto '{product_name}': {e}", exc_info=True)
                results[product_name] = {"status": "error", "error": str(e)}
        
        log.info(f"Análise de todos os produtos concluída: {len(results)}/{len(products)} produtos processados")
        
        return results
    
    def _load_topic_data(
        self, 
        embedding_view: str,
        product_name: str = None
    ) -> Tuple[List[str], np.ndarray, List[Dict]]:
        """
        Carrega dados para análise de tópicos
        
        Args:
            embedding_view: Visão de embedding ('full', 'agent', 'client')
            product_name: Nome do produto (opcional, None = todos os produtos)
        
        Returns:
            (documentos, embeddings, metadata)
        """
        if product_name:
            log.info(f"Carregando dados para análise de tópicos - Produto: {product_name}...")
        else:
            log.info(f"Carregando dados para análise de tópicos (todos os produtos)...")
        
        # Mapeia visão para colunas
        emb_col = settings_v3.EMBEDDING_COLUMN_MAP[embedding_view]
        valid_col = settings_v3.EMBEDDING_VALID_MAP[embedding_view]
        
        # Monta query baseada no tipo de documentos
        if settings_v3.TOPICS_USE_FULL_CALL:
            # Filtro de produto (opcional)
            product_filter = "AND p.name = %(product_name)s" if product_name else ""
            
            # Modo: Uma conversa completa = um documento
            sql = f"""
            SELECT 
                ce2.call_id,
                CASE 
                    WHEN '{embedding_view}' = 'full' THEN ce2.full_text
                    WHEN '{embedding_view}' = 'agent' THEN ce2.agent_text
                    WHEN '{embedding_view}' = 'client' THEN ce2.client_text
                END as text,
                ce2.{emb_col}::text as embedding_text,
                co.outcome,
                p.name AS product_name,
                cr.recorded_at
            FROM public.call_embeddings_v2 ce2
            JOIN public.call_records cr ON cr.call_id = ce2.call_id
            JOIN public.deals d ON cr.deal_id = d.deal_id
            JOIN public.pipelines p ON d.pipeline = p.pipeline_id
            JOIN public.call_outcomes co ON co.deal_id = cr.deal_id
            WHERE ce2.{emb_col} IS NOT NULL
              AND ce2.{valid_col} = TRUE
              AND co.outcome IN ('ganha', 'perdida')
              {product_filter}
            ORDER BY cr.recorded_at DESC, ce2.call_id ASC
            LIMIT %(limit)s
            """
        else:
            # Modo: Enunciados individuais (não implementado ainda)
            log.warning("Modo de enunciados não implementado, usando ligações completas")
            return [], np.array([]), []
        
        # Parâmetros da query
        params = {
            'limit': settings_v3.TOPICS_MAX_DOCUMENTS
        }
        if product_name:
            params['product_name'] = product_name
        
        rows = self.db.execute_safe(
            sql, 
            params, 
            fetch=True
        )
        
        if not rows:
            log.warning("Nenhuma ligação encontrada com embeddings válidos")
            return [], np.array([]), []
        
        log.info(f"Encontradas {len(rows)} ligações, processando...")
        
        docs = []
        embeddings = []
        metadata = []
        
        for row in rows:
            text = row["text"] or ""
            
            # Valida texto
            if self._is_too_short(text):
                continue
            
            # Processa embedding
            emb = from_pgvector(row["embedding_text"]) if row["embedding_text"] else None
            if emb is None or emb.size == 0:
                continue
            
            docs.append(text)
            embeddings.append(l2_normalize(emb))
            metadata.append({
                "call_id": row["call_id"],
                "outcome": row["outcome"],
                "product_name": row["product_name"] or "Desconhecido",
                "recorded_at": row["recorded_at"]
            })
        
        if not embeddings:
            log.warning("Nenhum documento válido após processamento")
            return [], np.array([]), []
        
        embeddings_array = np.vstack(embeddings).astype(np.float32)
        
        log.info(f"✓ Carregados {len(docs)} documentos válidos")
        log.info(f"  • Shape embeddings: {embeddings_array.shape}")
        log.info(f"  • Média de caracteres: {sum(len(d) for d in docs) / len(docs):.0f}")
        
        return docs, embeddings_array, metadata
    
    def _is_too_short(
        self, 
        text: str, 
        min_words: int = 10, 
        min_chars: int = 50
    ) -> bool:
        """Verifica se texto é muito curto"""
        words = len(text.split())
        return (words < min_words) or (len(text) < min_chars)
    
    def _setup_bertopic(self, n_docs: int = None) -> BERTopic:
        """
        Configura modelo BERTopic
        
        Args:
            n_docs: Número de documentos (para ajustar parâmetros automaticamente)
        """
        log.info("Configurando modelo BERTopic...")
        
        # Fixa seeds para reprodutibilidade
        import random
        np.random.seed(42)
        random.seed(42)
        log.info("  • Seeds fixadas (numpy=42, random=42)")
        
        # Ajusta parâmetros baseado no número de documentos
        if n_docs:
            # UMAP n_neighbors não pode ser maior que n_docs
            umap_n_neighbors = min(settings_v3.TOPICS_UMAP_N_NEIGHBORS, n_docs - 1)
            umap_n_neighbors = max(2, umap_n_neighbors)  # Mínimo 2
            
            # HDBSCAN min_cluster_size - ajuste adaptativo para mais granularidade
            # Estratégia escalonada para diferentes tamanhos de dataset
            if n_docs < 1000:
                # Datasets pequenos: muito granular (0.8%)
                min_cluster_size = max(10, int(n_docs * 0.008))
                min_samples = max(5, int(min_cluster_size * 0.4))
            elif n_docs < 1500:
                # Datasets médios: granular (1.2%)
                min_cluster_size = max(15, int(n_docs * 0.012))
                min_samples = max(7, int(min_cluster_size * 0.45))
            elif n_docs < 2000:
                # Datasets médio-grandes: menos granular (1.5%)
                min_cluster_size = max(20, int(n_docs * 0.015))
                min_samples = max(10, int(min_cluster_size * 0.5))
            else:
                # Para datasets grandes, mantém lógica original
                min_cluster_size = min(settings_v3.TOPICS_MIN_CLUSTER_SIZE, max(2, n_docs // 10))
                min_samples = min(settings_v3.TOPICS_MIN_SAMPLES, max(1, min_cluster_size // 3))
            
            # CountVectorizer min_df adaptativo (alinhado com V2 - usa percentual)
            # Para poucos docs, usa valor absoluto baixo, senão usa percentual
            if n_docs < 100:
                min_df_value = 1  # Valor absoluto para datasets muito pequenos
            else:
                min_df_value = 0.01  # 1% dos documentos (alinhado com V2)
            
            # Determina nível de granularidade
            if n_docs < 1000:
                granularity = "muito granular (<1k)"
            elif n_docs < 1500:
                granularity = "granular (1k-1.5k)"
            elif n_docs < 2000:
                granularity = "médio (1.5k-2k)"
            else:
                granularity = "padrão (>2k)"
            
            log.info(f"  • Documentos: {n_docs}")
            log.info(f"  • UMAP n_neighbors: {umap_n_neighbors}")
            log.info(f"  • Min cluster size: {min_cluster_size} (adaptativo: {granularity})")
            log.info(f"  • Min samples: {min_samples}")
            log.info(f"  • Min DF: {min_df_value}")
        else:
            umap_n_neighbors = settings_v3.TOPICS_UMAP_N_NEIGHBORS
            min_cluster_size = settings_v3.TOPICS_MIN_CLUSTER_SIZE
            min_samples = settings_v3.TOPICS_MIN_SAMPLES
            min_df_value = 0.01  # Padrão: 1% (alinhado com V2)
        
        # UMAP para redução de dimensionalidade
        umap_model = UMAP(
            n_neighbors=umap_n_neighbors,
            n_components=10,  # Preserva mais informação dimensional (alinhado com V2)
            min_dist=0.05,  # Alinhado com V2 (era 0.0)
            metric='cosine',
            spread=1.0,
            random_state=42
        )
        
        # HDBSCAN para clustering
        hdbscan_model = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            prediction_data=True,
            cluster_selection_method='eom',
            metric='euclidean'
        )
        
        # CountVectorizer para extração de palavras (alinhado com V2)
        # Calcula max_df adaptativo baseado no tamanho do dataset
        if n_docs and n_docs < 100:
            max_df_value = 0.95
        elif n_docs and n_docs < 500:
            max_df_value = 0.90
        else:
            max_df_value = 0.75  # Para datasets grandes, mais restritivo
        
        # Calcula max_features adaptativo
        if n_docs:
            max_features_value = min(2000, n_docs * 15)
        else:
            max_features_value = 3000
        
        # CountVectorizer para extração de palavras (usa stopwords expandidas)
        # Nota: Inclui n-gramas completos nas stopwords para filtrá-los corretamente
        vectorizer = CountVectorizer(
            ngram_range=(1, 3),  # Trigramas (alinhado com V2)
            stop_words=list(settings_v3.TOPICS_STOPWORDS),
            min_df=min_df_value,
            max_df=max_df_value,
            max_features=max_features_value,
            lowercase=True,
            token_pattern=r'\b[a-záàâãéèêíïóôõöúçñ]+\b'  # Suporte a PT-BR (alinhado com V2)
        )
        
        # c-TF-IDF
        ctfidf = ClassTfidfTransformer(bm25_weighting=True)
        
        # BERTopic (sem embedding_model porque passamos embeddings pré-calculados)
        # Também removemos KeyBERTInspired pois ele precisa do embedding_model
        topic_model = BERTopic(
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer,
            ctfidf_model=ctfidf,
            calculate_probabilities=True,
            verbose=False,
            language="multilingual",
            min_topic_size=max(10, n_docs // 100) if n_docs else 10,  # Alinhado com V2
            top_n_words=15,  # Alinhado com V2
            nr_topics=None  # Deixa BERTopic decidir número ótimo de tópicos
        )
        
        log.info("✓ Modelo configurado")
        return topic_model
    
    def _compute_topic_stats(
        self, 
        topic_model: BERTopic, 
        topics: np.ndarray, 
        metadata: List[Dict],
        embedding_view: str
    ) -> Dict:
        """Calcula estatísticas dos tópicos"""
        
        if not isinstance(topics, np.ndarray):
            topics = np.array(topics)
        
        n_topics = len(set(topics)) - (1 if -1 in topics else 0)
        n_outliers = np.sum(topics == -1)
        n_docs = len(topics)
        
        # Estatísticas por outcome
        outcome_stats = {}
        for outcome in ["ganha", "perdida"]:
            mask = np.array([m["outcome"] == outcome for m in metadata])
            
            if np.any(mask):
                outcome_topics = topics[mask]
                outcome_stats[outcome] = {
                    "n_docs": int(np.sum(mask)),
                    "n_topics": len(set(outcome_topics)) - (1 if -1 in outcome_topics else 0),
                    "n_outliers": int(np.sum(outcome_topics == -1)),
                    "coverage": float(1 - np.sum(outcome_topics == -1) / len(outcome_topics))
                }
        
        # Estatísticas por produto
        products = set(m["product_name"] for m in metadata)
        product_stats = {}
        for product in products:
            mask = np.array([m["product_name"] == product for m in metadata])
            
            if np.any(mask):
                product_topics = topics[mask]
                product_stats[product] = {
                    "n_docs": int(np.sum(mask)),
                    "n_topics": len(set(product_topics)) - (1 if -1 in product_topics else 0)
                }
        
        return {
            "n_topics": n_topics,
            "n_docs": n_docs,
            "n_outliers": n_outliers,
            "coverage": float(1 - n_outliers / n_docs),
            "embedding_view": embedding_view,
            "outcome_stats": outcome_stats,
            "product_stats": product_stats
        }
    
    def _save_results(
        self,
        topic_model: BERTopic,
        topics: np.ndarray,
        probs: np.ndarray,
        docs: List[str],
        metadata: List[Dict],
        embedding_view: str,
        stats: Dict,
        product_name: str = None
    ) -> Dict:
        """Salva resultados da análise"""
        
        # Cria diretórios
        topics_dir = os.path.join(settings_v3.V3_OUTPUT_DIR, "topics")
        os.makedirs(topics_dir, exist_ok=True)
        
        output_paths = {}
        
        # Sufixo do produto (para análises por produto)
        product_suffix = f"_{product_name.replace(' ', '_')}" if product_name else ""
        
        # 1. Salva modelo BERTopic
        model_path = os.path.join(topics_dir, f"bertopic_model_{embedding_view}{product_suffix}_{self.timestamp}")
        topic_model.save(model_path)
        output_paths["model"] = model_path
        log.info(f"✓ Modelo salvo: {model_path}")
        
        # 2. Salva info dos tópicos
        topic_info = topic_model.get_topic_info()
        topic_info_path = os.path.join(
            topics_dir, 
            f"topic_info_{embedding_view}{product_suffix}_{self.timestamp}.csv"
        )
        topic_info.to_csv(topic_info_path, index=False)
        output_paths["topic_info"] = topic_info_path
        log.info(f"✓ Info de tópicos salva: {topic_info_path}")
        
        # 3. Salva documentos com atribuições de tópicos
        results_df = pd.DataFrame({
            "call_id": [m["call_id"] for m in metadata],
            "document": docs,
            "topic": topics,
            "outcome": [m["outcome"] for m in metadata],
            "product_name": [m["product_name"] for m in metadata],
            "recorded_at": [m["recorded_at"] for m in metadata]
        })
        
        # Adiciona probabilidades se disponível
        if probs is not None and len(probs) > 0:
            results_df["probability"] = [float(np.max(p)) if len(p) > 0 else 0.0 for p in probs]
        
        results_path = os.path.join(
            topics_dir,
            f"topic_assignments_{embedding_view}{product_suffix}_{self.timestamp}.csv"
        )
        results_df.to_csv(results_path, index=False)
        output_paths["assignments"] = results_path
        log.info(f"✓ Atribuições salvas: {results_path}")
        
        # 4. Salva estatísticas
        stats_path = os.path.join(
            topics_dir,
            f"topic_stats_{embedding_view}{product_suffix}_{self.timestamp}.json"
        )
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False, default=str)
        output_paths["stats"] = stats_path
        log.info(f"✓ Estatísticas salvas: {stats_path}")
        
        return output_paths
    
    def _generate_visualizations(
        self,
        topic_model: BERTopic,
        topics: np.ndarray,
        probs: np.ndarray,
        embeddings: np.ndarray,
        docs: List[str],
        metadata: List[Dict],
        embedding_view: str,
        product_name: str = None
    ):
        """Gera visualizações dos tópicos"""
        
        if not settings_v3.TOPICS_GENERATE_PLOTS:
            log.info("Geração de plots desabilitada")
            return
        
        topics_dir = os.path.join(settings_v3.V3_OUTPUT_DIR, "topics")
        os.makedirs(topics_dir, exist_ok=True)
        
        # Sufixo do produto
        product_suffix = f"_{product_name.replace(' ', '_')}" if product_name else ""
        
        log.info("Gerando visualizações...")
        
        success_count = 0
        
        # 1. Barchart - palavras mais importantes
        try:
            fig = topic_model.visualize_barchart(top_n_topics=15, n_words=10)
            path = os.path.join(
                topics_dir, 
                f"topics_barchart_{embedding_view}{product_suffix}_{self.timestamp}.html"
            )
            fig.write_html(path)
            fig.write_image(path.replace('.html', '.png'), width=1200, height=800)
            success_count += 1
            log.info(f"✓ Barchart: {path}")
        except Exception as e:
            log.warning(f"Erro ao gerar barchart: {e}")
        
        # 2. Hierarquia de tópicos
        try:
            fig = topic_model.visualize_hierarchy()
            path = os.path.join(
                topics_dir,
                f"topics_hierarchy_{embedding_view}{product_suffix}_{self.timestamp}.html"
            )
            fig.write_html(path)
            fig.write_image(path.replace('.html', '.png'), width=1200, height=800)
            success_count += 1
            log.info(f"✓ Hierarchy: {path}")
        except Exception as e:
            log.warning(f"Erro ao gerar hierarchy: {e}")
        
        # 3. Heatmap de similaridade
        try:
            fig = topic_model.visualize_heatmap()
            path = os.path.join(
                topics_dir,
                f"topics_heatmap_{embedding_view}{product_suffix}_{self.timestamp}.html"
            )
            fig.write_html(path)
            fig.write_image(path.replace('.html', '.png'), width=1200, height=800)
            success_count += 1
            log.info(f"✓ Heatmap: {path}")
        except Exception as e:
            log.warning(f"Erro ao gerar heatmap: {e}")
        
        # 4. Visualização customizada por outcome (barchart)
        try:
            import plotly.express as px
            
            # Prepara dados
            df = pd.DataFrame({
                "topic": topics,
                "outcome": [m["outcome"] for m in metadata],
                "product": [m["product_name"] for m in metadata]
            })
            
            # Filtra outliers
            df = df[df["topic"] != -1]
            
            if len(df) > 0:
                # Conta por tópico e outcome
                counts = df.groupby(["topic", "outcome"]).size().reset_index(name="count")
                
                # Adiciona labels dos tópicos
                topic_labels = {}
                for topic_id in counts["topic"].unique():
                    words = topic_model.get_topic(int(topic_id))
                    if words:
                        label = f"Topic {topic_id}: {', '.join([w for w, _ in words[:3]])}"
                        topic_labels[topic_id] = label
                
                counts["topic_label"] = counts["topic"].map(topic_labels)
                
                # Cria gráfico
                fig = px.bar(
                    counts,
                    x="topic_label",
                    y="count",
                    color="outcome",
                    title=f"Distribuição de Tópicos por Outcome - {embedding_view}",
                    labels={"topic_label": "Tópico", "count": "Número de Documentos"},
                    barmode="group",
                    color_discrete_map={"ganha": "#2ecc71", "perdida": "#e74c3c"}
                )
                
                fig.update_layout(
                    xaxis_tickangle=-45,
                    height=600,
                    showlegend=True
                )
                
                path = os.path.join(
                    topics_dir,
                    f"topics_by_outcome_{embedding_view}{product_suffix}_{self.timestamp}.html"
                )
                fig.write_html(path)
                fig.write_image(path.replace('.html', '.png'), width=1200, height=800)
                success_count += 1
                log.info(f"✓ Topics by outcome: {path}")
        except Exception as e:
            log.warning(f"Erro ao gerar visualização por outcome: {e}")
        
        # 5. Visualização UMAP com tópicos e outcome (alpha para perdidas)
        try:
            import plotly.graph_objects as go
            
            # Pega coordenadas UMAP do modelo
            if hasattr(topic_model, 'umap_model') and topic_model.umap_model is not None:
                # Transforma embeddings com UMAP
                umap_embeddings = topic_model.umap_model.transform(embeddings)
                
                # Prepara dados
                df_viz = pd.DataFrame({
                    "x": umap_embeddings[:, 0],
                    "y": umap_embeddings[:, 1],
                    "topic": topics,
                    "outcome": [m["outcome"] for m in metadata],
                    "product": [m["product_name"] for m in metadata],
                    "call_id": [m["call_id"] for m in metadata]
                })
                
                # Filtra outliers para visualização
                df_viz = df_viz[df_viz["topic"] != -1]
                
                if len(df_viz) > 0:
                    # Mapa de cores por tópico
                    import plotly.colors as pc
                    n_topics = len(df_viz["topic"].unique())
                    colors = pc.qualitative.Plotly if n_topics <= 10 else pc.qualitative.Dark24
                    
                    topic_color_map = {}
                    for i, topic_id in enumerate(sorted(df_viz["topic"].unique())):
                        topic_color_map[topic_id] = colors[i % len(colors)]
                    
                    # Adiciona labels dos tópicos
                    topic_labels = {}
                    for topic_id in df_viz["topic"].unique():
                        words = topic_model.get_topic(int(topic_id))
                        if words:
                            label = f"Topic {topic_id}: {', '.join([w for w, _ in words[:3]])}"
                            topic_labels[topic_id] = label
                        else:
                            topic_labels[topic_id] = f"Topic {topic_id}"
                    
                    df_viz["topic_label"] = df_viz["topic"].map(topic_labels)
                    
                    # Converte cor hex para rgba
                    def hex_to_rgba(hex_color, alpha):
                        hex_color = hex_color.lstrip('#')
                        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                        return f'rgba({r},{g},{b},{alpha})'
                    
                    # Cria figura
                    fig = go.Figure()
                    
                    # Adiciona pontos por tópico
                    for topic_id in sorted(df_viz["topic"].unique()):
                        df_topic = df_viz[df_viz["topic"] == topic_id]
                        color_base = topic_color_map[topic_id]
                        
                        # Separa ganhas e perdidas
                        df_ganha = df_topic[df_topic["outcome"] == "ganha"]
                        df_perdida = df_topic[df_topic["outcome"] == "perdida"]
                        
                        # Pontos GANHA (opacidade 100%)
                        if len(df_ganha) > 0:
                            fig.add_trace(go.Scatter(
                                x=df_ganha["x"],
                                y=df_ganha["y"],
                                mode='markers',
                                name=f'{topic_labels[topic_id]} (Ganha)',
                                marker=dict(
                                    size=8,
                                    color=color_base,
                                    opacity=1.0,
                                    line=dict(width=1, color='white')
                                ),
                                text=[f"Call: {cid}<br>Outcome: Ganha<br>Produto: {prod}" 
                                      for cid, prod in zip(df_ganha["call_id"], df_ganha["product"])],
                                hovertemplate='<b>%{text}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>',
                                showlegend=True
                            ))
                        
                        # Pontos PERDIDA (opacidade 50%)
                        if len(df_perdida) > 0:
                            fig.add_trace(go.Scatter(
                                x=df_perdida["x"],
                                y=df_perdida["y"],
                                mode='markers',
                                name=f'{topic_labels[topic_id]} (Perdida)',
                                marker=dict(
                                    size=8,
                                    color=color_base,
                                    opacity=0.5,  # 50% de transparência
                                    line=dict(width=1, color='white')
                                ),
                                text=[f"Call: {cid}<br>Outcome: Perdida<br>Produto: {prod}" 
                                      for cid, prod in zip(df_perdida["call_id"], df_perdida["product"])],
                                hovertemplate='<b>%{text}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>',
                                showlegend=True,
                                visible='legendonly'  # Começa oculto na legenda
                            ))
                    
                    fig.update_layout(
                        title=f'Tópicos no Espaço UMAP - {embedding_view.upper()}<br>'
                              f'<sub>Opacidade 100% = Ganha | Opacidade 50% = Perdida</sub>',
                        xaxis_title='UMAP Dimensão 1',
                        yaxis_title='UMAP Dimensão 2',
                        height=700,
                        width=1000,
                        hovermode='closest',
                        legend=dict(
                            title="Tópicos e Outcomes",
                            yanchor="top",
                            y=0.99,
                            xanchor="left",
                            x=1.02
                        )
                    )
                    
                    path = os.path.join(
                        topics_dir,
                        f"topics_umap_alpha_{embedding_view}{product_suffix}_{self.timestamp}.html"
                    )
                    fig.write_html(path)
                    fig.write_image(path.replace('.html', '.png'), width=1200, height=800)
                    success_count += 1
                    log.info(f"✓ Topics UMAP com alpha: {path}")
        except Exception as e:
            log.warning(f"Erro ao gerar UMAP com alpha: {e}", exc_info=True)
        
        log.info(f"✓ {success_count} visualizações geradas")
    
    def _generate_wordclouds(
        self,
        topic_model: BERTopic,
        topics: np.ndarray,
        metadata: List[Dict],
        embedding_view: str,
        product_name: str = None
    ):
        """
        Gera word clouds para cada tópico, separadas por outcome
        
        Args:
            topic_model: Modelo BERTopic treinado
            topics: Array com atribuições de tópicos
            metadata: Lista de metadados dos documentos
            embedding_view: Visão de embedding utilizada
            product_name: Nome do produto (opcional)
        """
        if not HAS_WORDCLOUD:
            log.warning("WordCloud não disponível. Execute: pip install wordcloud")
            return
        
        try:
            import matplotlib.pyplot as plt
            from matplotlib import cm
            import matplotlib.colors as mcolors
        except ImportError:
            log.warning("Matplotlib não disponível")
            return
        
        log.info(f"\n🎨 Gerando word clouds para tópicos...")
        
        # Sufixo do produto
        product_suffix = f"_{product_name.replace(' ', '_')}" if product_name else ""
        
        # Cria diretório para word clouds
        topics_dir = os.path.join(settings_v3.V3_TOPICS_DIR)
        wordcloud_dir = os.path.join(topics_dir, f"wordclouds_{embedding_view}{product_suffix}_{self.timestamp}")
        os.makedirs(wordcloud_dir, exist_ok=True)
        
        # Pega informações dos tópicos
        topic_info = topic_model.get_topic_info()
        valid_topics = [t for t in topic_info['Topic'].values if t != -1]
        
        if not valid_topics:
            log.warning("Nenhum tópico válido encontrado")
            return
        
        # DataFrame com metadata
        df = pd.DataFrame({
            'topic': topics,
            'outcome': [m['outcome'] for m in metadata],
            'product_name': [m['product_name'] for m in metadata]
        })
        
        # Define cores profissionais
        color_schemes = {
            'ganha': {
                'colormap': 'Greens',
                'bg_color': 'white',
                'title_color': '#27ae60',
                'label': 'Vendas Ganhas'
            },
            'perdida': {
                'colormap': 'Reds',
                'bg_color': 'white',
                'title_color': '#e74c3c',
                'label': 'Vendas Perdidas'
            }
        }
        
        wordcloud_count = 0
        
        # Gera word cloud para cada tópico e outcome
        for topic_id in valid_topics:
            try:
                # Pega palavras e pesos do tópico
                topic_words = topic_model.get_topic(int(topic_id))
                
                if not topic_words:
                    continue
                
                # Converte para dicionário {palavra: peso}
                word_weights = {word: float(weight) for word, weight in topic_words}
                
                # Conta documentos neste tópico
                topic_docs = df[df['topic'] == topic_id]
                n_total = len(topic_docs)
                
                if n_total == 0:
                    continue
                
                # Gera word cloud para cada outcome
                for outcome in ['ganha', 'perdida']:
                    outcome_docs = topic_docs[topic_docs['outcome'] == outcome]
                    n_outcome = len(outcome_docs)
                    
                    if n_outcome < 3:  # Mínimo de documentos
                        continue
                    
                    # Configura word cloud
                    scheme = color_schemes[outcome]
                    
                    wc = WordCloud(
                        width=1200,
                        height=800,
                        background_color=scheme['bg_color'],
                        colormap=scheme['colormap'],
                        relative_scaling=0.5,
                        min_font_size=10,
                        max_words=50,
                        prefer_horizontal=0.7,
                        collocations=False
                    ).generate_from_frequencies(word_weights)
                    
                    # Cria figura
                    fig, ax = plt.subplots(figsize=(15, 10))
                    ax.imshow(wc, interpolation='bilinear')
                    ax.axis('off')
                    
                    # Título informativo
                    pct = (n_outcome / n_total) * 100
                    topic_name = topic_info[topic_info['Topic'] == topic_id]['Name'].values[0]
                    
                    title = (
                        f'Tópico {topic_id}: {scheme["label"]}\n'
                        f'{topic_name}\n'
                        f'{n_outcome} chamadas ({pct:.1f}% do tópico) | Visão: {embedding_view.upper()}'
                    )
                    
                    plt.title(
                        title,
                        fontsize=16,
                        fontweight='bold',
                        color=scheme['title_color'],
                        pad=20
                    )
                    
                    # Salva
                    filename = f"topic_{topic_id:02d}_{outcome}_{embedding_view}.png"
                    filepath = os.path.join(wordcloud_dir, filename)
                    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
                    plt.close()
                    
                    wordcloud_count += 1
                    log.info(f"  ✓ Tópico {topic_id} ({outcome}): {filename}")
            
            except Exception as e:
                log.warning(f"  ⚠ Erro ao gerar word cloud para tópico {topic_id}: {e}")
                continue
        
        # Gera índice HTML para navegação
        self._generate_wordcloud_index(
            wordcloud_dir, 
            topic_info, 
            df, 
            embedding_view,
            color_schemes
        )
        
        log.info(f"✓ {wordcloud_count} word clouds geradas em: {wordcloud_dir}")
    
    def _generate_wordcloud_index(
        self,
        wordcloud_dir: str,
        topic_info: pd.DataFrame,
        df: pd.DataFrame,
        embedding_view: str,
        color_schemes: Dict
    ):
        """
        Gera página HTML de índice para navegação das word clouds
        
        Args:
            wordcloud_dir: Diretório com as word clouds
            topic_info: DataFrame com informações dos tópicos
            df: DataFrame com atribuições
            embedding_view: Visão de embedding
            color_schemes: Esquemas de cores
        """
        try:
            # Lista arquivos PNG
            import glob
            wordcloud_files = sorted(glob.glob(os.path.join(wordcloud_dir, "*.png")))
            
            if not wordcloud_files:
                return
            
            # Agrupa por tópico
            topics_dict = {}
            for filepath in wordcloud_files:
                filename = os.path.basename(filepath)
                # Parse: topic_XX_outcome_view.png
                parts = filename.replace('.png', '').split('_')
                if len(parts) >= 3:
                    topic_id = int(parts[1])
                    outcome = parts[2]
                    
                    if topic_id not in topics_dict:
                        topics_dict[topic_id] = {}
                    
                    topics_dict[topic_id][outcome] = filename
            
            # Gera HTML
            html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Word Clouds - Análise de Tópicos ({embedding_view.upper()})</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 30px;
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            padding: 40px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }}
        
        h1 {{
            text-align: center;
            color: #2c3e50;
            margin-bottom: 10px;
            font-size: 2.5em;
        }}
        
        .subtitle {{
            text-align: center;
            color: #7f8c8d;
            margin-bottom: 40px;
            font-size: 1.2em;
        }}
        
        .topic-section {{
            margin-bottom: 50px;
            border: 2px solid #ecf0f1;
            border-radius: 10px;
            padding: 25px;
            background: #fafafa;
        }}
        
        .topic-header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }}
        
        .topic-header h2 {{
            margin: 0;
            font-size: 1.5em;
        }}
        
        .topic-stats {{
            font-size: 0.9em;
            margin-top: 5px;
            opacity: 0.9;
        }}
        
        .wordcloud-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 30px;
            margin-top: 20px;
        }}
        
        .wordcloud-card {{
            background: white;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }}
        
        .wordcloud-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        }}
        
        .wordcloud-card.ganha {{
            border-top: 4px solid #27ae60;
        }}
        
        .wordcloud-card.perdida {{
            border-top: 4px solid #e74c3c;
        }}
        
        .card-header {{
            padding: 15px 20px;
            font-weight: bold;
            font-size: 1.1em;
        }}
        
        .card-header.ganha {{
            background: #d5f4e6;
            color: #27ae60;
        }}
        
        .card-header.perdida {{
            background: #fadbd8;
            color: #e74c3c;
        }}
        
        .card-image {{
            width: 100%;
            height: auto;
            display: block;
            cursor: pointer;
        }}
        
        .card-footer {{
            padding: 12px 20px;
            background: #f8f9fa;
            font-size: 0.9em;
            color: #7f8c8d;
        }}
        
        .legend {{
            display: flex;
            justify-content: center;
            gap: 30px;
            margin: 30px 0;
            padding: 20px;
            background: #ecf0f1;
            border-radius: 10px;
        }}
        
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 10px;
            font-weight: 500;
        }}
        
        .legend-color {{
            width: 30px;
            height: 30px;
            border-radius: 5px;
        }}
        
        .legend-color.ganha {{
            background: #27ae60;
        }}
        
        .legend-color.perdida {{
            background: #e74c3c;
        }}
        
        .modal {{
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.9);
            cursor: pointer;
        }}
        
        .modal img {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            max-width: 95%;
            max-height: 95%;
            border-radius: 10px;
        }}
        
        .close-modal {{
            position: absolute;
            top: 20px;
            right: 40px;
            color: white;
            font-size: 40px;
            font-weight: bold;
            cursor: pointer;
        }}
        
        @media (max-width: 768px) {{
            .wordcloud-grid {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎨 Word Clouds - Análise de Tópicos</h1>
        <p class="subtitle">Visão: {embedding_view.upper()} | {len(topics_dict)} tópicos identificados</p>
        
        <div class="legend">
            <div class="legend-item">
                <div class="legend-color ganha"></div>
                <span>Vendas Ganhas</span>
            </div>
            <div class="legend-item">
                <div class="legend-color perdida"></div>
                <span>Vendas Perdidas</span>
            </div>
        </div>
"""
            
            # Adiciona seções para cada tópico
            for topic_id in sorted(topics_dict.keys()):
                topic_data = topic_info[topic_info['Topic'] == topic_id]
                if len(topic_data) == 0:
                    continue
                
                topic_name = topic_data['Name'].values[0]
                topic_count = topic_data['Count'].values[0]
                
                # Conta por outcome no DataFrame real (usado para word clouds)
                topic_df = df[df['topic'] == topic_id]
                n_ganha = len(topic_df[topic_df['outcome'] == 'ganha'])
                n_perdida = len(topic_df[topic_df['outcome'] == 'perdida'])
                n_total_real = n_ganha + n_perdida  # Total real de docs neste tópico
                
                # Usa o total real para calcular percentuais corretos
                pct_ganha = (100 * n_ganha / n_total_real) if n_total_real > 0 else 0
                pct_perdida = (100 * n_perdida / n_total_real) if n_total_real > 0 else 0
                
                html += f"""
        <div class="topic-section">
            <div class="topic-header">
                <h2>Tópico {topic_id}: {topic_name}</h2>
                <div class="topic-stats">
                    Total: {n_total_real} documentos | 
                    Ganhas: {n_ganha} ({pct_ganha:.1f}%) | 
                    Perdidas: {n_perdida} ({pct_perdida:.1f}%)
                </div>
            </div>
            
            <div class="wordcloud-grid">
"""
                
                for outcome in ['ganha', 'perdida']:
                    if outcome in topics_dict[topic_id]:
                        filename = topics_dict[topic_id][outcome]
                        n_docs = n_ganha if outcome == 'ganha' else n_perdida
                        label = color_schemes[outcome]['label']
                        
                        html += f"""
                <div class="wordcloud-card {outcome}">
                    <div class="card-header {outcome}">{label}</div>
                    <img src="{filename}" alt="Word Cloud" class="card-image" onclick="openModal('{filename}')">
                    <div class="card-footer">{n_docs} chamadas neste tópico</div>
                </div>
"""
                
                html += """
            </div>
        </div>
"""
            
            html += """
    </div>
    
    <div id="imageModal" class="modal" onclick="closeModal()">
        <span class="close-modal">&times;</span>
        <img id="modalImage" src="" alt="">
    </div>
    
    <script>
        function openModal(src) {
            document.getElementById('imageModal').style.display = 'block';
            document.getElementById('modalImage').src = src;
        }
        
        function closeModal() {
            document.getElementById('imageModal').style.display = 'none';
        }
        
        document.addEventListener('keydown', function(e) {
            if (e.key === 'Escape') {
                closeModal();
            }
        });
    </script>
</body>
</html>
"""
            
            # Salva HTML
            index_path = os.path.join(wordcloud_dir, "index.html")
            with open(index_path, 'w', encoding='utf-8') as f:
                f.write(html)
            
            log.info(f"✓ Índice HTML gerado: {index_path}")
            
        except Exception as e:
            log.error(f"Erro ao gerar índice HTML: {e}")
    
    def _generate_topics_over_time(
        self,
        topic_model: BERTopic,
        topics: np.ndarray,
        docs: List[str],
        metadata: List[Dict],
        embedding_view: str,
        product_name: str = None
    ):
        """
        Gera análises temporais dos tópicos
        
        Args:
            topic_model: Modelo BERTopic treinado
            topics: Array com atribuições de tópicos
            docs: Lista de documentos
            metadata: Lista de metadados dos documentos
            embedding_view: Visão de embedding utilizada
        """
        log.info(f"\n⏰ Gerando análises temporais de tópicos...")
        
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            import pandas as pd
            from datetime import datetime as dt
        except ImportError as e:
            log.warning(f"Erro ao importar bibliotecas para análise temporal: {e}")
            return
        
        # Sufixo do produto
        product_suffix = f"_{product_name.replace(' ', '_')}" if product_name else ""
        
        # Cria diretório
        topics_dir = settings_v3.V3_TOPICS_DIR
        
        # Prepara dados (SEM filtrar outliers inicialmente)
        df_full = pd.DataFrame({
            'topic': topics,
            'document': docs,
            'outcome': [m['outcome'] for m in metadata],
            'product_name': [m['product_name'] for m in metadata],
            'recorded_at': [m['recorded_at'] for m in metadata]
        })
        
        # Converte timestamps no DataFrame completo
        df_full['timestamp'] = pd.to_datetime(df_full['recorded_at'])
        df_full['date'] = df_full['timestamp'].dt.date
        df_full['weekday'] = df_full['timestamp'].dt.day_name()
        df_full['hour'] = df_full['timestamp'].dt.hour
        df_full['month'] = df_full['timestamp'].dt.to_period('M').astype(str)
        
        # Info dos tópicos
        topic_info = topic_model.get_topic_info()
        topic_labels = {row['Topic']: row['Name'] for _, row in topic_info.iterrows() if row['Topic'] != -1}
        df_full['topic_label'] = df_full['topic'].map(topic_labels)
        
        # DataFrame filtrado (sem outliers) para visualizações 2, 3, 4
        df = df_full[df_full['topic'] != -1].copy()
        
        if len(df) == 0:
            log.warning("Nenhum documento para análise temporal (todos outliers)")
            return
        
        viz_count = 0
        
        # 1. EVOLUÇÃO CALENDÁRIO (método nativo BERTopic)
        try:
            log.info("  1. Evolução calendário (nativo BERTopic)...")
            
            # Usa DataFrame COMPLETO (com outliers) porque BERTopic espera todos os docs
            timestamps_all = df_full['timestamp'].tolist()
            docs_all = df_full['document'].tolist()
            topics_all = df_full['topic'].tolist()
            
            # Topics over time (nativo)
            topics_over_time = topic_model.topics_over_time(
                docs_all, 
                timestamps_all,
                nr_bins=settings_v3.TOPICS_OVERTIME_BINS
            )
            
            # Visualização
            fig = topic_model.visualize_topics_over_time(
                topics_over_time,
                top_n_topics=10,
                width=1400,
                height=600
            )
            
            # Customiza título
            fig.update_layout(
                title=f'📈 Evolução de Tópicos ao Longo do Tempo - {embedding_view.upper()}'
            )
            
            path = os.path.join(
                topics_dir,
                f"topics_overtime_calendar_{embedding_view}{product_suffix}_{self.timestamp}.html"
            )
            fig.write_html(path)
            fig.write_image(path.replace('.html', '.png'), width=1200, height=800)
            viz_count += 1
            log.info(f"  ✓ Evolução calendário: {path}")
            
        except Exception as e:
            log.warning(f"  ⚠ Erro ao gerar evolução calendário: {e}")
        
        # 2. SAZONALIDADE - DIA DA SEMANA
        try:
            log.info("  2. Sazonalidade por dia da semana...")
            
            # Agrupa por dia da semana e tópico
            weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            weekday_pt = {
                'Monday': 'Segunda', 'Tuesday': 'Terça', 'Wednesday': 'Quarta',
                'Thursday': 'Quinta', 'Friday': 'Sexta', 'Saturday': 'Sábado', 'Sunday': 'Domingo'
            }
            
            weekday_counts = df.groupby(['weekday', 'topic_label']).size().reset_index(name='count')
            
            # Calcula proporção por dia
            total_by_weekday = df.groupby('weekday').size()
            weekday_counts['proportion'] = weekday_counts.apply(
                lambda row: row['count'] / total_by_weekday[row['weekday']] if row['weekday'] in total_by_weekday.index else 0,
                axis=1
            )
            
            # Top 10 tópicos
            top_topics = df['topic_label'].value_counts().head(10).index
            weekday_counts = weekday_counts[weekday_counts['topic_label'].isin(top_topics)]
            
            if len(weekday_counts) > 0:
                fig = go.Figure()
                
                for topic in top_topics:
                    topic_data = weekday_counts[weekday_counts['topic_label'] == topic]
                    
                    # Ordena por dia da semana
                    topic_data['weekday_order'] = topic_data['weekday'].map(
                        {day: i for i, day in enumerate(weekday_order)}
                    )
                    topic_data = topic_data.sort_values('weekday_order')
                    
                    # Traduz nomes
                    x_labels = [weekday_pt.get(day, day) for day in topic_data['weekday']]
                    
                    fig.add_trace(go.Scatter(
                        x=x_labels,
                        y=topic_data['proportion'] * 100,
                        mode='lines+markers',
                        name=topic[:50],  # Trunca nome longo
                        line=dict(width=2),
                        marker=dict(size=8)
                    ))
                
                fig.update_layout(
                    title=f'📅 Sazonalidade: Tópicos por Dia da Semana - {embedding_view.upper()}',
                    xaxis_title='Dia da Semana',
                    yaxis_title='Proporção (%)',
                    height=600,
                    width=1400,
                    hovermode='x unified',
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="right",
                        x=0.99
                    )
                )
                
                path = os.path.join(
                    topics_dir,
                    f"topics_overtime_weekday_{embedding_view}{product_suffix}_{self.timestamp}.html"
                )
                fig.write_html(path)
                fig.write_image(path.replace('.html', '.png'), width=1200, height=800)
                viz_count += 1
                log.info(f"  ✓ Sazonalidade (weekday): {path}")
        
        except Exception as e:
            log.warning(f"  ⚠ Erro ao gerar sazonalidade: {e}")
        
        # 3. ANÁLISE POR MÊS
        try:
            log.info("  3. Evolução mensal...")
            
            # Agrupa por mês e tópico
            monthly_counts = df.groupby(['month', 'topic_label']).size().reset_index(name='count')
            
            # Top 10 tópicos
            top_topics = df['topic_label'].value_counts().head(10).index
            monthly_counts = monthly_counts[monthly_counts['topic_label'].isin(top_topics)]
            
            if len(monthly_counts) > 0:
                fig = go.Figure()
                
                for topic in top_topics:
                    topic_data = monthly_counts[monthly_counts['topic_label'] == topic].sort_values('month')
                    
                    fig.add_trace(go.Bar(
                        x=topic_data['month'],
                        y=topic_data['count'],
                        name=topic[:50],
                        hovertemplate='<b>%{fullData.name}</b><br>Mês: %{x}<br>Documentos: %{y}<extra></extra>'
                    ))
                
                fig.update_layout(
                    title=f'📊 Evolução Mensal de Tópicos - {embedding_view.upper()}',
                    xaxis_title='Mês',
                    yaxis_title='Número de Documentos',
                    height=600,
                    width=1400,
                    barmode='stack',
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="right",
                        x=0.99
                    )
                )
                
                path = os.path.join(
                    topics_dir,
                    f"topics_overtime_monthly_{embedding_view}{product_suffix}_{self.timestamp}.html"
                )
                fig.write_html(path)
                fig.write_image(path.replace('.html', '.png'), width=1200, height=800)
                viz_count += 1
                log.info(f"  ✓ Evolução mensal: {path}")
        
        except Exception as e:
            log.warning(f"  ⚠ Erro ao gerar evolução mensal: {e}")
        
        # 4. COMPARAÇÃO: GANHAS vs PERDIDAS AO LONGO DO TEMPO
        try:
            log.info("  4. Comparação temporal: ganhas vs perdidas...")
            
            # Agrupa por data e outcome
            daily_outcome = df.groupby(['date', 'outcome', 'topic_label']).size().reset_index(name='count')
            
            # Top 5 tópicos
            top_topics = df['topic_label'].value_counts().head(5).index
            daily_outcome = daily_outcome[daily_outcome['topic_label'].isin(top_topics)]
            
            if len(daily_outcome) > 0:
                # Cria subplots para cada tópico
                n_topics = len(top_topics)
                fig = make_subplots(
                    rows=n_topics, 
                    cols=1,
                    subplot_titles=[f"{topic[:60]}" for topic in top_topics],
                    vertical_spacing=0.08
                )
                
                for idx, topic in enumerate(top_topics, 1):
                    topic_data = daily_outcome[daily_outcome['topic_label'] == topic]
                    
                    for outcome in ['ganha', 'perdida']:
                        outcome_data = topic_data[topic_data['outcome'] == outcome].sort_values('date')
                        
                        if len(outcome_data) > 0:
                            fig.add_trace(
                                go.Scatter(
                                    x=outcome_data['date'],
                                    y=outcome_data['count'],
                                    mode='lines+markers',
                                    name=outcome.capitalize() if idx == 1 else None,
                                    line=dict(
                                        color='#2ecc71' if outcome == 'ganha' else '#e74c3c',
                                        width=2
                                    ),
                                    marker=dict(size=6),
                                    showlegend=(idx == 1),
                                    legendgroup=outcome
                                ),
                                row=idx,
                                col=1
                            )
                
                fig.update_layout(
                    title=f'🎯 Tópicos: Ganhas vs Perdidas ao Longo do Tempo - {embedding_view.upper()}',
                    height=300 * n_topics,
                    width=1400,
                    hovermode='x unified'
                )
                
                # Atualiza eixos
                for idx in range(1, n_topics + 1):
                    fig.update_xaxes(title_text="Data" if idx == n_topics else "", row=idx, col=1)
                    fig.update_yaxes(title_text="Docs", row=idx, col=1)
                
                path = os.path.join(
                    topics_dir,
                    f"topics_overtime_outcome_{embedding_view}{product_suffix}_{self.timestamp}.html"
                )
                fig.write_html(path)
                fig.write_image(path.replace('.html', '.png'), width=1200, height=800)
                viz_count += 1
                log.info(f"  ✓ Comparação ganhas/perdidas: {path}")
        
        except Exception as e:
            log.warning(f"  ⚠ Erro ao gerar comparação temporal: {e}")
        
        log.info(f"\n✓ {viz_count} visualizações temporais geradas")
    
    def _generate_pca_scatter_plots(
        self,
        embeddings: np.ndarray,
        topics: np.ndarray,
        metadata: List[Dict],
        embedding_view: str,
        product_name: str = None
    ):
        """
        Gera scatter plots 2D de embeddings reduzidos por PCA
        
        Args:
            embeddings: Array de embeddings (n_samples, 1024)
            topics: Array com atribuições de tópicos
            metadata: Lista de metadados dos documentos
            embedding_view: Visão de embedding utilizada
        """
        log.info(f"\n📊 Gerando scatter plots PCA...")
        
        try:
            from sklearn.decomposition import PCA
            import matplotlib.pyplot as plt
            from matplotlib.patches import Ellipse
            from scipy.stats import chi2
            import plotly.express as px
            import plotly.graph_objects as go
        except ImportError as e:
            log.warning(f"Erro ao importar bibliotecas para PCA scatter: {e}")
            return
        
        # Sufixo do produto
        product_suffix = f"_{product_name.replace(' ', '_')}" if product_name else ""
        
        topics_dir = settings_v3.V3_TOPICS_DIR
        
        # Remove outliers para visualização mais limpa
        topics_array = np.array(topics)
        mask = topics_array != -1
        embeddings_filtered = embeddings[mask]
        topics_filtered = topics_array[mask]
        metadata_filtered = [m for i, m in enumerate(metadata) if mask[i]]
        
        if len(embeddings_filtered) < 10:
            log.warning("Poucos documentos para PCA scatter (após remover outliers)")
            return
        
        # PCA
        log.info("  Aplicando PCA (2 componentes)...")
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(embeddings_filtered)
        
        var1, var2 = pca.explained_variance_ratio_
        log.info(f"  Variância explicada: PC1={var1:.1%}, PC2={var2:.1%}")
        
        # Extrai metadados
        outcomes = np.array([m['outcome'] for m in metadata_filtered])
        products = [m['product_name'] for m in metadata_filtered]
        call_ids = [m.get('call_id', i) for i, m in enumerate(metadata_filtered)]
        
        # Duração (se disponível, senão usa placeholder)
        try:
            durations = np.array([m.get('duration', 60.0) for m in metadata_filtered])
        except:
            durations = np.ones(len(metadata_filtered)) * 60.0
        
        # Normaliza duração para tamanho dos pontos (20-200)
        if durations.max() > durations.min():
            sizes = 20 + (durations - durations.min()) / (durations.max() - durations.min()) * 180
        else:
            sizes = np.ones(len(durations)) * 50
        
        viz_count = 0
        
        # 1. SCATTER ESTÁTICO: Colorido por Tópico
        try:
            log.info("  1. Scatter estático (por tópico)...")
            
            fig, ax = plt.subplots(figsize=(14, 10))
            
            # Scatter principal
            scatter = ax.scatter(
                X_pca[:, 0], X_pca[:, 1],
                c=topics_filtered,
                s=sizes,
                alpha=0.6,
                cmap='tab20',
                edgecolors='white',
                linewidth=0.5
            )
            
            # Elipses de confiança para cada tópico
            unique_topics = np.unique(topics_filtered)
            for topic_id in unique_topics:
                mask_topic = (topics_filtered == topic_id)
                if np.sum(mask_topic) < 3:
                    continue
                
                points = X_pca[mask_topic]
                mean = points.mean(axis=0)
                cov = np.cov(points.T)
                
                try:
                    eigenvalues, eigenvectors = np.linalg.eig(cov)
                    if np.all(eigenvalues > 0):
                        angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
                        chi2_val = chi2.ppf(0.95, 2)
                        width, height = 2 * np.sqrt(chi2_val * eigenvalues)
                        
                        ellipse = Ellipse(
                            mean, width, height,
                            angle=np.degrees(angle),
                            alpha=0.15,
                            facecolor=plt.cm.tab20(topic_id % 20),
                            edgecolor='black',
                            linewidth=1,
                            linestyle='--'
                        )
                        ax.add_patch(ellipse)
                        
                        # Label do tópico
                        ax.text(mean[0], mean[1], f'T{topic_id}',
                               fontsize=10, weight='bold',
                               ha='center', va='center',
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black'))
                except:
                    pass
            
            # Eixos e labels
            ax.set_xlabel(f'Componente Principal 1 ({var1:.1%} da variância)', fontsize=12)
            ax.set_ylabel(f'Componente Principal 2 ({var2:.1%} da variância)', fontsize=12)
            ax.set_title(f'PCA: Embeddings por Tópico - {embedding_view.upper()}', fontsize=14, weight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # Colorbar
            plt.colorbar(scatter, ax=ax, label='Tópico ID', pad=0.02)
            
            # Anotação de tamanho
            ax.text(0.02, 0.98, '⚫ Tamanho do ponto = Duração da ligação',
                   transform=ax.transAxes, fontsize=10,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7, edgecolor='black'))
            
            plt.tight_layout()
            
            path = os.path.join(topics_dir, f"pca_scatter_by_topic_{embedding_view}_{self.timestamp}.png")
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.close()
            
            viz_count += 1
            log.info(f"  ✓ Scatter por tópico: {path}")
        
        except Exception as e:
            log.warning(f"  ⚠ Erro ao gerar scatter por tópico: {e}")
        
        # 2. SCATTER ESTÁTICO: Colorido por Outcome
        try:
            log.info("  2. Scatter estático (por outcome)...")
            
            fig, ax = plt.subplots(figsize=(14, 10))
            
            # Cores por outcome
            colors = np.where(outcomes == 'ganha', '#2ecc71', '#e74c3c')
            
            # Scatter principal
            for outcome, color, label in [('ganha', '#2ecc71', 'Venda Ganha'), 
                                          ('perdida', '#e74c3c', 'Venda Perdida')]:
                mask_outcome = (outcomes == outcome)
                ax.scatter(
                    X_pca[mask_outcome, 0], X_pca[mask_outcome, 1],
                    c=color,
                    s=sizes[mask_outcome],
                    alpha=0.6,
                    label=label,
                    edgecolors='white',
                    linewidth=0.5
                )
            
            # Eixos e labels
            ax.set_xlabel(f'Componente Principal 1 ({var1:.1%} da variância)', fontsize=12)
            ax.set_ylabel(f'Componente Principal 2 ({var2:.1%} da variância)', fontsize=12)
            ax.set_title(f'PCA: Embeddings por Outcome - {embedding_view.upper()}', fontsize=14, weight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
            
            # Anotação de tamanho
            ax.text(0.02, 0.98, '⚫ Tamanho do ponto = Duração da ligação',
                   transform=ax.transAxes, fontsize=10,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7, edgecolor='black'))
            
            plt.tight_layout()
            
            path = os.path.join(topics_dir, f"pca_scatter_by_outcome_{embedding_view}_{self.timestamp}.png")
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.close()
            
            viz_count += 1
            log.info(f"  ✓ Scatter por outcome: {path}")
        
        except Exception as e:
            log.warning(f"  ⚠ Erro ao gerar scatter por outcome: {e}")
        
        # 3. SCATTER INTERATIVO (Plotly)
        try:
            log.info("  3. Scatter interativo (Plotly)...")
            
            # Prepara DataFrame
            df = pd.DataFrame({
                'PC1': X_pca[:, 0],
                'PC2': X_pca[:, 1],
                'Tópico': topics_filtered.astype(str),
                'Outcome': outcomes,
                'Produto': products,
                'Call ID': call_ids,
                'Duração': durations,
                'Size': sizes
            })
            
            # Figura interativa
            fig = px.scatter(
                df,
                x='PC1',
                y='PC2',
                color='Tópico',
                size='Size',
                hover_data={
                    'Call ID': True,
                    'Outcome': True,
                    'Produto': True,
                    'Duração': ':.1f',
                    'PC1': ':.3f',
                    'PC2': ':.3f',
                    'Size': False
                },
                labels={
                    'PC1': f'Componente Principal 1 ({var1:.1%})',
                    'PC2': f'Componente Principal 2 ({var2:.1%})'
                },
                title=f'PCA Interativo: Embeddings por Tópico - {embedding_view.upper()}',
                color_discrete_sequence=px.colors.qualitative.Plotly,
                width=1200,
                height=800
            )
            
            # Customizações
            fig.update_traces(
                marker=dict(
                    line=dict(width=0.5, color='white'),
                    opacity=0.7
                )
            )
            
            fig.update_layout(
                hovermode='closest',
                template='plotly_white',
                font=dict(size=12),
                legend=dict(
                    title='Tópico',
                    yanchor="top",
                    y=0.99,
                    xanchor="right",
                    x=0.99
                )
            )
            
            path = os.path.join(topics_dir, f"pca_scatter_interactive_{embedding_view}{product_suffix}_{self.timestamp}.html")
            fig.write_html(path)
            fig.write_image(path.replace('.html', '.png'), width=1200, height=800)
            
            viz_count += 1
            log.info(f"  ✓ Scatter interativo: {path}")
        
        except Exception as e:
            log.warning(f"  ⚠ Erro ao gerar scatter interativo: {e}")
        
        # 4. SCATTER INTERATIVO: Por Outcome (MELHORADO)
        try:
            log.info("  4. Scatter interativo por outcome (com melhorias)...")
            
            import plotly.graph_objects as go
            from scipy.stats import gaussian_kde
            
            # Cria figura vazia
            fig = go.Figure()
            
            # Máscara por outcome
            mask_perdida = (df['Outcome'] == 'perdida')
            mask_ganha = (df['Outcome'] == 'ganha')
            
            # 1. PERDIDAS PRIMEIRO (ficam embaixo)
            fig.add_trace(go.Scatter(
                x=df[mask_perdida]['PC1'],
                y=df[mask_perdida]['PC2'],
                mode='markers',
                name='Venda Perdida',
                marker=dict(
                    color='#e74c3c',
                    size=df[mask_perdida]['Size'] * 0.8,  # Ligeiramente menores
                    opacity=0.4,
                    line=dict(width=0.3, color='white')
                ),
                customdata=df[mask_perdida][['Call ID', 'Tópico', 'Produto', 'Duração']].values,
                hovertemplate='<b>Perdida</b><br>' +
                             'Call ID: %{customdata[0]}<br>' +
                             'Tópico: %{customdata[1]}<br>' +
                             'Produto: %{customdata[2]}<br>' +
                             'Duração: %{customdata[3]:.1f}s<br>' +
                             'PC1: %{x:.3f}<br>' +
                             'PC2: %{y:.3f}<extra></extra>'
            ))
            
            # 2. GANHAS DEPOIS (ficam em cima, mais visíveis)
            fig.add_trace(go.Scatter(
                x=df[mask_ganha]['PC1'],
                y=df[mask_ganha]['PC2'],
                mode='markers',
                name='Venda Ganha',
                marker=dict(
                    color='#2ecc71',
                    size=df[mask_ganha]['Size'],
                    opacity=0.75,
                    line=dict(width=0.5, color='white')
                ),
                customdata=df[mask_ganha][['Call ID', 'Tópico', 'Produto', 'Duração']].values,
                hovertemplate='<b>Ganha</b><br>' +
                             'Call ID: %{customdata[0]}<br>' +
                             'Tópico: %{customdata[1]}<br>' +
                             'Produto: %{customdata[2]}<br>' +
                             'Duração: %{customdata[3]:.1f}s<br>' +
                             'PC1: %{x:.3f}<br>' +
                             'PC2: %{y:.3f}<extra></extra>'
            ))
            
            # 3. CONTORNO DE DENSIDADE (ganhas)
            ganhas_points = df[mask_ganha][['PC1', 'PC2']].values
            
            if len(ganhas_points) > 10:
                try:
                    kde = gaussian_kde(ganhas_points.T)
                    
                    # Grid para avaliar densidade
                    x_min, x_max = df['PC1'].min(), df['PC1'].max()
                    y_min, y_max = df['PC2'].min(), df['PC2'].max()
                    
                    x_grid = np.linspace(x_min, x_max, 50)
                    y_grid = np.linspace(y_min, y_max, 50)
                    X, Y = np.meshgrid(x_grid, y_grid)
                    
                    positions = np.vstack([X.ravel(), Y.ravel()])
                    Z = np.reshape(kde(positions).T, X.shape)
                    
                    # Adiciona contorno
                    fig.add_trace(go.Contour(
                        x=x_grid,
                        y=y_grid,
                        z=Z,
                        colorscale='Greens',
                        opacity=0.25,
                        showscale=False,
                        contours=dict(
                            start=Z.min(),
                            end=Z.max(),
                            size=(Z.max()-Z.min())/5
                        ),
                        name='Densidade de Ganhas',
                        hoverinfo='skip',
                        showlegend=True
                    ))
                    log.info("    ✓ Contornos de densidade adicionados")
                except Exception as e_kde:
                    log.warning(f"    ⚠ Não foi possível gerar contornos: {e_kde}")
            
            # 4. LINHAS DE QUADRANTES
            fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.3, 
                         annotation_text="", showlegend=False)
            fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.3,
                         annotation_text="", showlegend=False)
            
            # 5. ANOTAÇÃO DO QUADRANTE ÓTIMO
            # Calcula médias para determinar quadrante ótimo
            pc1_ganha_mean = df[mask_ganha]['PC1'].mean()
            pc2_ganha_mean = df[mask_ganha]['PC2'].mean()
            
            # Posiciona anotação no quadrante onde ganhas se concentram
            x_max_range = df['PC1'].max()
            y_max_range = df['PC2'].max()
            
            annot_x = x_max_range * 0.6 if pc1_ganha_mean > 0 else -x_max_range * 0.6
            annot_y = y_max_range * 0.7 if pc2_ganha_mean > 0 else -y_max_range * 0.7
            
            fig.add_annotation(
                x=annot_x,
                y=annot_y,
                text="🎯 Maior probabilidade<br>de conversão<br>(relativa)",
                showarrow=True,
                arrowhead=2,
                arrowcolor="green",
                arrowwidth=2,
                ax=-50 if pc1_ganha_mean > 0 else 50,
                ay=-50 if pc2_ganha_mean > 0 else 50,
                font=dict(size=11, color="green"),
                bgcolor="white",
                bordercolor="green",
                borderwidth=2,
                opacity=0.9
            )
            
            # Layout
            fig.update_layout(
                title=f'PCA Interativo: Embeddings por Outcome - {embedding_view.upper()}',
                xaxis_title=f'Componente Principal 1 ({var1:.1%})',
                yaxis_title=f'Componente Principal 2 ({var2:.1%})',
                width=1200,
                height=800,
                hovermode='closest',
                template='plotly_white',
                font=dict(size=12),
                legend=dict(
                    title='Outcome',
                    yanchor="top",
                    y=0.99,
                    xanchor="right",
                    x=0.99,
                    bgcolor="rgba(255,255,255,0.8)"
                )
            )
            
            path = os.path.join(topics_dir, f"pca_scatter_outcome_interactive_{embedding_view}{product_suffix}_{self.timestamp}.html")
            fig.write_html(path)
            fig.write_image(path.replace('.html', '.png'), width=1200, height=800)
            
            viz_count += 1
            log.info(f"  ✓ Scatter outcome interativo (melhorado): {path}")
        
        except Exception as e:
            log.warning(f"  ⚠ Erro ao gerar scatter outcome interativo: {e}")
        
        log.info(f"\n✓ {viz_count} scatter plots PCA gerados")

