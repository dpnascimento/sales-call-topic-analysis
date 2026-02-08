#!/usr/bin/env python3
"""
Script de teste para scatter plots PCA + BERTopic

Testa a nova funcionalidade que gera visualizações 2D dos embeddings
reduzidos por PCA, coloridos por tópico ou outcome.

Usage:
    source .venv/bin/activate
    python v3/test_pca_scatter.py
"""

import sys
import os

# Adiciona o diretório raiz ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from v3.core.database_v3 import DatabaseManagerV3
from v3.analysis.topics_v3 import TopicAnalyzerV3
from v3.config import settings_v3
import logging

# Configura logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)

def main():
    """
    Testa a geração de scatter plots PCA
    """
    log.info("=" * 80)
    log.info("TESTE: Scatter Plots PCA + BERTopic")
    log.info("=" * 80)
    
    # Inicializa conexão
    db = DatabaseManagerV3()
    db.connect()
    
    try:
        # Inicializa analisador
        analyzer = TopicAnalyzerV3(db)
        
        # Força geração de plots
        settings_v3.TOPICS_GENERATE_PLOTS = True
        log.info(f"✓ TOPICS_GENERATE_PLOTS = {settings_v3.TOPICS_GENERATE_PLOTS}")
        
        # Testa apenas com view 'full' (mais rápido)
        view = 'full'
        log.info(f"\n{'=' * 80}")
        log.info(f"Testando view: {view.upper()}")
        log.info(f"{'=' * 80}\n")
        
        result = analyzer.analyze_topics(embedding_view=view)
        
        if result['status'] == 'success':
            log.info(f"\n✓ Análise concluída para view '{view}':")
            log.info(f"  - Tópicos: {result['stats']['n_topics']}")
            log.info(f"  - Documentos: {result['stats']['n_docs']}")
            if 'n_outliers' in result['stats']:
                n_outliers = result['stats']['n_outliers']
                n_docs = result['stats']['n_docs']
                pct = (100 * n_outliers / n_docs) if n_docs > 0 else 0
                log.info(f"  - Outliers: {n_outliers} ({pct:.1f}%)")
        else:
            log.error(f"\n✗ Erro na análise de '{view}': {result.get('error', 'Desconhecido')}")
        
        log.info(f"\n{'=' * 80}")
        log.info("✓ TESTE CONCLUÍDO")
        log.info(f"{'=' * 80}")
        log.info(f"\nArquivos gerados em: {settings_v3.V3_TOPICS_DIR}")
        log.info(f"\nProcure por arquivos PCA:")
        log.info(f"  - pca_scatter_by_topic_*.png (estático por tópico)")
        log.info(f"  - pca_scatter_by_outcome_*.png (estático por outcome)")
        log.info(f"  - pca_scatter_interactive_*.html (interativo por tópico)")
        log.info(f"  - pca_scatter_outcome_interactive_*.html (interativo por outcome)")
    
    finally:
        db.close()

if __name__ == "__main__":
    main()

