#!/usr/bin/env python3
"""
Script de teste para análise temporal de tópicos (Topics over Time)

Testa a funcionalidade de análise temporal que gera:
1. Evolução calendário (nativo BERTopic)
2. Sazonalidade por dia da semana
3. Evolução mensal
4. Comparação temporal: ganhas vs perdidas

Usage:
    source .venv/bin/activate
    python test_topics_overtime.py
"""

import sys
import os

from core.database_v3 import DatabaseManagerV3
from analysis.topics_v3 import TopicAnalyzerV3
from config import settings_v3
import logging

# Configura logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)

def main():
    """
    Testa a análise temporal de tópicos
    """
    log.info("=" * 80)
    log.info("TESTE: Análise Temporal de Tópicos (Topics over Time)")
    log.info("=" * 80)
    
    # Inicializa conexão
    db = DatabaseManagerV3()
    db.connect()
    
    try:
        # Inicializa analisador
        analyzer = TopicAnalyzerV3(db)
        
        # Força análise temporal
        settings_v3.TOPICS_GENERATE_OVERTIME = True
        log.info(f"✓ TOPICS_GENERATE_OVERTIME = {settings_v3.TOPICS_GENERATE_OVERTIME}")
        log.info(f"✓ TOPICS_OVERTIME_BINS = {settings_v3.TOPICS_OVERTIME_BINS}")
        
        # Testa com cada view
        for view in ['full', 'agent', 'client']:
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
        log.info(f"\nProcure por arquivos:")
        log.info(f"  - topics_overtime_calendar_*.html")
        log.info(f"  - topics_overtime_weekday_*.html")
        log.info(f"  - topics_overtime_monthly_*.html")
        log.info(f"  - topics_overtime_outcome_*.html")
    
    finally:
        db.close()

if __name__ == "__main__":
    main()

