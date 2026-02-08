#!/usr/bin/env python3
"""
Script de teste para as novas visualizações implementadas:
1. Heatmap de distâncias entre centroides
2. Word clouds por tópico

Uso:
    python test_new_visualizations.py [--heatmap-only | --wordclouds-only]
"""
import sys
import os
import logging
from pathlib import Path

from core.database_v3 import DatabaseManagerV3
from visualization.comparison_plots import ComparisonPlotter
from analysis.topics_v3 import TopicAnalyzerV3
from config import settings_v3
from datetime import datetime

# Configura logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)


def test_centroid_heatmap():
    """Testa geração do heatmap de distâncias entre centroides"""
    log.info("\n" + "="*80)
    log.info("🔥 TESTE: Heatmap de Distâncias entre Centroides")
    log.info("="*80)
    
    try:
        db = DatabaseManagerV3()
        db.connect()  # Estabelece conexão
        plotter = ComparisonPlotter(db)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(
            settings_v3.V3_PLOTS_DIR,
            f"TEST_centroid_heatmap_{timestamp}.png"
        )
        
        log.info(f"Gerando heatmap...")
        plotter.plot_centroid_distance_heatmap(
            output_path=output_path,
            embedding_views=['agent', 'client']
        )
        
        if os.path.exists(output_path):
            size = os.path.getsize(output_path) / 1024  # KB
            log.info(f"✅ SUCESSO! Heatmap gerado: {output_path} ({size:.1f} KB)")
            return True
        else:
            log.error(f"❌ FALHA! Arquivo não foi criado: {output_path}")
            return False
            
    except Exception as e:
        log.error(f"❌ ERRO ao gerar heatmap: {e}", exc_info=True)
        return False
    finally:
        if 'db' in locals():
            db.close()


def test_wordclouds():
    """Testa geração de word clouds por tópico"""
    log.info("\n" + "="*80)
    log.info("🎨 TESTE: Word Clouds por Tópico")
    log.info("="*80)
    
    try:
        db = DatabaseManagerV3()
        db.connect()  # Estabelece conexão
        analyzer = TopicAnalyzerV3(db)
        
        # Testa apenas com uma visão (full) para ser rápido
        log.info("Testando com visão 'full'...")
        result = analyzer.analyze_topics(embedding_view='full')
        
        if result.get('status') == 'success':
            timestamp = analyzer.timestamp
            wordcloud_dir = os.path.join(
                settings_v3.V3_TOPICS_DIR,
                f"wordclouds_full_{timestamp}"
            )
            
            if os.path.exists(wordcloud_dir):
                # Conta arquivos PNG
                import glob
                png_files = glob.glob(os.path.join(wordcloud_dir, "*.png"))
                index_html = os.path.join(wordcloud_dir, "index.html")
                
                log.info(f"✅ SUCESSO! Word clouds geradas:")
                log.info(f"  • Diretório: {wordcloud_dir}")
                log.info(f"  • PNGs gerados: {len(png_files)}")
                log.info(f"  • Índice HTML: {'✓' if os.path.exists(index_html) else '✗'}")
                
                if os.path.exists(index_html):
                    log.info(f"  • Abra no navegador: file://{index_html}")
                
                return True
            else:
                log.warning(f"⚠️  Diretório não encontrado: {wordcloud_dir}")
                return False
        elif result.get('status') == 'no_data':
            log.warning("⚠️  Sem dados suficientes para análise de tópicos")
            return False
        else:
            log.error(f"❌ FALHA na análise de tópicos: {result}")
            return False
            
    except Exception as e:
        log.error(f"❌ ERRO ao gerar word clouds: {e}", exc_info=True)
        return False
    finally:
        if 'db' in locals():
            db.close()


def main():
    """Execução principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Testa novas visualizações')
    parser.add_argument('--heatmap-only', action='store_true',
                       help='Testa apenas heatmap de centroides')
    parser.add_argument('--wordclouds-only', action='store_true',
                       help='Testa apenas word clouds')
    
    args = parser.parse_args()
    
    results = {}
    
    if args.wordclouds_only:
        results['wordclouds'] = test_wordclouds()
    elif args.heatmap_only:
        results['heatmap'] = test_centroid_heatmap()
    else:
        # Testa ambos
        results['heatmap'] = test_centroid_heatmap()
        results['wordclouds'] = test_wordclouds()
    
    # Resumo
    log.info("\n" + "="*80)
    log.info("📊 RESUMO DOS TESTES")
    log.info("="*80)
    
    for name, success in results.items():
        status = "✅ PASSOU" if success else "❌ FALHOU"
        log.info(f"  • {name.upper()}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        log.info("\n🎉 Todos os testes passaram!")
        sys.exit(0)
    else:
        log.error("\n❌ Alguns testes falharam")
        sys.exit(1)


if __name__ == "__main__":
    main()

