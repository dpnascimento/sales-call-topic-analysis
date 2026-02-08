#!/usr/bin/env python3
"""
Script Standalone para Análise BERTopic
Executa apenas a análise de tópicos sem rodar o pipeline completo
"""
import sys
from pathlib import Path
import logging
import argparse

from core.database_v3 import DatabaseManagerV3
from analysis.topics_v3 import TopicAnalyzerV3
from config import settings_v3

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


def parse_args():
    """Parse argumentos da linha de comando"""
    parser = argparse.ArgumentParser(
        description='Executa análise BERTopic em embeddings V2'
    )
    parser.add_argument(
        '--views',
        nargs='+',
        choices=['full', 'agent', 'client'],
        default=['full'],
        help='Visões de embedding a analisar (padrão: full)'
    )
    parser.add_argument(
        '--max-docs',
        type=int,
        default=None,
        help=f'Máximo de documentos (padrão: {settings_v3.TOPICS_MAX_DOCUMENTS})'
    )
    parser.add_argument(
        '--min-cluster-size',
        type=int,
        default=None,
        help=f'Tamanho mínimo do cluster (padrão: {settings_v3.TOPICS_MIN_CLUSTER_SIZE})'
    )
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Não gerar visualizações HTML'
    )
    parser.add_argument(
        '--by-product',
        action='store_true',
        help='Analisa tópicos separadamente por produto'
    )
    parser.add_argument(
        '--products',
        nargs='+',
        default=None,
        help='Lista de produtos específicos a analisar (ex: "Seguro de Carga" "Seguro Garantia")'
    )
    return parser.parse_args()


def main():
    """Função principal"""
    args = parse_args()
    
    log.info("="*80)
    log.info("🔬 ANÁLISE BERTOPIC STANDALONE")
    log.info("="*80)
    log.info(f"Visões a analisar: {', '.join(args.views)}")
    log.info(f"Output: {settings_v3.V3_TOPICS_DIR}")
    log.info("")
    
    # Aplica configurações customizadas
    if args.max_docs:
        settings_v3.TOPICS_MAX_DOCUMENTS = args.max_docs
        log.info(f"⚙️  Max documentos: {args.max_docs}")
    
    if args.min_cluster_size:
        settings_v3.TOPICS_MIN_CLUSTER_SIZE = args.min_cluster_size
        log.info(f"⚙️  Min cluster size: {args.min_cluster_size}")
    
    if args.no_plots:
        settings_v3.TOPICS_GENERATE_PLOTS = False
        log.info(f"⚙️  Plots desabilitados")
    
    log.info("")
    
    # Conecta ao banco
    log.info("1️⃣  Conectando ao banco...")
    db = DatabaseManagerV3()
    db.connect()
    log.info("✓ Conectado")
    
    # Inicializa analisador
    topic_analyzer = TopicAnalyzerV3(db)
    
    # Executa análise para cada visão
    results = {}
    
    # Decide se analisa por produto ou todos juntos
    if args.by_product or args.products:
        # Análise por produto
        for view in args.views:
            log.info("\n" + "="*80)
            log.info(f"📊 Analisando visão: {view.upper()} - POR PRODUTO")
            log.info("="*80)
            log.info("")
            
            try:
                if args.products:
                    # Produtos específicos
                    log.info(f"Produtos selecionados: {', '.join(args.products)}")
                    for product_name in args.products:
                        log.info(f"\n🔍 Processando: {product_name}")
                        result = topic_analyzer.analyze_topics_by_product(product_name, embedding_view=view)
                        results[f"{view}_{product_name}"] = result
                else:
                    # Todos os produtos
                    log.info(f"Processando TODOS os produtos...")
                    product_results = topic_analyzer.analyze_all_products(embedding_view=view)
                    results[view] = {"status": "success", "products": product_results}
                    
            except Exception as e:
                log.error(f"❌ Erro ao processar visão '{view}' por produto: {e}", exc_info=True)
                results[view] = {"status": "error", "error": str(e)}
    else:
        # Análise global (todos os produtos juntos)
        for view in args.views:
            log.info("\n" + "="*80)
            log.info(f"📊 Analisando visão: {view.upper()} - TODOS PRODUTOS JUNTOS")
            log.info("="*80)
            log.info("")
            
            try:
                result = topic_analyzer.analyze_topics(embedding_view=view)
                results[view] = result
                
                if result.get("status") == "success":
                    stats = result.get("stats", {})
                    log.info("")
                    log.info("✅ Análise concluída com sucesso!")
                    log.info(f"  • Tópicos identificados: {stats.get('n_topics', 0)}")
                    log.info(f"  • Documentos processados: {stats.get('n_docs', 0)}")
                    log.info(f"  • Cobertura: {stats.get('coverage', 0):.1%}")
                    log.info(f"  • Outliers: {stats.get('n_outliers', 0)}")
                    
                    # Estatísticas por outcome
                    if 'outcome_stats' in stats:
                        log.info("")
                        log.info("  📈 Por Outcome:")
                        for outcome, ostats in stats['outcome_stats'].items():
                            log.info(f"    • {outcome.capitalize()}: {ostats['n_docs']} docs, "
                                    f"{ostats['n_topics']} tópicos, "
                                    f"cobertura {ostats['coverage']:.1%}")
                    
                    # Caminhos dos outputs
                    if 'output_paths' in result:
                        paths = result['output_paths']
                        log.info("")
                        log.info("  📁 Arquivos gerados:")
                        for key, path in paths.items():
                            log.info(f"    • {key}: {path}")
                
                elif result.get("status") == "no_data":
                    log.warning(f"⚠️  Sem dados disponíveis para visão '{view}'")
                
                elif result.get("status") == "error":
                    log.error(f"❌ Erro na visão '{view}': {result.get('error')}")
                
                elif result.get("status") == "disabled":
                    log.warning(f"⚠️  Análise de tópicos desabilitada (DO_TOPICS=False)")
            
            except Exception as e:
                log.error(f"❌ Erro ao processar visão '{view}': {e}", exc_info=True)
                results[view] = {"status": "error", "error": str(e)}
    
    # Fecha conexão
    db.close()
    
    # Resumo final
    log.info("\n" + "="*80)
    log.info("🎉 ANÁLISE BERTOPIC CONCLUÍDA")
    log.info("="*80)
    log.info(f"Visões processadas: {len(results)}")
    
    success_count = sum(1 for r in results.values() if r.get("status") == "success")
    log.info(f"Sucesso: {success_count}/{len(results)}")
    
    if success_count > 0:
        log.info(f"\n📂 Resultados em: {settings_v3.V3_TOPICS_DIR}")
        log.info("\n✨ Dica: Abra os arquivos .html no navegador para visualizações interativas!")
    
    return 0 if success_count > 0 else 1


if __name__ == "__main__":
    sys.exit(main())

