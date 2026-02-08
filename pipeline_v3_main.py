#!/usr/bin/env python3
"""
Pipeline Principal - Análise de Padrões Semânticos em Vendas
Integra todas as análises: protótipos, padrões, comparações e visualizações
"""
import sys
from pathlib import Path
import logging
import os
from datetime import datetime

from core.database_v3 import DatabaseManagerV3
from analysis.prototypes_v3 import PrototypeAnalyzerV3
from analysis.patterns_by_product import PatternsByProductAnalyzer
from analysis.patterns_by_product_status import PatternsByProductStatusAnalyzer
from analysis.comparisons import EmbeddingViewComparator
from analysis.embedding_geometry import EmbeddingGeometryAnalyzer
from analysis.topics_v3 import TopicAnalyzerV3
from visualization.umap_plots import UMAPVisualizer
from visualization.comparison_plots import ComparisonPlotter
from visualization.pca_umap_plots import PCAUMAPVisualizer
from config import settings_v3

# Configuração de logging
logging.basicConfig(
    level=getattr(logging, settings_v3.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(settings_v3.LOG_FILE),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


class PipelineV3:
    """Pipeline principal de análise"""
    
    def __init__(self):
        self.db = None
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = None
        
        # Analisadores
        self.prototype_analyzer = None
        self.patterns_analyzer = None
        self.patterns_status_analyzer = None
        self.view_comparator = None
        self.geometry_analyzer = None
        self.topic_analyzer = None
        self.umap_visualizer = None
        self.comparison_plotter = None
        self.pca_umap_visualizer = None
    
    def setup(self):
        """Inicializa conexão e analisadores"""
        log.info("="*80)
        log.info("🚀 PIPELINE - ANÁLISE DE PADRÕES SEMÂNTICOS EM VENDAS")
        log.info("="*80)
        log.info(f"Timestamp: {self.timestamp}")
        log.info(f"Output dir: {settings_v3.V3_OUTPUT_DIR}")
        
        # Conecta ao banco
        log.info("\n1️⃣  Conectando ao banco de dados...")
        self.db = DatabaseManagerV3()
        self.db.connect()
        log.info("✓ Conexão estabelecida")
        
        # Inicializa analisadores
        log.info("\n2️⃣  Inicializando analisadores...")
        self.prototype_analyzer = PrototypeAnalyzerV3(self.db)
        self.patterns_analyzer = PatternsByProductAnalyzer(self.db)
        self.patterns_status_analyzer = PatternsByProductStatusAnalyzer(self.db)
        self.view_comparator = EmbeddingViewComparator(self.db)
        self.geometry_analyzer = EmbeddingGeometryAnalyzer(self.db)
        self.topic_analyzer = TopicAnalyzerV3(self.db)
        self.umap_visualizer = UMAPVisualizer(self.db)
        self.comparison_plotter = ComparisonPlotter(self.db)
        self.pca_umap_visualizer = PCAUMAPVisualizer()
        log.info("✓ Analisadores inicializados")
    
    def run_prototype_analysis(self):
        """Executa análise de protótipos"""
        log.info("\n" + "="*80)
        log.info("3️⃣  ANÁLISE DE PROTÓTIPOS")
        log.info("="*80)
        
        results = {}
        
        for view in settings_v3.EMBEDDING_VIEWS:
            log.info(f"\n🔍 Visão: {view.upper()}")
            
            # Protótipos globais
            global_protos = self.prototype_analyzer.compute_global_prototypes(view)
            results[f"global_{view}"] = global_protos
            
            # Protótipos por produto
            if settings_v3.COMPUTE_PROTOTYPES_PER_PRODUCT:
                product_protos = self.prototype_analyzer.compute_all_products_prototypes(view)
                results[f"products_{view}"] = product_protos
                
                # Comparação de separação entre produtos
                product_comparison = self.prototype_analyzer.compare_products_separation(view)
                results[f"product_separation_{view}"] = product_comparison
        
        # Exporta resultados
        output_file = os.path.join(settings_v3.V3_DATA_DIR, f"prototypes_{self.timestamp}.json")
        self.prototype_analyzer.export_results(output_file)
        
        return results
    
    def run_pattern_analysis(self):
        """Executa análise de padrões linguísticos"""
        log.info("\n" + "="*80)
        log.info("4️⃣  ANÁLISE DE PADRÕES LINGUÍSTICOS")
        log.info("="*80)
        
        results = {}
        
        # Análise por produto (papel: AGENTE)
        log.info("\n🎯 Analisando padrões por produto (AGENTE)...")
        agent_patterns = self.patterns_analyzer.analyze_all_products(role="AGENTE")
        results["agent"] = agent_patterns
        
        # Análise por produto (papel: CLIENTE)
        log.info("\n🎯 Analisando padrões por produto (CLIENTE)...")
        client_patterns = self.patterns_analyzer.analyze_all_products(role="CLIENTE")
        results["client"] = client_patterns
        
        # Exporta resultados
        output_file_agent = os.path.join(settings_v3.V3_DATA_DIR, f"patterns_agent_{self.timestamp}.json")
        output_file_client = os.path.join(settings_v3.V3_DATA_DIR, f"patterns_client_{self.timestamp}.json")
        
        self.patterns_analyzer.export_results(agent_patterns, output_file_agent)
        self.patterns_analyzer.export_results(client_patterns, output_file_client)
        
        return results
    
    def run_pattern_status_analysis(self):
        """Executa análise de padrões por produto + status"""
        log.info("\n" + "="*80)
        log.info("5️⃣  ANÁLISE DE PADRÕES POR PRODUTO + STATUS")
        log.info("="*80)
        
        results = {}
        
        # Insights consolidados
        log.info("\n💡 Gerando insights por produto e status...")
        insights = self.patterns_status_analyzer.generate_insights_by_product_status(role="AGENTE")
        results["insights"] = insights
        
        # Comparação cross-product
        log.info("\n🔄 Comparando patterns através de produtos...")
        cross_product = self.patterns_status_analyzer.compare_status_patterns_across_products(role="AGENTE")
        results["cross_product"] = cross_product
        
        # Exporta resultados
        output_file = os.path.join(settings_v3.V3_DATA_DIR, f"patterns_status_{self.timestamp}.json")
        self.patterns_status_analyzer.export_results(results, output_file)
        
        return results
    
    def run_view_comparison(self):
        """Executa comparação entre visões de embedding"""
        log.info("\n" + "="*80)
        log.info("6️⃣  COMPARAÇÃO DE VISÕES DE EMBEDDING")
        log.info("="*80)
        
        # Comparação global
        log.info("\n🌐 Comparação global...")
        global_comparison = self.view_comparator.compare_views_global()
        
        # Comparação por produtos
        log.info("\n📊 Comparação por produtos...")
        products_comparison = self.view_comparator.compare_views_all_products()
        
        # Recomendações
        log.info("\n💡 Gerando recomendações...")
        recommendations = self.view_comparator.generate_view_recommendations()
        
        # Exporta resultados
        output_file = os.path.join(settings_v3.V3_DATA_DIR, f"view_comparison_{self.timestamp}.json")
        results = self.view_comparator.export_comparison_results(output_file)
        
        return results
    
    def run_visualizations(self, comparison_results):
        """Gera todas as visualizações"""
        log.info("\n" + "="*80)
        log.info("7️⃣  GERANDO VISUALIZAÇÕES")
        log.info("="*80)
        
        # 1. UMAPs comparativos por visão
        if settings_v3.CREATE_UMAP_PER_VIEW:
            log.info("\n📈 Criando UMAPs comparativos...")
            comparative_umaps = self.umap_visualizer.create_comparative_umap()
            
            output_file = os.path.join(settings_v3.V3_PLOTS_DIR, f"umap_comparative_{self.timestamp}.png")
            self.umap_visualizer.plot_comparative_umaps(comparative_umaps, output_file)
        
        # 2. UMAPs por produto
        if settings_v3.CREATE_UMAP_PER_PRODUCT:
            log.info("\n📊 Criando UMAPs por produto...")
            for view in settings_v3.EMBEDDING_VIEWS:
                product_grid = self.umap_visualizer.create_product_grid_umap(view, max_products=9)
                
                output_file = os.path.join(settings_v3.V3_PLOTS_DIR, f"umap_products_{view}_{self.timestamp}.png")
                self.umap_visualizer.plot_product_grid(product_grid, output_file)
        
        # 3. Comparação de métricas entre visões
        log.info("\n📊 Criando gráficos de comparação de métricas...")
        if "global_comparison" in comparison_results:
            output_file = os.path.join(settings_v3.V3_PLOTS_DIR, f"view_metrics_{self.timestamp}.png")
            self.comparison_plotter.plot_view_comparison_metrics(
                comparison_results["global_comparison"],
                output_file
            )
        
        # 4. Performance de produtos por visão (heatmap)
        log.info("\n🔥 Criando heatmap de performance...")
        if "products_comparison" in comparison_results:
            output_file = os.path.join(settings_v3.V3_PLOTS_DIR, f"product_performance_{self.timestamp}.png")
            self.comparison_plotter.plot_product_performance_by_view(
                comparison_results["products_comparison"],
                output_file
            )
        
        # 5. Win rate por produto
        log.info("\n📊 Criando gráfico de win rate...")
        products = self.db.get_products()
        output_file = os.path.join(settings_v3.V3_PLOTS_DIR, f"win_rate_{self.timestamp}.png")
        self.comparison_plotter.plot_win_rate_by_product(products, output_file)
        
        # 6. Dashboard resumido
        log.info("\n🎨 Criando dashboard resumido...")
        output_file = os.path.join(settings_v3.V3_PLOTS_DIR, f"dashboard_{self.timestamp}.png")
        self.comparison_plotter.create_summary_dashboard(
            comparison_results.get("global_comparison", {}),
            products,
            output_file
        )
        
        # 7. Heatmap de distâncias entre centroides
        log.info("\n🔥 Criando heatmap de distâncias entre centroides...")
        try:
            output_file = os.path.join(settings_v3.V3_PLOTS_DIR, f"centroid_distance_heatmap_{self.timestamp}.png")
            self.comparison_plotter.plot_centroid_distance_heatmap(
                output_path=output_file,
                embedding_views=['agent', 'client']
            )
        except Exception as e:
            log.error(f"Erro ao gerar heatmap de centroides: {e}")
        
        log.info("\n✓ Todas as visualizações criadas")
    
    def run_pca_umap_analysis(self):
        """Executa análise PCA interpretável integrada com UMAP"""
        log.info("\n" + "="*80)
        log.info("7️⃣  ANÁLISE PCA + UMAP INTEGRADA")
        log.info("="*80)
        log.info("")
        
        results = {}
        
        for view in settings_v3.EMBEDDING_VIEWS:
            log.info(f"\n🔬 Analisando visão: {view.upper()}")
            
            # 1. Análise geométrica completa (PCA, LDA, outliers)
            geometry_results = self.geometry_analyzer.run_complete_analysis(
                embedding_view=view,
                include_patterns=False,  # Por enquanto sem padrões
                sample_size=settings_v3.UMAP_SAMPLE_SIZE
            )
            
            if not geometry_results:
                log.warning(f"Dados insuficientes para análise de {view}")
                continue
            
            # 2. Cria UMAP (se ainda não foi criado)
            umap_results = self.umap_visualizer.create_umap_by_view(
                embedding_view=view,
                sample_size=settings_v3.UMAP_SAMPLE_SIZE
            )
            
            if not umap_results:
                log.warning(f"Não foi possível criar UMAP para {view}")
                continue
            
            # 3. Extrai dados para visualização
            import numpy as np
            
            pca_analysis = geometry_results.get('pca_analysis', {})
            pca_projections = np.array(pca_analysis.get('pca_projections', []))
            components = pca_analysis.get('components', [])
            
            umap_coords = np.array(umap_results.get('embedding', []))
            metadata = umap_results.get('metadata', [])
            outcomes = [m['outcome'] for m in metadata]
            
            if len(pca_projections) == 0 or len(umap_coords) == 0:
                log.warning(f"Dados vazios para visualização de {view}")
                continue
            
            # Interpretações dos PCs
            pc_interpretations = [c['interpretation'] for c in components[:2]]
            variance_ratios = [c['variance_explained'] for c in components]
            
            log.info(f"  PC1: {pc_interpretations[0]} ({variance_ratios[0]:.1%})")
            log.info(f"  PC2: {pc_interpretations[1]} ({variance_ratios[1]:.1%})")
            
            # 4. Cria visualizações integradas
            log.info(f"\n📊 Criando visualizações PCA+UMAP para {view}...")
            
            # Biplot PCA
            self.pca_umap_visualizer.plot_pca_biplot(
                pca_projections=pca_projections[:, :2],
                outcomes=outcomes,
                component_interpretations=pc_interpretations,
                title=f"PCA Biplot - {view.capitalize()}",
                filename=f"pca_biplot_{view}_{self.timestamp}.png"
            )
            
            # UMAP colorido por PC1
            self.pca_umap_visualizer.plot_umap_colored_by_pc(
                umap_coords=umap_coords,
                pc_scores=pca_projections[:, 0],
                pc_label=pc_interpretations[0],
                outcomes=outcomes,
                title=f"UMAP × PC1 - {view.capitalize()}",
                filename=f"umap_by_pc1_{view}_{self.timestamp}.png"
            )
            
            # UMAP colorido por PC2
            self.pca_umap_visualizer.plot_umap_colored_by_pc(
                umap_coords=umap_coords,
                pc_scores=pca_projections[:, 1],
                pc_label=pc_interpretations[1],
                outcomes=outcomes,
                title=f"UMAP × PC2 - {view.capitalize()}",
                filename=f"umap_by_pc2_{view}_{self.timestamp}.png"
            )
            
            # Visualização integrada 2x2
            self.pca_umap_visualizer.plot_integrated_pca_umap(
                pca_projections=pca_projections[:, :2],
                umap_coords=umap_coords,
                outcomes=outcomes,
                pc1_interpretation=pc_interpretations[0],
                pc2_interpretation=pc_interpretations[1],
                title=f"Análise Integrada PCA+UMAP - {view.capitalize()}",
                filename=f"integrated_pca_umap_{view}_{self.timestamp}.png"
            )
            
            # Gráfico de variância explicada
            self.pca_umap_visualizer.plot_pca_variance_explained(
                variance_ratios=variance_ratios,
                interpretations=[c['interpretation'] for c in components],
                title=f"Variância Explicada - {view.capitalize()}",
                filename=f"pca_variance_{view}_{self.timestamp}.png"
            )
            
            # 5. Exporta resultados
            import json
            output_file = os.path.join(
                settings_v3.V3_DATA_DIR,
                f"pca_analysis_{view}_{self.timestamp}.json"
            )
            
            with open(output_file, 'w') as f:
                json.dump(geometry_results, f, indent=2)
            
            log.info(f"✓ Análise PCA+UMAP de {view} concluída")
            log.info(f"  Dados exportados: {output_file}")
            
            results[view] = geometry_results
        
        log.info("\n✓ Análise PCA+UMAP concluída para todas as visões")
        
        return results
    
    def run_topic_analysis(self):
        """Executa análise de tópicos com BERTopic"""
        log.info("\n" + "="*80)
        log.info("8️⃣  ANÁLISE DE TÓPICOS (BERTopic)")
        log.info("="*80)
        log.info("")
        
        if not settings_v3.DO_TOPICS:
            log.info("Análise de tópicos desabilitada")
            return {"status": "disabled"}
        
        results = {}
        
        # Analisa para cada visão de embedding
        for view in settings_v3.EMBEDDING_VIEWS:
            log.info(f"\n🔬 Analisando tópicos - visão: {view.upper()}")
            
            try:
                result = self.topic_analyzer.analyze_topics(embedding_view=view)
                results[view] = result
                
                if result.get("status") == "success":
                    stats = result.get("stats", {})
                    log.info(f"✓ {stats.get('n_topics', 0)} tópicos identificados")
                    log.info(f"  • Documentos: {stats.get('n_docs', 0)}")
                    log.info(f"  • Cobertura: {stats.get('coverage', 0):.1%}")
                elif result.get("status") == "no_data":
                    log.warning(f"Sem dados para visão {view}")
                elif result.get("status") == "error":
                    log.error(f"Erro na visão {view}: {result.get('error')}")
            
            except Exception as e:
                log.error(f"Erro ao analisar tópicos para visão {view}: {e}", exc_info=True)
                results[view] = {"status": "error", "error": str(e)}
        
        log.info("\n✓ Análise de tópicos concluída")
        return results
    
    def run(self):
        """Executa pipeline completo"""
        try:
            # Setup
            self.setup()
            
            # Análises
            prototype_results = self.run_prototype_analysis()
            pattern_results = self.run_pattern_analysis()
            pattern_status_results = self.run_pattern_status_analysis()
            comparison_results = self.run_view_comparison()
            
            # Visualizações
            self.run_visualizations(comparison_results)
            
            # Análise PCA + UMAP
            pca_umap_results = self.run_pca_umap_analysis()
            
            # Análise de Tópicos (BERTopic)
            topic_results = self.run_topic_analysis()
            
            # Finalização
            log.info("\n" + "="*80)
            log.info("✅ PIPELINE CONCLUÍDO COM SUCESSO!")
            log.info("="*80)
            log.info(f"\n📁 Resultados salvos em: {settings_v3.V3_OUTPUT_DIR}")
            log.info(f"  • Dados: {settings_v3.V3_DATA_DIR}")
            log.info(f"  • Gráficos: {settings_v3.V3_PLOTS_DIR}")
            log.info(f"  • Relatórios: {settings_v3.V3_REPORTS_DIR}")
            
            return {
                "status": "success",
                "timestamp": self.timestamp,
                "prototype_results": prototype_results,
                "pattern_results": pattern_results,
                "pattern_status_results": pattern_status_results,
                "comparison_results": comparison_results,
                "pca_umap_results": pca_umap_results,
                "topic_results": topic_results
            }
            
        except Exception as e:
            log.error(f"\n\n✗ ERRO FATAL: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e)
            }
        
        finally:
            if self.db:
                self.db.close()


def main():
    """Ponto de entrada principal"""
    pipeline = PipelineV3()
    results = pipeline.run()
    
    if results["status"] == "success":
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()

