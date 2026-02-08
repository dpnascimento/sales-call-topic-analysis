#!/usr/bin/env python3
"""
Extrai métricas de protótipos do pipeline V3 para preencher tabelas do TCC

Usage:
    source .venv/bin/activate
    python v3/extract_prototype_metrics.py
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from v3.core.database_v3 import DatabaseManagerV3
from v3.analysis.prototypes_v3 import PrototypeAnalyzerV3
from v3.config import settings_v3
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

def print_table_separator():
    print("=" * 80)

def extract_global_prototypes():
    """Extrai métricas globais para cada visão"""
    print_table_separator()
    print("📊 TABELA X: ESTATÍSTICAS DOS PROTÓTIPOS GLOBAIS")
    print_table_separator()
    
    db = DatabaseManagerV3()
    db.connect()
    
    try:
        analyzer = PrototypeAnalyzerV3(db)
        
        for view in ['full', 'agent', 'client']:
            print(f"\n🔍 Visão: {view.upper()}")
            print("-" * 80)
            
            results = analyzer.compute_global_prototypes(embedding_view=view)
            
            if 'ganha' in results and 'perdida' in results:
                ganha = results['ganha']
                perdida = results['perdida']
                sep = results.get('separation', {})
                
                # Tabela formatada
                print(f"\n| Métrica                      | Vendas Ganhas     | Vendas Perdidas   |")
                print(f"|------------------------------|-------------------|-------------------|")
                print(f"| N (amostras)                 | {ganha['n_samples']:>17,} | {perdida['n_samples']:>17,} |")
                print(f"| Coesão intra-cluster         | {ganha['cohesion']:>17.3f} | {perdida['cohesion']:>17.3f} |")
                print(f"| Distância média ao protótipo | {ganha['mean_distance']:>17.3f} | {perdida['mean_distance']:>17.3f} |")
                print(f"| Desvio padrão da distância   | {ganha['std_distance']:>17.3f} | {perdida['std_distance']:>17.3f} |")
                print(f"| Mediana da distância         | {ganha['median_distance']:>17.3f} | {perdida['median_distance']:>17.3f} |")
                
                if sep:
                    print(f"\n📏 Separação entre protótipos:")
                    print(f"   - Distância cosseno: {sep['distance']:.4f}")
                    if 'silhouette' in sep:
                        sil = sep['silhouette']
                        print(f"   - Silhueta média: {sil.get('overall', 0):.4f}")
                        if 'ganha' in sil:
                            print(f"   - Silhueta (ganhas): {sil['ganha']:.4f}")
                        if 'perdida' in sil:
                            print(f"   - Silhueta (perdidas): {sil['perdida']:.4f}")
                
                # Win rate
                total = ganha['n_samples'] + perdida['n_samples']
                win_rate = 100 * ganha['n_samples'] / total
                print(f"\n📈 Win Rate: {win_rate:.1f}% ({ganha['n_samples']:,} / {total:,})")
                
            else:
                print("❌ Dados insuficientes para esta visão")
    
    finally:
        db.close()

def extract_view_comparison():
    """Extrai comparação entre visões"""
    print("\n")
    print_table_separator()
    print("📊 TABELA Y: SEPARAÇÃO DE PROTÓTIPOS POR VISÃO")
    print_table_separator()
    
    db = DatabaseManagerV3()
    db.connect()
    
    try:
        analyzer = PrototypeAnalyzerV3(db)
        
        results_by_view = {}
        
        for view in ['full', 'agent', 'client']:
            results = analyzer.compute_global_prototypes(embedding_view=view)
            
            if 'separation' in results:
                sep = results['separation']
                results_by_view[view] = {
                    'distance': sep['distance'],
                    'silhouette': sep['silhouette']['overall']
                }
        
        # Tabela formatada
        print(f"\n| Visão      | Distância | Silhueta | Ranking |")
        print(f"|------------|-----------|----------|---------|")
        
        # Ordena por silhueta
        ranked = sorted(results_by_view.items(), key=lambda x: x[1]['silhouette'], reverse=True)
        
        for rank, (view, metrics) in enumerate(ranked, 1):
            star = "**" if rank == 1 else "  "
            print(f"| {star}{view:8s}{star} | {metrics['distance']:9.4f} | {metrics['silhouette']:8.4f} | {rank:7d} |")
        
        # Diferenças percentuais
        if len(ranked) >= 2:
            best_view, best_metrics = ranked[0]
            print(f"\n✨ Melhor visão: {best_view.upper()}")
            
            for view, metrics in ranked[1:]:
                diff = ((best_metrics['silhouette'] - metrics['silhouette']) / metrics['silhouette']) * 100
                print(f"   → {diff:+.1f}% melhor que '{view}'")
    
    finally:
        db.close()

def extract_product_prototypes(product_name="Seguro Carga"):
    """Extrai métricas para um produto específico"""
    print("\n")
    print_table_separator()
    print(f"📊 TABELA Z: PROTÓTIPOS ESPECÍFICOS - {product_name}")
    print_table_separator()
    
    db = DatabaseManagerV3()
    db.connect()
    
    try:
        analyzer = PrototypeAnalyzerV3(db)
        
        results = analyzer.compute_product_prototypes(
            product_name=product_name,
            embedding_view='full'
        )
        
        if 'ganha' in results and 'perdida' in results:
            ganha = results['ganha']
            perdida = results['perdida']
            sep = results.get('separation', {})
            
            print(f"\n| Métrica                      | Ganhas (n={ganha['n_samples']}) | Perdidas (n={perdida['n_samples']}) |")
            print(f"|------------------------------|-------------------|---------------------|")
            print(f"| Coesão intra-cluster         | {ganha['cohesion']:>17.3f} | {perdida['cohesion']:>19.3f} |")
            print(f"| Distância média ao protótipo | {ganha['mean_distance']:>17.3f} | {perdida['mean_distance']:>19.3f} |")
            print(f"| Desvio padrão                | {ganha['std_distance']:>17.3f} | {perdida['std_distance']:>19.3f} |")
            
            if sep:
                print(f"\n📏 Separação:")
                print(f"   - Distância inter-cluster: {sep['distance']:.4f}")
                if 'silhouette' in sep:
                    print(f"   - Silhueta: {sep['silhouette']['overall']:.4f}")
            
            # Win rate do produto
            total = ganha['n_samples'] + perdida['n_samples']
            win_rate = 100 * ganha['n_samples'] / total
            print(f"\n📈 Win Rate ({product_name}): {win_rate:.1f}%")
        
        else:
            print(f"❌ Dados insuficientes para produto '{product_name}'")
    
    finally:
        db.close()

def list_available_products():
    """Lista produtos disponíveis para análise"""
    print("\n")
    print_table_separator()
    print("📋 PRODUTOS DISPONÍVEIS")
    print_table_separator()
    
    db = DatabaseManagerV3()
    db.connect()
    
    try:
        with db.get_cursor() as cur:
            sql = """
            SELECT 
                p.name,
                COUNT(DISTINCT ce.call_id) as n_calls,
                SUM(CASE WHEN co.outcome = 'ganha' THEN 1 ELSE 0 END) as n_ganha,
                SUM(CASE WHEN co.outcome = 'perdida' THEN 1 ELSE 0 END) as n_perdida
            FROM call_embeddings_v2 ce
            JOIN call_records cr ON cr.call_id = ce.call_id
            JOIN deals d ON d.deal_id = cr.deal_id
            JOIN pipelines p ON p.pipeline_id = d.pipeline
            JOIN call_outcomes co ON co.deal_id = d.deal_id
            WHERE ce.full_valid = TRUE
              AND co.outcome IN ('ganha', 'perdida')
            GROUP BY p.name
            HAVING COUNT(DISTINCT ce.call_id) >= 20
            ORDER BY COUNT(DISTINCT ce.call_id) DESC
            """
            
            cur.execute(sql)
            products = cur.fetchall()
            
            print(f"\n| Produto                      | Total | Ganhas | Perdidas | Win Rate |")
            print(f"|------------------------------|-------|--------|----------|----------|")
            
            for row in products:
                name = row['name'][:28]
                total = row['n_calls']
                ganha = row['n_ganha']
                perdida = row['n_perdida']
                wr = 100 * ganha / total if total > 0 else 0
                
                print(f"| {name:28s} | {total:5,} | {ganha:6,} | {perdida:8,} | {wr:7.1f}% |")
            
            print(f"\nTotal de produtos elegíveis: {len(products)}")
    
    finally:
        db.close()

def extract_product_comparison():
    """Extrai comparação completa entre produtos (visão full apenas)"""
    print("\n")
    print_table_separator()
    print("📊 TABELA: COMPARAÇÃO DE PROTÓTIPOS POR PRODUTO (VISÃO FULL)")
    print_table_separator()
    
    db = DatabaseManagerV3()
    db.connect()
    
    try:
        # 1. Busca produtos elegíveis
        with db.get_cursor() as cur:
            sql = """
            SELECT 
                p.name,
                COUNT(DISTINCT ce.call_id) as n_calls,
                SUM(CASE WHEN co.outcome = 'ganha' THEN 1 ELSE 0 END) as n_ganha,
                SUM(CASE WHEN co.outcome = 'perdida' THEN 1 ELSE 0 END) as n_perdida
            FROM call_embeddings_v2 ce
            JOIN call_records cr ON cr.call_id = ce.call_id
            JOIN deals d ON d.deal_id = cr.deal_id
            JOIN pipelines p ON p.pipeline_id = d.pipeline
            JOIN call_outcomes co ON co.deal_id = d.deal_id
            WHERE ce.full_valid = TRUE
              AND co.outcome IN ('ganha', 'perdida')
            GROUP BY p.name
            HAVING COUNT(DISTINCT ce.call_id) >= 20
            ORDER BY COUNT(DISTINCT ce.call_id) DESC
            """
            
            cur.execute(sql)
            products = cur.fetchall()
        
        print(f"\n✓ Encontrados {len(products)} produtos elegíveis (≥20 ligações)")
        
        # 2. Para cada produto, calcula métricas
        analyzer = PrototypeAnalyzerV3(db)
        product_metrics = []
        
        for row in products:
            product_name = row['name']
            print(f"\n⏳ Processando: {product_name}...", end=" ")
            
            try:
                results = analyzer.compute_product_prototypes(
                    product_name=product_name,
                    embedding_view='full'
                )
                
                if 'ganha' in results and 'perdida' in results:
                    ganha = results['ganha']
                    perdida = results['perdida']
                    sep = results.get('separation', {})
                    
                    n_total = ganha['n_samples'] + perdida['n_samples']
                    win_rate = 100 * ganha['n_samples'] / n_total if n_total > 0 else 0
                    
                    product_metrics.append({
                        'name': product_name,
                        'n_ganha': ganha['n_samples'],
                        'n_perdida': perdida['n_samples'],
                        'n_total': n_total,
                        'win_rate': win_rate,
                        'cohesion_ganha': ganha['cohesion'],
                        'cohesion_perdida': perdida['cohesion'],
                        'mean_dist_ganha': ganha['mean_distance'],
                        'mean_dist_perdida': perdida['mean_distance'],
                        'separation_distance': sep.get('distance', 0) if sep else 0,
                        'silhouette': sep.get('silhouette', {}).get('overall', 0) if sep else 0
                    })
                    print("✓")
                else:
                    print("⚠️ (dados insuficientes)")
            
            except Exception as e:
                print(f"❌ (erro: {e})")
        
        # 3. Tabela comparativa completa
        print("\n")
        print("=" * 150)
        print("TABELA COMPARATIVA: MÉTRICAS DE PROTÓTIPOS POR PRODUTO")
        print("=" * 150)
        
        # Header
        print(f"\n{'Produto':<30} | {'N Ganha':>8} | {'N Perdida':>10} | {'Win Rate':>8} | {'Coesão G':>9} | {'Coesão P':>9} | {'Separação':>10} | {'Silhueta':>9}")
        print(f"{'-'*30}-|{'-'*10}|{'-'*12}|{'-'*10}|{'-'*11}|{'-'*11}|{'-'*12}|{'-'*10}")
        
        # Ordena por win rate
        product_metrics.sort(key=lambda x: x['win_rate'], reverse=True)
        
        for pm in product_metrics:
            name = pm['name'][:28]
            print(f"{name:<30} | {pm['n_ganha']:>8,} | {pm['n_perdida']:>10,} | {pm['win_rate']:>7.1f}% | "
                  f"{pm['cohesion_ganha']:>9.3f} | {pm['cohesion_perdida']:>9.3f} | "
                  f"{pm['separation_distance']:>10.4f} | {pm['silhouette']:>9.4f}")
        
        # 4. Estatísticas agregadas
        print("\n")
        print("=" * 150)
        print("ESTATÍSTICAS AGREGADAS (POR PRODUTO)")
        print("=" * 150)
        
        if product_metrics:
            import numpy as np
            
            win_rates = [pm['win_rate'] for pm in product_metrics]
            separations = [pm['separation_distance'] for pm in product_metrics]
            silhouettes = [pm['silhouette'] for pm in product_metrics]
            
            print(f"\n📊 Win Rate:")
            print(f"   - Média: {np.mean(win_rates):.1f}%")
            print(f"   - Mediana: {np.median(win_rates):.1f}%")
            print(f"   - Desvio: ±{np.std(win_rates):.1f}%")
            print(f"   - Min/Max: {np.min(win_rates):.1f}% / {np.max(win_rates):.1f}%")
            
            print(f"\n📏 Separação (distância cosseno):")
            print(f"   - Média: {np.mean(separations):.4f}")
            print(f"   - Mediana: {np.median(separations):.4f}")
            print(f"   - Min/Max: {np.min(separations):.4f} / {np.max(separations):.4f}")
            
            print(f"\n🎯 Silhueta:")
            print(f"   - Média: {np.mean(silhouettes):.4f}")
            print(f"   - Mediana: {np.median(silhouettes):.4f}")
            print(f"   - Min/Max: {np.min(silhouettes):.4f} / {np.max(silhouettes):.4f}")
            
            # Ranking dos 3 melhores produtos
            print(f"\n🏆 TOP 3 PRODUTOS (por Silhueta):")
            top3 = sorted(product_metrics, key=lambda x: x['silhouette'], reverse=True)[:3]
            for i, pm in enumerate(top3, 1):
                print(f"   {i}. {pm['name']}: Silhueta={pm['silhouette']:.4f}, Win Rate={pm['win_rate']:.1f}%")
            
            # Insights
            print(f"\n💡 INSIGHTS:")
            high_sil = [pm for pm in product_metrics if pm['silhouette'] > np.median(silhouettes)]
            low_sil = [pm for pm in product_metrics if pm['silhouette'] <= np.median(silhouettes)]
            
            if high_sil:
                avg_wr_high = np.mean([pm['win_rate'] for pm in high_sil])
                avg_wr_low = np.mean([pm['win_rate'] for pm in low_sil]) if low_sil else 0
                
                print(f"   - Produtos com silhueta acima da mediana têm win rate médio de {avg_wr_high:.1f}%")
                print(f"   - Produtos com silhueta abaixo da mediana têm win rate médio de {avg_wr_low:.1f}%")
                
                if avg_wr_high > avg_wr_low:
                    print(f"   ✓ Maior separação semântica correlaciona com melhor desempenho (+{avg_wr_high - avg_wr_low:.1f}pp)")
                else:
                    print(f"   ⚠️ Separação semântica não correlaciona diretamente com win rate")
        
        # 5. Tabela LaTeX (para copiar direto para o TCC)
        print("\n")
        print("=" * 150)
        print("TABELA LATEX (copie para o TCC)")
        print("=" * 150)
        
        print("\n\\begin{table}[htbp]")
        print("\\centering")
        print("\\caption{Métricas de Protótipos por Produto (Visão \\textit{full})}")
        print("\\label{tab:prototipos-produto}")
        print("\\begin{tabular}{lrrrrr}")
        print("\\toprule")
        print("\\textbf{Produto} & \\textbf{N Ganhas} & \\textbf{N Perdidas} & \\textbf{Win Rate} & \\textbf{Separação} & \\textbf{Silhueta} \\\\")
        print("\\midrule")
        
        # Top 5 produtos
        for pm in product_metrics[:5]:
            name = pm['name'].replace('&', '\\&')  # Escape LaTeX
            print(f"{name} & {pm['n_ganha']:,} & {pm['n_perdida']:,} & "
                  f"{pm['win_rate']:.1f}\\% & {pm['separation_distance']:.4f} & {pm['silhouette']:.4f} \\\\")
        
        if len(product_metrics) > 5:
            print("\\midrule")
            print(f"\\textit{{Outros {len(product_metrics)-5} produtos}} & \\multicolumn{{5}}{{c}}{{...}} \\\\")
        
        print("\\bottomrule")
        print("\\end{tabular}")
        print("\\end{table}")
    
    finally:
        db.close()

def main():
    """Executa todas as extrações"""
    print("\n" + "🎯 " * 20)
    print("EXTRAÇÃO DE MÉTRICAS DE PROTÓTIPOS PARA TCC")
    print("🎯 " * 20 + "\n")
    
    # 1. Protótipos globais
    extract_global_prototypes()
    
    # 2. Comparação entre visões
    extract_view_comparison()
    
    # 3. Lista produtos disponíveis
    list_available_products()
    
    # 4. NOVO: Comparação completa por produto
    extract_product_comparison()
    
    # Instruções finais
    print("\n")
    print_table_separator()
    print("✅ EXTRAÇÃO CONCLUÍDA")
    print_table_separator()
    print("\n📝 Use os números acima para preencher as tabelas em:")
    print("   METODOLOGIA_PROTOTIPOS_ENRIQUECIDA.md")
    print("\n💡 Copie e cole as tabelas Markdown/LaTeX diretamente no seu TCC!")
    print("")

if __name__ == "__main__":
    main()

