#!/usr/bin/env python3
"""
Extrai apenas a comparação de protótipos por produto (visão full)

Usage:
    source .venv/bin/activate
    python extract_product_comparison.py
"""

import sys
import os
from pathlib import Path

from core.database_v3 import DatabaseManagerV3
from analysis.prototypes_v3 import PrototypeAnalyzerV3
from config import settings_v3
import logging

logging.basicConfig(level=logging.WARNING, format='%(message)s')
log = logging.getLogger(__name__)

def extract_product_comparison():
    """Extrai comparação completa entre produtos (visão full apenas)"""
    print("\n" + "=" * 150)
    print("📊 COMPARAÇÃO DE PROTÓTIPOS POR PRODUTO (VISÃO FULL)")
    print("=" * 150)
    
    db = DatabaseManagerV3()
    db.connect()
    
    try:
        # 1. Busca produtos elegíveis
        print("\n⏳ Buscando produtos elegíveis...")
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
        
        print(f"✓ Encontrados {len(products)} produtos elegíveis (≥20 ligações)\n")
        
        # 2. Para cada produto, calcula métricas
        analyzer = PrototypeAnalyzerV3(db)
        product_metrics = []
        
        for i, row in enumerate(products, 1):
            product_name = row['name']
            print(f"[{i}/{len(products)}] Processando: {product_name[:40]:40s} ... ", end="", flush=True)
            
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
                print(f"❌ (erro: {str(e)[:30]})")
        
        # 3. Tabela comparativa completa
        print("\n")
        print("=" * 150)
        print("TABELA COMPARATIVA: MÉTRICAS DE PROTÓTIPOS POR PRODUTO")
        print("=" * 150)
        
        # Header
        print(f"\n{'Produto':<30} | {'N Ganha':>8} | {'N Perdida':>10} | {'Win Rate':>8} | {'Coesão G':>9} | {'Coesão P':>9} | {'Separação':>10} | {'Silhueta':>9}")
        print(f"{'-'*30}-|{'-'*10}|{'-'*12}|{'-'*10}|{'-'*11}|{'-'*11}|{'-'*12}|{'-'*10}")
        
        # Ordena por silhueta (mais relevante)
        product_metrics.sort(key=lambda x: x['silhouette'], reverse=True)
        
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
            cohesions_g = [pm['cohesion_ganha'] for pm in product_metrics]
            cohesions_p = [pm['cohesion_perdida'] for pm in product_metrics]
            
            print(f"\n📊 Win Rate:")
            print(f"   - Média: {np.mean(win_rates):.1f}%")
            print(f"   - Mediana: {np.median(win_rates):.1f}%")
            print(f"   - Desvio: ±{np.std(win_rates):.1f}%")
            print(f"   - Min/Max: {np.min(win_rates):.1f}% / {np.max(win_rates):.1f}%")
            
            print(f"\n🎯 Coesão Intra-cluster:")
            print(f"   - Ganhas: {np.mean(cohesions_g):.3f} ± {np.std(cohesions_g):.3f}")
            print(f"   - Perdidas: {np.mean(cohesions_p):.3f} ± {np.std(cohesions_p):.3f}")
            
            print(f"\n📏 Separação (distância cosseno):")
            print(f"   - Média: {np.mean(separations):.4f}")
            print(f"   - Mediana: {np.median(separations):.4f}")
            print(f"   - Desvio: ±{np.std(separations):.4f}")
            print(f"   - Min/Max: {np.min(separations):.4f} / {np.max(separations):.4f}")
            
            print(f"\n🏆 Silhueta:")
            print(f"   - Média: {np.mean(silhouettes):.4f}")
            print(f"   - Mediana: {np.median(silhouettes):.4f}")
            print(f"   - Desvio: ±{np.std(silhouettes):.4f}")
            print(f"   - Min/Max: {np.min(silhouettes):.4f} / {np.max(silhouettes):.4f}")
            
            # Ranking dos 5 melhores produtos
            print(f"\n🥇 TOP 5 PRODUTOS (por Silhueta):")
            top5 = sorted(product_metrics, key=lambda x: x['silhouette'], reverse=True)[:5]
            for i, pm in enumerate(top5, 1):
                print(f"   {i}. {pm['name'][:50]:50s} | Sil={pm['silhouette']:.4f} | WR={pm['win_rate']:.1f}% | Sep={pm['separation_distance']:.4f}")
            
            # Bottom 3
            print(f"\n🔻 BOTTOM 3 PRODUTOS (por Silhueta):")
            bottom3 = sorted(product_metrics, key=lambda x: x['silhouette'])[:3]
            for i, pm in enumerate(bottom3, 1):
                print(f"   {i}. {pm['name'][:50]:50s} | Sil={pm['silhouette']:.4f} | WR={pm['win_rate']:.1f}% | Sep={pm['separation_distance']:.4f}")
            
            # Insights de correlação
            print(f"\n💡 ANÁLISE DE CORRELAÇÃO:")
            
            # Correlação Silhueta x Win Rate
            from scipy import stats
            corr_sil_wr, p_val_sil_wr = stats.pearsonr(silhouettes, win_rates)
            print(f"   - Silhueta x Win Rate: r={corr_sil_wr:.3f} (p={p_val_sil_wr:.4f})")
            
            if p_val_sil_wr < 0.05:
                if corr_sil_wr > 0:
                    print(f"     ✓ Correlação POSITIVA significativa: maior separação → melhor desempenho")
                else:
                    print(f"     ⚠️ Correlação NEGATIVA significativa: maior separação → pior desempenho (contra-intuitivo!)")
            else:
                print(f"     ⊘ Correlação NÃO significativa: separação semântica não prediz win rate")
            
            # Comparação High vs Low Silhouette
            median_sil = np.median(silhouettes)
            high_sil = [pm for pm in product_metrics if pm['silhouette'] > median_sil]
            low_sil = [pm for pm in product_metrics if pm['silhouette'] <= median_sil]
            
            if high_sil and low_sil:
                avg_wr_high = np.mean([pm['win_rate'] for pm in high_sil])
                avg_wr_low = np.mean([pm['win_rate'] for pm in low_sil])
                
                print(f"\n   - Produtos com silhueta > mediana ({median_sil:.4f}):")
                print(f"     → Win rate médio: {avg_wr_high:.1f}% (n={len(high_sil)})")
                print(f"   - Produtos com silhueta ≤ mediana:")
                print(f"     → Win rate médio: {avg_wr_low:.1f}% (n={len(low_sil)})")
                
                diff = avg_wr_high - avg_wr_low
                if abs(diff) > 2:
                    if diff > 0:
                        print(f"     ✓ Diferença de +{diff:.1f}pp a favor de produtos com maior separação")
                    else:
                        print(f"     ⚠️ Diferença de {diff:.1f}pp CONTRA produtos com maior separação")
                else:
                    print(f"     ≈ Diferença mínima ({diff:.1f}pp): separação não impacta win rate")
        
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
        print("\\textbf{Produto} & \\textbf{N Ganhas} & \\textbf{N Perdidas} & \\textbf{Win Rate (\\%)} & \\textbf{Separação} & \\textbf{Silhueta} \\\\")
        print("\\midrule")
        
        # Top 5 produtos
        for pm in product_metrics[:5]:
            name = pm['name'].replace('&', '\\&').replace('_', '\\_')  # Escape LaTeX
            print(f"{name} & {pm['n_ganha']} & {pm['n_perdida']} & "
                  f"{pm['win_rate']:.1f} & {pm['separation_distance']:.4f} & {pm['silhouette']:.4f} \\\\")
        
        if len(product_metrics) > 5:
            print("\\midrule")
            # Estatísticas agregadas
            print(f"\\textit{{Média ({len(product_metrics)} produtos)}} & "
                  f"{np.mean([pm['n_ganha'] for pm in product_metrics]):.0f} & "
                  f"{np.mean([pm['n_perdida'] for pm in product_metrics]):.0f} & "
                  f"{np.mean(win_rates):.1f} & "
                  f"{np.mean(separations):.4f} & "
                  f"{np.mean(silhouettes):.4f} \\\\")
        
        print("\\bottomrule")
        print("\\end{tabular}")
        print("\\end{table}")
        
        # 6. Texto para discussão
        print("\n")
        print("=" * 150)
        print("TEXTO PARA DISCUSSÃO NO TCC")
        print("=" * 150)
        
        print(f"""
A análise granular por produto revelou heterogeneidade nos padrões de separação semântica.
Dentre {len(product_metrics)} produtos analisados, o coeficiente de silhueta variou de 
{np.min(silhouettes):.4f} a {np.max(silhouettes):.4f} (μ={np.mean(silhouettes):.4f}, σ={np.std(silhouettes):.4f}).

Os produtos com maior separação foram:
""")
        for i, pm in enumerate(top5[:3], 1):
            print(f"{i}. {pm['name']} (Sil={pm['silhouette']:.4f}, WR={pm['win_rate']:.1f}%)")
        
        print(f"""
A correlação entre silhueta e win rate foi r={corr_sil_wr:.3f} (p={p_val_sil_wr:.4f}), indicando
""", end="")
        
        if p_val_sil_wr < 0.05:
            if corr_sil_wr > 0:
                print("uma associação significativa entre separação semântica e desempenho comercial.")
            else:
                print("uma relação contra-intuitiva que requer investigação adicional.")
        else:
            print("que a separação semântica, por si só, não prediz o sucesso comercial.")
        
        print("""
Este resultado sugere que fatores contextuais específicos de cada produto (complexidade,
ticket médio, maturidade do processo) podem modular a relação entre padrões linguísticos
e desfechos de vendas.
        """)
    
    finally:
        db.close()
        print("\n✅ Análise concluída!")
        print("=" * 150 + "\n")

if __name__ == "__main__":
    extract_product_comparison()

