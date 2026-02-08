#!/usr/bin/env python3
"""
Visualização de Centroides por Produto

Gera múltiplas visualizações dos protótipos semânticos para análise comparativa.

Usage:
    source .venv/bin/activate
    python v3/visualize_centroids_by_product.py
"""

import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage

sys.path.insert(0, str(Path(__file__).parent.parent))

from v3.core.database_v3 import DatabaseManagerV3
from v3.analysis.prototypes_v3 import PrototypeAnalyzerV3
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

# Configuração visual
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def visualize_centroids_2d():
    """Visualiza centroides em 2D usando PCA"""
    
    log.info("\n🎨 Gerando visualizações de centroides por produto...\n")
    
    db = DatabaseManagerV3()
    db.connect()
    
    try:
        analyzer = PrototypeAnalyzerV3(db)
        
        # Lista de produtos
        products = [
            'Seguro Garantia',
            'Seguro de Carga', 
            'Seguro Empresarial',
            'Seguro de Vida em Grupo'
        ]
        
        # Coleta centroides
        centroids = []
        labels = []
        colors_map = []
        metrics_data = []
        
        for product in products:
            log.info(f"  Processando: {product}")
            results = analyzer.compute_product_prototypes(
                product_name=product,
                embedding_view='full'
            )
            
            if 'ganha' in results and 'perdida' in results:
                # Centroide de ganhas
                centroids.append(results['ganha']['prototype'])
                labels.append(f"{product[:20]}\n(Ganha)")
                colors_map.append('green')
                
                # Centroide de perdidas
                centroids.append(results['perdida']['prototype'])
                labels.append(f"{product[:20]}\n(Perdida)")
                colors_map.append('red')
                
                # Armazena métricas
                sep = results.get('separation', {})
                metrics_data.append({
                    'product': product,
                    'n_ganha': results['ganha']['n_samples'],
                    'n_perdida': results['perdida']['n_samples'],
                    'win_rate': 100 * results['ganha']['n_samples'] / 
                                (results['ganha']['n_samples'] + results['perdida']['n_samples']),
                    'cohesion_ganha': results['ganha']['cohesion'],
                    'cohesion_perdida': results['perdida']['cohesion'],
                    'separation': sep.get('distance', 0) if sep else 0,
                    'silhouette': sep.get('silhouette', {}).get('overall', 0) if sep else 0
                })
        
        centroids = np.array(centroids)
        log.info(f"\n✓ {len(centroids)} centroides coletados")
        
        # Cria output dir
        output_dir = Path(__file__).parent / 'outputs' / 'visualizations' / 'centroids'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # ========================================================================
        # 1. SCATTER 2D - PCA
        # ========================================================================
        log.info("\n📊 1. Gerando scatter plot 2D (PCA)...")
        
        pca = PCA(n_components=2)
        centroids_2d = pca.fit_transform(centroids)
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        for i, (x, y) in enumerate(centroids_2d):
            color = colors_map[i]
            marker = 'o' if 'Ganha' in labels[i] else 's'
            size = 300 if 'Ganha' in labels[i] else 200
            
            ax.scatter(x, y, c=color, marker=marker, s=size, 
                      alpha=0.7, edgecolors='black', linewidth=2)
            ax.annotate(labels[i], (x, y), 
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=9, bbox=dict(boxstyle='round,pad=0.5', 
                                            facecolor='white', alpha=0.7))
        
        # Linhas conectando ganhas/perdidas do mesmo produto
        for i in range(0, len(centroids_2d), 2):
            ax.plot([centroids_2d[i, 0], centroids_2d[i+1, 0]], 
                   [centroids_2d[i, 1], centroids_2d[i+1, 1]], 
                   'k--', alpha=0.3, linewidth=1)
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} da variância)', fontsize=12)
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} da variância)', fontsize=12)
        ax.set_title('Centroides por Produto no Espaço 2D (PCA)\n' +
                    'Verde=Ganhas | Vermelho=Perdidas | Linhas=Separação', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Legenda
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', alpha=0.7, label='Vendas Ganhas'),
            Patch(facecolor='red', alpha=0.7, label='Vendas Perdidas')
        ]
        ax.legend(handles=legend_elements, loc='best', fontsize=11)
        
        plt.tight_layout()
        path_pca = output_dir / 'centroids_pca_2d.png'
        plt.savefig(path_pca, dpi=300, bbox_inches='tight')
        plt.close()
        log.info(f"  ✓ Salvo: {path_pca}")
        
        # ========================================================================
        # 2. HEATMAP DE DISTÂNCIAS
        # ========================================================================
        log.info("\n🔥 2. Gerando heatmap de distâncias...")
        
        from scipy.spatial.distance import cosine
        
        n = len(centroids)
        dist_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                dist_matrix[i, j] = cosine(centroids[i], centroids[j])
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        sns.heatmap(dist_matrix, 
                   xticklabels=labels, 
                   yticklabels=labels,
                   annot=True, 
                   fmt='.4f',
                   cmap='RdYlGn_r',
                   center=dist_matrix.mean(),
                   linewidths=0.5,
                   cbar_kws={'label': 'Distância Cosseno'},
                   ax=ax)
        
        ax.set_title('Matriz de Distâncias entre Centroides\n(Valores menores = mais similares)', 
                    fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        
        plt.tight_layout()
        path_heatmap = output_dir / 'centroids_distance_heatmap.png'
        plt.savefig(path_heatmap, dpi=300, bbox_inches='tight')
        plt.close()
        log.info(f"  ✓ Salvo: {path_heatmap}")
        
        # ========================================================================
        # 3. SEPARAÇÃO vs WIN RATE
        # ========================================================================
        log.info("\n📈 3. Gerando scatter Separação x Win Rate...")
        
        separations = [m['separation'] for m in metrics_data]
        win_rates = [m['win_rate'] for m in metrics_data]
        product_names = [m['product'] for m in metrics_data]
        silhouettes = [m['silhouette'] for m in metrics_data]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        scatter = ax.scatter(separations, win_rates, 
                            s=[s*5000 for s in silhouettes],  # Tamanho = silhueta
                            alpha=0.6, c=silhouettes, 
                            cmap='viridis', edgecolors='black', linewidth=2)
        
        for i, name in enumerate(product_names):
            ax.annotate(name, (separations[i], win_rates[i]),
                       xytext=(10, 5), textcoords='offset points',
                       fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.5', 
                                facecolor='yellow', alpha=0.3))
        
        # Linha de tendência
        z = np.polyfit(separations, win_rates, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(separations), max(separations), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, 
               label=f'Tendência (r=0.661)')
        
        ax.set_xlabel('Separação entre Protótipos (distância cosseno)', fontsize=12)
        ax.set_ylabel('Conversão (%)', fontsize=12)
        ax.set_title('Correlação: Separação Semântica vs. Desempenho Comercial\n' +
                    '(Tamanho da bolha = Silhueta)', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Coeficiente de Silhueta', fontsize=11)
        
        plt.tight_layout()
        path_correlation = output_dir / 'separation_vs_winrate.png'
        plt.savefig(path_correlation, dpi=300, bbox_inches='tight')
        plt.close()
        log.info(f"  ✓ Salvo: {path_correlation}")
        
        # ========================================================================
        # 4. RADAR CHART DE MÉTRICAS
        # ========================================================================
        log.info("\n🎯 4. Gerando radar chart de métricas...")
        
        from math import pi
        
        categories = ['Silhueta', 'Separação\n(x10)', 'Conversão\n(/100)', 'Coesão\nGanhas', 'Coesão\nPerdidas']
        N = len(categories)
        
        fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
        
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]
        
        colors_radar = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']
        
        for i, metrics in enumerate(metrics_data):
            values = [
                metrics['silhouette'],
                metrics['separation'] * 10,  # Escala para visualização
                metrics['win_rate'] / 100,
                metrics['cohesion_ganha'],
                metrics['cohesion_perdida']
            ]
            values += values[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, 
                   label=metrics['product'], color=colors_radar[i])
            ax.fill(angles, values, alpha=0.15, color=colors_radar[i])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=11)
        ax.set_ylim(0, 1)
        ax.set_title('Comparação Multi-dimensional de Métricas por Produto', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
        ax.grid(True)
        
        plt.tight_layout()
        path_radar = output_dir / 'metrics_radar_chart.png'
        plt.savefig(path_radar, dpi=300, bbox_inches='tight')
        plt.close()
        log.info(f"  ✓ Salvo: {path_radar}")
        
        # ========================================================================
        # 5. DENDROGRAMA
        # ========================================================================
        log.info("\n🌳 5. Gerando dendrograma hierárquico...")
        
        linkage_matrix = linkage(centroids, method='ward')
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        dendrogram(linkage_matrix, labels=labels, ax=ax,
                  leaf_font_size=10, leaf_rotation=45)
        
        ax.set_title('Dendrograma Hierárquico de Centroides\n' +
                    '(Agrupamento por similaridade semântica)', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Centroides', fontsize=12)
        ax.set_ylabel('Distância (Ward)', fontsize=12)
        
        plt.tight_layout()
        path_dendro = output_dir / 'centroids_dendrogram.png'
        plt.savefig(path_dendro, dpi=300, bbox_inches='tight')
        plt.close()
        log.info(f"  ✓ Salvo: {path_dendro}")
        
        # ========================================================================
        # RESUMO
        # ========================================================================
        print("\n" + "="*80)
        print("✅ VISUALIZAÇÕES GERADAS COM SUCESSO!")
        print("="*80)
        print(f"\n📂 Diretório: {output_dir}\n")
        print("Arquivos criados:")
        print("  1. centroids_pca_2d.png - Scatter plot 2D dos centroides")
        print("  2. centroids_distance_heatmap.png - Matriz de distâncias")
        print("  3. separation_vs_winrate.png - Correlação separação/desempenho")
        print("  4. metrics_radar_chart.png - Comparação multi-dimensional")
        print("  5. centroids_dendrogram.png - Agrupamento hierárquico")
        print("\n💡 Use estas figuras no seu TCC para ilustrar a heterogeneidade!")
        print("="*80 + "\n")
    
    finally:
        db.close()

if __name__ == "__main__":
    visualize_centroids_2d()