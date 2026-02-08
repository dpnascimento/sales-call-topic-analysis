"""
Gráficos comparativos entre visões de embedding e produtos
"""
import logging
from typing import Dict, List
import numpy as np
from pathlib import Path

from core.database_v3 import DatabaseManagerV3
from config import settings_v3

log = logging.getLogger(__name__)


class ComparisonPlotter:
    """
    Cria gráficos comparativos entre visões e produtos
    """
    
    def __init__(self, db: DatabaseManagerV3):
        self.db = db
    
    def plot_view_comparison_metrics(
        self,
        comparison_data: Dict,
        output_path: str
    ):
        """
        Plota métricas de comparação entre visões
        
        Args:
            comparison_data: Dict com métricas (de EmbeddingViewComparator)
            output_path: Caminho de saída
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            if "by_view" not in comparison_data:
                log.warning("Dados de comparação inválidos")
                return
            
            views = list(comparison_data["by_view"].keys())
            metrics_data = comparison_data["by_view"]
            
            # Extrai métricas
            silhouettes = [metrics_data[v]["silhouette_overall"] for v in views]
            distances = [metrics_data[v]["separation_distance"] for v in views]
            cohesion_ganha = [metrics_data[v]["cohesion_ganha"] for v in views]
            cohesion_perdida = [metrics_data[v]["cohesion_perdida"] for v in views]
            
            # Cria subplots
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # 1. Silhueta
            ax = axes[0, 0]
            bars = ax.bar(views, silhouettes, color=['#3498db', '#2ecc71', '#e74c3c'][:len(views)])
            ax.set_title("Silhueta (Qualidade de Separação)", fontsize=12, fontweight='bold')
            ax.set_ylabel("Silhueta Score", fontsize=10)
            ax.set_ylim(0, max(silhouettes) * 1.2)
            ax.grid(axis='y', alpha=0.3)
            
            # Anota valores
            for bar, val in zip(bars, silhouettes):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.4f}', ha='center', va='bottom', fontsize=9)
            
            # 2. Distância de separação
            ax = axes[0, 1]
            bars = ax.bar(views, distances, color=['#3498db', '#2ecc71', '#e74c3c'][:len(views)])
            ax.set_title("Distância entre Protótipos", fontsize=12, fontweight='bold')
            ax.set_ylabel("Distância Cosseno", fontsize=10)
            ax.set_ylim(0, max(distances) * 1.2)
            ax.grid(axis='y', alpha=0.3)
            
            for bar, val in zip(bars, distances):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.4f}', ha='center', va='bottom', fontsize=9)
            
            # 3. Coesão Ganha
            ax = axes[1, 0]
            bars = ax.bar(views, cohesion_ganha, color='#2ecc71', alpha=0.7)
            ax.set_title("Coesão Intra-Cluster (Ganha)", fontsize=12, fontweight='bold')
            ax.set_ylabel("Coesão Score", fontsize=10)
            ax.set_ylim(0, 1)
            ax.grid(axis='y', alpha=0.3)
            
            for bar, val in zip(bars, cohesion_ganha):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.4f}', ha='center', va='bottom', fontsize=9)
            
            # 4. Coesão Perdida
            ax = axes[1, 1]
            bars = ax.bar(views, cohesion_perdida, color='#e74c3c', alpha=0.7)
            ax.set_title("Coesão Intra-Cluster (Perdida)", fontsize=12, fontweight='bold')
            ax.set_ylabel("Coesão Score", fontsize=10)
            ax.set_ylim(0, 1)
            ax.grid(axis='y', alpha=0.3)
            
            for bar, val in zip(bars, cohesion_perdida):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.4f}', ha='center', va='bottom', fontsize=9)
            
            plt.suptitle("Comparação de Métricas entre Visões de Embedding", 
                        fontsize=16, fontweight='bold', y=1.00)
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            log.info(f"Gráfico de comparação salvo em: {output_path}")
            
        except Exception as e:
            log.error(f"Erro ao plotar comparação de métricas: {e}")
    
    def plot_product_performance_by_view(
        self,
        products_comparison: Dict,
        output_path: str
    ):
        """
        Plota performance de produtos através das visões
        
        Args:
            products_comparison: Dict com comparações por produto
            output_path: Caminho de saída
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            if "product_comparisons" not in products_comparison:
                log.warning("Dados de produtos inválidos")
                return
            
            product_data = products_comparison["product_comparisons"]
            
            # Organiza dados
            products = list(product_data.keys())[:10]  # Top 10
            views = settings_v3.EMBEDDING_VIEWS
            
            # Matriz de silhuetas
            silhouette_matrix = []
            for view in views:
                view_silhouettes = []
                for product in products:
                    if product in product_data:
                        by_view = product_data[product].get("by_view", {})
                        if view in by_view:
                            view_silhouettes.append(by_view[view]["silhouette_overall"])
                        else:
                            view_silhouettes.append(0)
                    else:
                        view_silhouettes.append(0)
                silhouette_matrix.append(view_silhouettes)
            
            # Cria heatmap
            fig, ax = plt.subplots(figsize=(14, 6))
            
            im = ax.imshow(silhouette_matrix, cmap='RdYlGn', aspect='auto', vmin=-0.5, vmax=0.5)
            
            # Configurações dos eixos
            ax.set_xticks(np.arange(len(products)))
            ax.set_yticks(np.arange(len(views)))
            ax.set_xticklabels(products, rotation=45, ha='right', fontsize=9)
            ax.set_yticklabels(views, fontsize=10)
            
            # Anota valores
            for i in range(len(views)):
                for j in range(len(products)):
                    val = silhouette_matrix[i][j]
                    if val != 0:
                        text = ax.text(j, i, f'{val:.3f}',
                                     ha="center", va="center", color="black", fontsize=8)
            
            ax.set_title("Silhueta por Produto e Visão de Embedding", 
                        fontsize=14, fontweight='bold', pad=20)
            ax.set_xlabel("Produtos", fontsize=12)
            ax.set_ylabel("Visões", fontsize=12)
            
            # Colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Silhueta Score', rotation=270, labelpad=20, fontsize=10)
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            log.info(f"Heatmap de performance salvo em: {output_path}")
            
        except Exception as e:
            log.error(f"Erro ao plotar performance por produto: {e}")
    
    def plot_win_rate_by_product(
        self,
        products: List[Dict],
        output_path: str
    ):
        """
        Plota win rate por produto
        
        Args:
            products: Lista de produtos (de db.get_products())
            output_path: Caminho de saída
        """
        try:
            import matplotlib.pyplot as plt
            
            if not products:
                log.warning("Lista de produtos vazia")
                return
            
            # Ordena por win rate
            products = sorted(products, key=lambda x: x['win_rate'], reverse=True)
            
            product_names = [p['product_name'] for p in products[:15]]  # Top 15
            win_rates = [p['win_rate'] * 100 for p in products[:15]]
            n_calls = [p['n_calls'] for p in products[:15]]
            
            # Cria figura
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # 1. Win Rate
            colors = ['#2ecc71' if wr >= 50 else '#e74c3c' for wr in win_rates]
            bars = ax1.barh(product_names, win_rates, color=colors, alpha=0.7)
            ax1.set_xlabel("Win Rate (%)", fontsize=12)
            ax1.set_title("Win Rate por Produto", fontsize=14, fontweight='bold')
            ax1.axvline(50, color='gray', linestyle='--', linewidth=1, alpha=0.5)
            ax1.grid(axis='x', alpha=0.3)
            
            # Anota valores
            for i, (bar, val) in enumerate(zip(bars, win_rates)):
                width = bar.get_width()
                ax1.text(width + 1, bar.get_y() + bar.get_height()/2,
                        f'{val:.1f}%', ha='left', va='center', fontsize=9)
            
            # 2. Número de chamadas
            bars = ax2.barh(product_names, n_calls, color='#3498db', alpha=0.7)
            ax2.set_xlabel("Número de Chamadas", fontsize=12)
            ax2.set_title("Volume de Chamadas por Produto", fontsize=14, fontweight='bold')
            ax2.grid(axis='x', alpha=0.3)
            
            # Anota valores
            for i, (bar, val) in enumerate(zip(bars, n_calls)):
                width = bar.get_width()
                ax2.text(width + (max(n_calls)*0.01), bar.get_y() + bar.get_height()/2,
                        f'{val}', ha='left', va='center', fontsize=9)
            
            plt.suptitle("Análise de Produtos", fontsize=16, fontweight='bold', y=1.00)
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            log.info(f"Gráfico de win rate salvo em: {output_path}")
            
        except Exception as e:
            log.error(f"Erro ao plotar win rate: {e}")
    
    def plot_pattern_category_impact(
        self,
        pattern_analysis: Dict,
        output_path: str,
        top_n: int = 10
    ):
        """
        Plota impacto de categorias de padrões
        
        Args:
            pattern_analysis: Dict com análise de padrões
            output_path: Caminho de saída
            top_n: Número de categorias a mostrar
        """
        try:
            import matplotlib.pyplot as plt
            
            if "categories" not in pattern_analysis:
                log.warning("Análise de padrões sem categorias")
                return
            
            # Agrega impacto por categoria
            category_impacts = {}
            
            for cat_name, patterns in pattern_analysis["categories"].items():
                # Calcula impacto médio dos patterns significativos
                significant = [p for p in patterns if p['significant']]
                if significant:
                    avg_impact = np.mean([abs(p['diff']) for p in significant])
                    n_significant = len(significant)
                    category_impacts[cat_name] = {
                        'avg_impact': avg_impact,
                        'n_patterns': n_significant
                    }
            
            # Ordena por impacto
            sorted_cats = sorted(category_impacts.items(), 
                               key=lambda x: x[1]['avg_impact'], 
                               reverse=True)[:top_n]
            
            cat_names = [c[0] for c in sorted_cats]
            impacts = [c[1]['avg_impact'] for c in sorted_cats]
            n_patterns = [c[1]['n_patterns'] for c in sorted_cats]
            
            # Cria figura
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # 1. Impacto médio
            bars = ax1.barh(cat_names, impacts, color='#3498db', alpha=0.7)
            ax1.set_xlabel("Impacto Médio (%)", fontsize=12)
            ax1.set_title("Impacto Médio por Categoria", fontsize=14, fontweight='bold')
            ax1.grid(axis='x', alpha=0.3)
            
            for bar, val in zip(bars, impacts):
                width = bar.get_width()
                ax1.text(width + 0.3, bar.get_y() + bar.get_height()/2,
                        f'{val:.1f}%', ha='left', va='center', fontsize=9)
            
            # 2. Número de patterns significativos
            bars = ax2.barh(cat_names, n_patterns, color='#2ecc71', alpha=0.7)
            ax2.set_xlabel("Número de Patterns Significativos", fontsize=12)
            ax2.set_title("Patterns Significativos por Categoria", fontsize=14, fontweight='bold')
            ax2.grid(axis='x', alpha=0.3)
            
            for bar, val in zip(bars, n_patterns):
                width = bar.get_width()
                ax2.text(width + 0.3, bar.get_y() + bar.get_height()/2,
                        str(val), ha='left', va='center', fontsize=9)
            
            product = pattern_analysis.get("product_name", "Global")
            plt.suptitle(f"Análise de Categorias de Padrões - {product}", 
                        fontsize=16, fontweight='bold', y=1.00)
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            log.info(f"Gráfico de categorias salvo em: {output_path}")
            
        except Exception as e:
            log.error(f"Erro ao plotar categorias: {e}")
    
    def create_summary_dashboard(
        self,
        view_comparison: Dict,
        products: List[Dict],
        output_path: str
    ):
        """
        Cria dashboard resumido com múltiplos gráficos
        
        Args:
            view_comparison: Comparação de visões
            products: Lista de produtos
            output_path: Caminho de saída
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
            
            fig = plt.figure(figsize=(18, 12))
            gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
            
            # 1. Silhueta por visão (top-left, 2 cols)
            ax1 = fig.add_subplot(gs[0, :2])
            if "by_view" in view_comparison:
                views = list(view_comparison["by_view"].keys())
                silhouettes = [view_comparison["by_view"][v]["silhouette_overall"] for v in views]
                bars = ax1.bar(views, silhouettes, color=['#3498db', '#2ecc71', '#e74c3c'][:len(views)])
                ax1.set_title("Silhueta por Visão", fontsize=12, fontweight='bold')
                ax1.set_ylabel("Silhueta Score")
                ax1.grid(axis='y', alpha=0.3)
                for bar, val in zip(bars, silhouettes):
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height,
                           f'{val:.4f}', ha='center', va='bottom')
            
            # 2. Best view count (top-right)
            ax2 = fig.add_subplot(gs[0, 2])
            if "best_view_counts" in view_comparison:
                counts = view_comparison["best_view_counts"]
                ax2.pie(counts.values(), labels=counts.keys(), autopct='%1.1f%%',
                       colors=['#3498db', '#2ecc71', '#e74c3c'][:len(counts)])
                ax2.set_title("Melhor Visão por Produto", fontsize=12, fontweight='bold')
            
            # 3. Top products by win rate (middle row)
            ax3 = fig.add_subplot(gs[1, :])
            if products:
                top_products = sorted(products, key=lambda x: x['win_rate'], reverse=True)[:10]
                names = [p['product_name'][:20] for p in top_products]
                win_rates = [p['win_rate'] * 100 for p in top_products]
                colors = ['#2ecc71' if wr >= 50 else '#e74c3c' for wr in win_rates]
                bars = ax3.barh(names, win_rates, color=colors, alpha=0.7)
                ax3.set_xlabel("Win Rate (%)")
                ax3.set_title("Top 10 Produtos por Win Rate", fontsize=12, fontweight='bold')
                ax3.axvline(50, color='gray', linestyle='--', alpha=0.5)
                ax3.grid(axis='x', alpha=0.3)
            
            # 4. Volume distribution (bottom-left)
            ax4 = fig.add_subplot(gs[2, 0])
            if products:
                volumes = [p['n_calls'] for p in products]
                ax4.hist(volumes, bins=20, color='#3498db', alpha=0.7, edgecolor='black')
                ax4.set_xlabel("Número de Chamadas")
                ax4.set_ylabel("Frequência")
                ax4.set_title("Distribuição de Volume", fontsize=12, fontweight='bold')
                ax4.grid(axis='y', alpha=0.3)
            
            # 5. Win rate distribution (bottom-middle)
            ax5 = fig.add_subplot(gs[2, 1])
            if products:
                win_rates_dist = [p['win_rate'] * 100 for p in products]
                ax5.hist(win_rates_dist, bins=20, color='#2ecc71', alpha=0.7, edgecolor='black')
                ax5.set_xlabel("Win Rate (%)")
                ax5.set_ylabel("Frequência")
                ax5.set_title("Distribuição de Win Rate", fontsize=12, fontweight='bold')
                ax5.axvline(50, color='red', linestyle='--', alpha=0.5)
                ax5.grid(axis='y', alpha=0.3)
            
            # 6. Summary stats (bottom-right)
            ax6 = fig.add_subplot(gs[2, 2])
            ax6.axis('off')
            
            # Calcula estatísticas
            if products and "by_view" in view_comparison:
                n_products = len(products)
                avg_win_rate = np.mean([p['win_rate'] for p in products]) * 100
                total_calls = sum([p['n_calls'] for p in products])
                best_view = view_comparison.get("best_view", "N/A")
                best_silhouette = max([view_comparison["by_view"][v]["silhouette_overall"] 
                                      for v in view_comparison["by_view"].keys()])
                
                stats_text = f"""
ESTATÍSTICAS GERAIS

Produtos Analisados: {n_products}
Total de Chamadas: {total_calls}

Win Rate Médio: {avg_win_rate:.1f}%

Melhor Visão: {best_view.upper()}
Melhor Silhueta: {best_silhouette:.4f}
                """
                
                ax6.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                        verticalalignment='center')
            
            plt.suptitle("Dashboard de Análise", fontsize=18, fontweight='bold', y=0.98)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            log.info(f"Dashboard salvo em: {output_path}")
            
        except Exception as e:
            log.error(f"Erro ao criar dashboard: {e}")
    
    def plot_centroid_distance_heatmap(
        self,
        output_path: str,
        embedding_views: List[str] = None
    ):
        """
        Plota heatmap de distâncias entre centroides de diferentes grupos:
        - Vendas ganhas (agent)
        - Vendas ganhas (client)
        - Vendas perdidas (agent)
        - Vendas perdidas (client)
        
        Args:
            output_path: Caminho de saída para o arquivo
            embedding_views: Lista de views a analisar (default: ['agent', 'client'])
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            from core.embeddings_v3 import centroid, cosine_distance, from_pgvector
            
            if embedding_views is None:
                embedding_views = ['agent', 'client']
            
            log.info(f"Calculando centroides para heatmap de distâncias...")
            
            # Dicionário para armazenar centroides
            centroids = {}
            group_labels = []
            
            # Para cada combinação de view e outcome
            for view in embedding_views:
                for outcome in ['ganha', 'perdida']:
                    group_name = f"{outcome.title()} ({view.title()})"
                    group_labels.append(group_name)
                    
                    # Define coluna de embedding baseado na view (usa settings_v3)
                    from config import settings_v3
                    emb_col = settings_v3.EMBEDDING_COLUMN_MAP[view]
                    valid_col = settings_v3.EMBEDDING_VALID_MAP[view]
                    
                    # Busca embeddings do banco (usa mesma estrutura do get_all_embeddings_by_view)
                    query = f"""
                        SELECT DISTINCT ON (e.call_id)
                            e.{emb_col}::text AS embedding_text
                        FROM public.call_embeddings_v2 e
                        JOIN public.call_records cr ON cr.call_id = e.call_id
                        JOIN public.call_outcomes co ON co.deal_id = cr.deal_id
                        WHERE e.{valid_col} = TRUE
                        AND e.{emb_col} IS NOT NULL
                        AND cr.call_duration >= 10
                        AND co.outcome = %s
                        ORDER BY e.call_id, co.outcome_date DESC NULLS LAST, cr.recorded_at
                    """
                    
                    with self.db.get_cursor() as cur:  # Usa dict_row padrão
                        cur.execute(query, (outcome,))
                        rows = cur.fetchall()
                        
                        if rows:
                            embeddings = [from_pgvector(row['embedding_text']) for row in rows]
                            embeddings = [e for e in embeddings if e.size > 0]
                            
                            if embeddings:
                                centroid_vec = centroid(embeddings)
                                centroids[group_name] = centroid_vec
                                log.info(f"  ✓ {group_name}: {len(embeddings)} embeddings")
                            else:
                                log.warning(f"  ⚠ {group_name}: sem embeddings válidos")
                        else:
                            log.warning(f"  ⚠ {group_name}: sem dados")
            
            if len(centroids) < 2:
                log.warning("Centroides insuficientes para criar heatmap")
                return
            
            # Calcula matriz de distâncias
            n_groups = len(group_labels)
            distance_matrix = np.zeros((n_groups, n_groups))
            
            for i, label_i in enumerate(group_labels):
                for j, label_j in enumerate(group_labels):
                    if label_i in centroids and label_j in centroids:
                        if i == j:
                            distance_matrix[i, j] = 0.0
                        else:
                            dist = cosine_distance(centroids[label_i], centroids[label_j])
                            distance_matrix[i, j] = dist
                    else:
                        distance_matrix[i, j] = np.nan
            
            # Cria heatmap
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Usa seaborn para heatmap mais bonito
            mask = np.isnan(distance_matrix)
            
            sns.heatmap(
                distance_matrix,
                annot=True,
                fmt='.4f',
                cmap='RdYlGn_r',  # Invertido: vermelho = distante, verde = próximo
                xticklabels=group_labels,
                yticklabels=group_labels,
                square=True,
                linewidths=1,
                cbar_kws={'label': 'Distância Cosseno'},
                vmin=0,
                vmax=1,
                mask=mask,
                ax=ax
            )
            
            ax.set_title(
                'Matriz de Distâncias entre Centroides\n'
                'Vendas Ganhas vs Perdidas | Vendedor vs Cliente',
                fontsize=14,
                fontweight='bold',
                pad=20
            )
            
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            log.info(f"✓ Heatmap de centroides salvo em: {output_path}")
            
            # Log das distâncias interessantes
            log.info("\n📊 Distâncias principais:")
            
            # Distância entre ganhas e perdidas (mesmo view)
            for view in embedding_views:
                label_ganha = f"Ganha ({view.title()})"
                label_perdida = f"Perdida ({view.title()})"
                
                if label_ganha in centroids and label_perdida in centroids:
                    i = group_labels.index(label_ganha)
                    j = group_labels.index(label_perdida)
                    dist = distance_matrix[i, j]
                    log.info(f"  • {view.title()}: Ganha ↔ Perdida = {dist:.4f}")
            
            # Distância entre agent e client (mesmo outcome)
            if 'agent' in embedding_views and 'client' in embedding_views:
                for outcome in ['Ganha', 'Perdida']:
                    label_agent = f"{outcome} (Agent)"
                    label_client = f"{outcome} (Client)"
                    
                    if label_agent in centroids and label_client in centroids:
                        i = group_labels.index(label_agent)
                        j = group_labels.index(label_client)
                        dist = distance_matrix[i, j]
                        log.info(f"  • {outcome}: Agent ↔ Client = {dist:.4f}")
            
        except Exception as e:
            log.error(f"Erro ao plotar heatmap de centroides: {e}", exc_info=True)

