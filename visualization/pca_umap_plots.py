"""
Visualizações integradas: PCA + UMAP
Mostra UMAP colorido por scores de PCA para interpretação
"""
import logging
from typing import Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from config import settings_v3

log = logging.getLogger(__name__)


class PCAUMAPVisualizer:
    """
    Cria visualizações que integram PCA interpretável com UMAP
    """
    
    def __init__(self, output_dir: str = None):
        self.output_dir = Path(output_dir or settings_v3.V3_PLOTS_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Estilo
        sns.set_style("whitegrid")
        plt.rcParams['figure.dpi'] = 150
    
    def plot_pca_biplot(
        self,
        pca_projections: np.ndarray,
        outcomes: List[str],
        component_interpretations: List[str],
        title: str = "PCA Biplot",
        filename: str = "pca_biplot.png"
    ):
        """
        Cria biplot PCA (PC1 vs PC2) colorido por outcome
        
        Args:
            pca_projections: Projeções PCA (n_samples, n_components)
            outcomes: Lista de outcomes
            component_interpretations: Interpretações de cada PC
            title: Título do gráfico
            filename: Nome do arquivo de saída
        """
        log.info(f"Criando PCA biplot: {filename}")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Converte outcomes para cores
        outcomes_array = np.array(outcomes)
        colors = {'ganha': '#2ecc71', 'perdida': '#e74c3c'}
        
        for outcome, color in colors.items():
            mask = outcomes_array == outcome
            if mask.sum() > 0:
                ax.scatter(
                    pca_projections[mask, 0],
                    pca_projections[mask, 1],
                    c=color,
                    label=outcome.capitalize(),
                    alpha=0.6,
                    s=50,
                    edgecolors='white',
                    linewidth=0.5
                )
        
        # Eixos
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3, linewidth=1)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3, linewidth=1)
        
        # Labels
        pc1_interp = component_interpretations[0] if len(component_interpretations) > 0 else "PC1"
        pc2_interp = component_interpretations[1] if len(component_interpretations) > 1 else "PC2"
        
        ax.set_xlabel(pc1_interp, fontsize=12, fontweight='bold')
        ax.set_ylabel(pc2_interp, fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        ax.legend(loc='upper right', frameon=True, shadow=True)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        log.info(f"✓ Biplot salvo: {filename}")
    
    def plot_umap_colored_by_pc(
        self,
        umap_coords: np.ndarray,
        pc_scores: np.ndarray,
        pc_label: str,
        outcomes: List[str],
        title: str = "UMAP colorido por PC",
        filename: str = "umap_by_pc.png"
    ):
        """
        Cria UMAP colorido por scores de um PC específico
        
        Mostra como clusters visuais no UMAP se relacionam com PCs interpretáveis
        
        Args:
            umap_coords: Coordenadas UMAP (n_samples, 2)
            pc_scores: Scores do PC (n_samples,)
            pc_label: Label interpretado do PC
            outcomes: Lista de outcomes
            title: Título do gráfico
            filename: Nome do arquivo
        """
        log.info(f"Criando UMAP colorido por {pc_label}: {filename}")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Painel 1: UMAP por outcome (referência)
        outcomes_array = np.array(outcomes)
        colors_outcome = {'ganha': '#2ecc71', 'perdida': '#e74c3c'}
        
        for outcome, color in colors_outcome.items():
            mask = outcomes_array == outcome
            if mask.sum() > 0:
                ax1.scatter(
                    umap_coords[mask, 0],
                    umap_coords[mask, 1],
                    c=color,
                    label=outcome.capitalize(),
                    alpha=0.6,
                    s=40,
                    edgecolors='white',
                    linewidth=0.3
                )
        
        ax1.set_xlabel('UMAP 1', fontsize=11)
        ax1.set_ylabel('UMAP 2', fontsize=11)
        ax1.set_title('UMAP por Outcome', fontsize=12, fontweight='bold')
        ax1.legend(loc='best', frameon=True)
        ax1.grid(True, alpha=0.2)
        
        # Painel 2: UMAP colorido por PC score
        scatter = ax2.scatter(
            umap_coords[:, 0],
            umap_coords[:, 1],
            c=pc_scores,
            cmap='RdYlGn',  # Verde = alto, Vermelho = baixo
            alpha=0.7,
            s=40,
            edgecolors='white',
            linewidth=0.3
        )
        
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label(f'{pc_label} Score', fontsize=10)
        
        ax2.set_xlabel('UMAP 1', fontsize=11)
        ax2.set_ylabel('UMAP 2', fontsize=11)
        ax2.set_title(f'UMAP colorido por {pc_label}', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.2)
        
        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        log.info(f"✓ UMAP+PC salvo: {filename}")
    
    def plot_pca_variance_explained(
        self,
        variance_ratios: List[float],
        interpretations: List[str],
        title: str = "Variância Explicada por PC",
        filename: str = "pca_variance.png"
    ):
        """
        Gráfico de barras mostrando variância explicada por cada PC
        """
        log.info(f"Criando gráfico de variância: {filename}")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        n_components = len(variance_ratios)
        cumulative = np.cumsum(variance_ratios)
        
        # Barras
        x = np.arange(n_components)
        bars = ax.bar(
            x,
            variance_ratios,
            color='steelblue',
            alpha=0.7,
            edgecolor='black',
            linewidth=0.5
        )
        
        # Linha cumulativa
        ax2 = ax.twinx()
        ax2.plot(
            x,
            cumulative,
            color='red',
            marker='o',
            linewidth=2,
            label='Cumulativa'
        )
        ax2.set_ylabel('Variância Cumulativa', fontsize=11, color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.set_ylim([0, 1.05])
        
        # Labels individuais
        for i, (bar, var, interp) in enumerate(zip(bars, variance_ratios, interpretations)):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height + 0.005,
                f'{var:.1%}',
                ha='center',
                va='bottom',
                fontsize=9
            )
            
            # Interpretação abreviada (se disponível)
            if interp and "Eixo:" in interp:
                short_interp = interp.replace("Eixo: ", "").replace(" ↔ ", "/")[:25]
                ax.text(
                    bar.get_x() + bar.get_width()/2.,
                    -0.01,
                    short_interp,
                    ha='center',
                    va='top',
                    fontsize=7,
                    rotation=45,
                    style='italic'
                )
        
        ax.set_xlabel('Componente Principal', fontsize=11, fontweight='bold')
        ax.set_ylabel('Variância Explicada', fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels([f'PC{i+1}' for i in range(n_components)])
        ax.set_ylim([0, max(variance_ratios) * 1.15])
        ax.grid(axis='y', alpha=0.3)
        
        # Linha de referência (80% variância)
        ax2.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, linewidth=1)
        ax2.text(n_components-1, 0.82, '80%', color='green', fontsize=9)
        
        ax2.legend(loc='upper left')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        log.info(f"✓ Gráfico de variância salvo: {filename}")
    
    def plot_pc_correlations_heatmap(
        self,
        pc_pattern_correlations: Dict[int, List[Dict]],
        top_k: int = 10,
        title: str = "Correlações PC × Padrões",
        filename: str = "pc_pattern_correlations.png"
    ):
        """
        Heatmap mostrando correlações entre PCs e padrões linguísticos
        
        Args:
            pc_pattern_correlations: {pc_id: [{"pattern": name, "correlation": val}, ...]}
            top_k: Top-k padrões a mostrar por PC
            title: Título
            filename: Arquivo de saída
        """
        log.info(f"Criando heatmap de correlações: {filename}")
        
        # Coleta todos os padrões mencionados
        all_patterns = set()
        for correlations in pc_pattern_correlations.values():
            for corr in correlations[:top_k]:
                all_patterns.add(corr['pattern'])
        
        pattern_list = sorted(all_patterns)
        pc_list = sorted(pc_pattern_correlations.keys())
        
        # Monta matriz de correlações
        corr_matrix = np.zeros((len(pc_list), len(pattern_list)))
        
        for i, pc_id in enumerate(pc_list):
            correlations = {c['pattern']: c['correlation'] 
                          for c in pc_pattern_correlations[pc_id]}
            for j, pattern in enumerate(pattern_list):
                corr_matrix[i, j] = correlations.get(pattern, 0)
        
        # Plot
        fig, ax = plt.subplots(figsize=(14, max(6, len(pc_list) * 0.6)))
        
        sns.heatmap(
            corr_matrix,
            xticklabels=pattern_list,
            yticklabels=[f'PC{pc_id}' for pc_id in pc_list],
            cmap='RdBu_r',
            center=0,
            vmin=-1,
            vmax=1,
            annot=True,
            fmt='.2f',
            linewidths=0.5,
            cbar_kws={'label': 'Correlação de Spearman'},
            ax=ax
        )
        
        ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
        ax.set_xlabel('Padrão Linguístico', fontsize=11, fontweight='bold')
        ax.set_ylabel('Componente Principal', fontsize=11, fontweight='bold')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        log.info(f"✓ Heatmap salvo: {filename}")
    
    def plot_integrated_pca_umap(
        self,
        pca_projections: np.ndarray,
        umap_coords: np.ndarray,
        outcomes: List[str],
        pc1_interpretation: str,
        pc2_interpretation: str,
        title: str = "Análise Integrada: PCA + UMAP",
        filename: str = "integrated_pca_umap.png"
    ):
        """
        Painel 2x2 mostrando PCA e UMAP lado a lado
        
        Layout:
        [PCA biplot]  [UMAP por outcome]
        [UMAP by PC1] [UMAP by PC2]
        """
        log.info(f"Criando visualização integrada: {filename}")
        
        fig = plt.figure(figsize=(16, 14))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        outcomes_array = np.array(outcomes)
        colors = {'ganha': '#2ecc71', 'perdida': '#e74c3c'}
        
        # Painel 1: PCA Biplot
        ax1 = fig.add_subplot(gs[0, 0])
        for outcome, color in colors.items():
            mask = outcomes_array == outcome
            if mask.sum() > 0:
                ax1.scatter(
                    pca_projections[mask, 0],
                    pca_projections[mask, 1],
                    c=color,
                    label=outcome.capitalize(),
                    alpha=0.6,
                    s=40,
                    edgecolors='white',
                    linewidth=0.3
                )
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
        ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
        ax1.set_xlabel(pc1_interpretation, fontsize=10)
        ax1.set_ylabel(pc2_interpretation, fontsize=10)
        ax1.set_title('(A) PCA Biplot', fontsize=11, fontweight='bold', loc='left')
        ax1.legend(loc='best', frameon=True, fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # Painel 2: UMAP por outcome
        ax2 = fig.add_subplot(gs[0, 1])
        for outcome, color in colors.items():
            mask = outcomes_array == outcome
            if mask.sum() > 0:
                ax2.scatter(
                    umap_coords[mask, 0],
                    umap_coords[mask, 1],
                    c=color,
                    label=outcome.capitalize(),
                    alpha=0.6,
                    s=40,
                    edgecolors='white',
                    linewidth=0.3
                )
        ax2.set_xlabel('UMAP 1', fontsize=10)
        ax2.set_ylabel('UMAP 2', fontsize=10)
        ax2.set_title('(B) UMAP por Outcome', fontsize=11, fontweight='bold', loc='left')
        ax2.legend(loc='best', frameon=True, fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # Painel 3: UMAP colorido por PC1
        ax3 = fig.add_subplot(gs[1, 0])
        scatter1 = ax3.scatter(
            umap_coords[:, 0],
            umap_coords[:, 1],
            c=pca_projections[:, 0],
            cmap='RdYlGn',
            alpha=0.7,
            s=40,
            edgecolors='white',
            linewidth=0.3
        )
        cbar1 = plt.colorbar(scatter1, ax=ax3)
        cbar1.set_label('PC1 Score', fontsize=9)
        ax3.set_xlabel('UMAP 1', fontsize=10)
        ax3.set_ylabel('UMAP 2', fontsize=10)
        ax3.set_title(f'(C) UMAP por {pc1_interpretation}', fontsize=11, fontweight='bold', loc='left')
        ax3.grid(True, alpha=0.3)
        
        # Painel 4: UMAP colorido por PC2
        ax4 = fig.add_subplot(gs[1, 1])
        scatter2 = ax4.scatter(
            umap_coords[:, 0],
            umap_coords[:, 1],
            c=pca_projections[:, 1],
            cmap='RdYlGn',
            alpha=0.7,
            s=40,
            edgecolors='white',
            linewidth=0.3
        )
        cbar2 = plt.colorbar(scatter2, ax=ax4)
        cbar2.set_label('PC2 Score', fontsize=9)
        ax4.set_xlabel('UMAP 1', fontsize=10)
        ax4.set_ylabel('UMAP 2', fontsize=10)
        ax4.set_title(f'(D) UMAP por {pc2_interpretation}', fontsize=11, fontweight='bold', loc='left')
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        log.info(f"✓ Visualização integrada salva: {filename}")


