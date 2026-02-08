"""
Visualizações UMAP para diferentes visões de embedding
"""
import logging
from typing import List, Dict, Optional
import numpy as np
from pathlib import Path

from core.database_v3 import DatabaseManagerV3
from core.embeddings_v3 import from_pgvector
from config import settings_v3

log = logging.getLogger(__name__)


class UMAPVisualizer:
    """
    Cria visualizações UMAP para diferentes visões de embedding
    """
    
    def __init__(self, db: DatabaseManagerV3):
        self.db = db
    
    def create_umap_by_view(
        self,
        embedding_view: str = "full",
        sample_size: int = None
    ) -> Dict:
        """
        Cria visualização UMAP para uma visão específica
        
        Args:
            embedding_view: 'full', 'agent' ou 'client'
            sample_size: Limite de amostras (None = todas)
            
        Returns:
            Dict com coordenadas UMAP e metadados
        """
        log.info(f"Criando UMAP - visão: {embedding_view}")
        
        if sample_size is None:
            sample_size = settings_v3.UMAP_SAMPLE_SIZE
        
        # Busca embeddings
        rows = self.db.get_all_embeddings_by_view(
            embedding_view=embedding_view,
            limit=sample_size
        )
        
        if not rows or len(rows) < 10:
            log.warning(f"Dados insuficientes para UMAP ({len(rows)} amostras)")
            return {}
        
        # Converte embeddings
        embeddings = []
        metadata = []
        
        for row in rows:
            vec = from_pgvector(row['embedding_text'])
            if vec.size > 0:
                embeddings.append(vec)
                metadata.append({
                    'call_id': row['call_id'],
                    'outcome': row['outcome'],
                    'product_name': row['product_name']
                })
        
        if len(embeddings) < 10:
            log.warning("Embeddings válidos insuficientes para UMAP")
            return {}
        
        # Aplica UMAP
        try:
            import umap
            
            reducer = umap.UMAP(
                n_neighbors=settings_v3.UMAP_N_NEIGHBORS,
                min_dist=settings_v3.UMAP_MIN_DIST,
                metric=settings_v3.UMAP_METRIC,
                random_state=42
            )
            
            log.info(f"Aplicando UMAP em {len(embeddings)} embeddings...")
            embedding_2d = reducer.fit_transform(np.array(embeddings))
            
            log.info(f"✓ UMAP concluído: {embedding_2d.shape}")
            
        except Exception as e:
            log.error(f"Erro ao aplicar UMAP: {e}")
            return {}
        
        return {
            "embedding_view": embedding_view,
            "n_samples": len(embeddings),
            "coordinates": embedding_2d,
            "metadata": metadata
        }
    
    def create_umap_by_product(
        self,
        product_name: str,
        embedding_view: str = "full"
    ) -> Dict:
        """
        Cria UMAP específico para um produto
        
        Args:
            product_name: Nome do produto
            embedding_view: Visão de embedding
            
        Returns:
            Dict com coordenadas UMAP
        """
        log.info(f"Criando UMAP - produto: {product_name}, visão: {embedding_view}")
        
        # Busca embeddings do produto
        rows = self.db.get_all_embeddings_by_view(
            embedding_view=embedding_view,
            product_name=product_name
        )
        
        if not rows or len(rows) < 10:
            log.warning(f"Dados insuficientes para produto '{product_name}'")
            return {}
        
        # Converte embeddings
        embeddings = []
        metadata = []
        
        for row in rows:
            vec = from_pgvector(row['embedding_text'])
            if vec.size > 0:
                embeddings.append(vec)
                metadata.append({
                    'call_id': row['call_id'],
                    'outcome': row['outcome']
                })
        
        if len(embeddings) < 10:
            return {}
        
        # Aplica UMAP
        try:
            import umap
            
            reducer = umap.UMAP(
                n_neighbors=min(15, len(embeddings) - 1),
                min_dist=settings_v3.UMAP_MIN_DIST,
                metric=settings_v3.UMAP_METRIC,
                random_state=42
            )
            
            embedding_2d = reducer.fit_transform(np.array(embeddings))
            
        except Exception as e:
            log.error(f"Erro ao aplicar UMAP para produto: {e}")
            return {}
        
        return {
            "product_name": product_name,
            "embedding_view": embedding_view,
            "n_samples": len(embeddings),
            "coordinates": embedding_2d,
            "metadata": metadata
        }
    
    def plot_umap(
        self,
        umap_data: Dict,
        output_path: str,
        title: str = None
    ):
        """
        Plota e salva visualização UMAP
        
        Args:
            umap_data: Dados retornados por create_umap_*
            output_path: Caminho do arquivo de saída
            title: Título do gráfico
        """
        if not umap_data or "coordinates" not in umap_data:
            log.warning("Dados UMAP inválidos")
            return
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            coords = umap_data["coordinates"]
            metadata = umap_data["metadata"]
            
            # Cores por outcome
            colors = ['#2ecc71' if m['outcome'] == 'ganha' else '#e74c3c' for m in metadata]
            
            # Cria figura
            plt.figure(figsize=(12, 8))
            
            # Scatter plot
            plt.scatter(
                coords[:, 0],
                coords[:, 1],
                c=colors,
                alpha=0.6,
                s=50,
                edgecolors='white',
                linewidth=0.5
            )
            
            # Título
            if title is None:
                view = umap_data.get("embedding_view", "unknown")
                title = f"UMAP - Visão: {view}"
                
                if "product_name" in umap_data:
                    title += f" | Produto: {umap_data['product_name']}"
            
            plt.title(title, fontsize=16, fontweight='bold')
            plt.xlabel("UMAP Dimensão 1", fontsize=12)
            plt.ylabel("UMAP Dimensão 2", fontsize=12)
            
            # Legenda
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#2ecc71', label='Ganha'),
                Patch(facecolor='#e74c3c', label='Perdida')
            ]
            plt.legend(handles=legend_elements, loc='best', fontsize=10)
            
            # Grid
            plt.grid(True, alpha=0.3)
            
            # Salva
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            log.info(f"UMAP salvo em: {output_path}")
            
        except Exception as e:
            log.error(f"Erro ao plotar UMAP: {e}")
    
    def create_comparative_umap(
        self,
        views: List[str] = None,
        sample_size: int = None
    ) -> Dict:
        """
        Cria UMAPs comparativos para múltiplas visões
        
        Args:
            views: Lista de visões (None = todas)
            sample_size: Limite de amostras
            
        Returns:
            Dict mapeando visão -> dados UMAP
        """
        if views is None:
            views = settings_v3.EMBEDDING_VIEWS
        
        log.info(f"Criando UMAPs comparativos para visões: {views}")
        
        results = {}
        
        for view in views:
            try:
                umap_data = self.create_umap_by_view(view, sample_size)
                if umap_data:
                    results[view] = umap_data
            except Exception as e:
                log.error(f"Erro ao criar UMAP para visão '{view}': {e}")
        
        return results
    
    def plot_comparative_umaps(
        self,
        comparative_data: Dict,
        output_path: str
    ):
        """
        Plota UMAPs comparativos lado a lado
        
        Args:
            comparative_data: Dict retornado por create_comparative_umap
            output_path: Caminho do arquivo de saída
        """
        if not comparative_data:
            log.warning("Dados comparativos vazios")
            return
        
        try:
            import matplotlib.pyplot as plt
            
            n_views = len(comparative_data)
            fig, axes = plt.subplots(1, n_views, figsize=(6*n_views, 5))
            
            if n_views == 1:
                axes = [axes]
            
            for ax, (view, data) in zip(axes, comparative_data.items()):
                coords = data["coordinates"]
                metadata = data["metadata"]
                
                colors = ['#2ecc71' if m['outcome'] == 'ganha' else '#e74c3c' for m in metadata]
                
                ax.scatter(
                    coords[:, 0],
                    coords[:, 1],
                    c=colors,
                    alpha=0.6,
                    s=30,
                    edgecolors='white',
                    linewidth=0.5
                )
                
                ax.set_title(f"Visão: {view.upper()}", fontsize=14, fontweight='bold')
                ax.set_xlabel("UMAP 1", fontsize=10)
                ax.set_ylabel("UMAP 2", fontsize=10)
                ax.grid(True, alpha=0.3)
            
            # Legenda global
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#2ecc71', label='Ganha'),
                Patch(facecolor='#e74c3c', label='Perdida')
            ]
            fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2, fontsize=10)
            
            plt.suptitle("Comparação de Visões de Embedding (UMAP)", fontsize=16, fontweight='bold', y=1.02)
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            log.info(f"UMAP comparativo salvo em: {output_path}")
            
        except Exception as e:
            log.error(f"Erro ao plotar UMAPs comparativos: {e}")
    
    def create_product_grid_umap(
        self,
        embedding_view: str = "full",
        max_products: int = 9
    ) -> Dict:
        """
        Cria grid de UMAPs para múltiplos produtos
        
        Args:
            embedding_view: Visão de embedding
            max_products: Máximo de produtos no grid
            
        Returns:
            Dict mapeando produto -> dados UMAP
        """
        log.info(f"Criando grid de UMAPs - visão: {embedding_view}")
        
        # Busca produtos
        products = self.db.get_products(min_calls=settings_v3.MIN_CALLS_PER_PRODUCT)
        
        results = {}
        
        for i, product in enumerate(products[:max_products]):
            product_name = product['product_name']
            
            try:
                umap_data = self.create_umap_by_product(product_name, embedding_view)
                if umap_data:
                    results[product_name] = umap_data
            except Exception as e:
                log.error(f"Erro ao criar UMAP para produto '{product_name}': {e}")
        
        return results
    
    def plot_product_grid(
        self,
        grid_data: Dict,
        output_path: str
    ):
        """
        Plota grid de produtos
        
        Args:
            grid_data: Dict retornado por create_product_grid_umap
            output_path: Caminho de saída
        """
        if not grid_data:
            log.warning("Grid de produtos vazio")
            return
        
        try:
            import matplotlib.pyplot as plt
            import math
            
            n_products = len(grid_data)
            ncols = min(3, n_products)
            nrows = math.ceil(n_products / ncols)
            
            fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows))
            axes = axes.flatten() if nrows * ncols > 1 else [axes]
            
            for ax, (product_name, data) in zip(axes, grid_data.items()):
                coords = data["coordinates"]
                metadata = data["metadata"]
                
                colors = ['#2ecc71' if m['outcome'] == 'ganha' else '#e74c3c' for m in metadata]
                
                ax.scatter(
                    coords[:, 0],
                    coords[:, 1],
                    c=colors,
                    alpha=0.6,
                    s=30,
                    edgecolors='white',
                    linewidth=0.5
                )
                
                ax.set_title(product_name, fontsize=12, fontweight='bold')
                ax.set_xlabel("UMAP 1", fontsize=9)
                ax.set_ylabel("UMAP 2", fontsize=9)
                ax.grid(True, alpha=0.3)
            
            # Remove eixos extras
            for i in range(n_products, len(axes)):
                fig.delaxes(axes[i])
            
            # Legenda
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#2ecc71', label='Ganha'),
                Patch(facecolor='#e74c3c', label='Perdida')
            ]
            fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=10)
            
            view = list(grid_data.values())[0].get("embedding_view", "unknown")
            plt.suptitle(f"Produtos por UMAP - Visão: {view.upper()}", fontsize=16, fontweight='bold', y=1.02)
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            log.info(f"Grid de produtos salvo em: {output_path}")
            
        except Exception as e:
            log.error(f"Erro ao plotar grid de produtos: {e}")

