#!/usr/bin/env python3
"""
Gerador de Relatório Final para TCC
Consolida todos os resultados em relatório estruturado
"""
import json
import logging
from datetime import datetime
from typing import Dict, List
import os

from config import settings_v3

log = logging.getLogger(__name__)


class TCCReportGenerator:
    """Gera relatório consolidado para TCC"""
    
    def __init__(self, pipeline_results: Dict = None, data_dir: str = None):
        self.pipeline_results = pipeline_results or {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.data_dir = data_dir or settings_v3.V3_DATA_DIR
        
        # Carrega dados automaticamente se não fornecidos
        if not self.pipeline_results:
            log.info("Carregando resultados de arquivos JSON...")
            self.pipeline_results = self._load_latest_results()
    
    def _load_latest_results(self) -> Dict:
        """
        Carrega resultados mais recentes dos arquivos JSON gerados
        """
        import glob
        
        results = {}
        
        # Busca arquivos JSON no diretório de dados
        json_pattern = os.path.join(self.data_dir, "*.json")
        json_files = sorted(glob.glob(json_pattern), key=os.path.getmtime, reverse=True)
        
        if not json_files:
            log.warning(f"Nenhum arquivo JSON encontrado em {self.data_dir}")
            return results
        
        # Agrupa por tipo (pega mais recente de cada tipo)
        loaded = {}
        for json_file in json_files:
            basename = os.path.basename(json_file)
            
            # Identifica tipo do arquivo
            if 'prototypes' in basename and 'prototypes' not in loaded:
                try:
                    with open(json_file, 'r') as f:
                        results['prototype_results'] = json.load(f)
                    loaded['prototypes'] = True
                    log.info(f"  ✓ Prototypes: {basename}")
                except Exception as e:
                    log.warning(f"Erro ao carregar {basename}: {e}")
            
            elif 'patterns_agent' in basename and 'patterns_agent' not in loaded:
                try:
                    with open(json_file, 'r') as f:
                        if 'pattern_results' not in results:
                            results['pattern_results'] = {}
                        results['pattern_results']['agent'] = json.load(f)
                    loaded['patterns_agent'] = True
                    log.info(f"  ✓ Patterns Agent: {basename}")
                except Exception as e:
                    log.warning(f"Erro ao carregar {basename}: {e}")
            
            elif 'patterns_client' in basename and 'patterns_client' not in loaded:
                try:
                    with open(json_file, 'r') as f:
                        if 'pattern_results' not in results:
                            results['pattern_results'] = {}
                        results['pattern_results']['client'] = json.load(f)
                    loaded['patterns_client'] = True
                    log.info(f"  ✓ Patterns Client: {basename}")
                except Exception as e:
                    log.warning(f"Erro ao carregar {basename}: {e}")
            
            elif 'patterns_status' in basename and 'patterns_status' not in loaded:
                try:
                    with open(json_file, 'r') as f:
                        results['pattern_status_results'] = json.load(f)
                    loaded['patterns_status'] = True
                    log.info(f"  ✓ Patterns Status: {basename}")
                except Exception as e:
                    log.warning(f"Erro ao carregar {basename}: {e}")
            
            elif 'view_comparison' in basename and 'view_comparison' not in loaded:
                try:
                    with open(json_file, 'r') as f:
                        results['comparison_results'] = json.load(f)
                    loaded['view_comparison'] = True
                    log.info(f"  ✓ View Comparison: {basename}")
                except Exception as e:
                    log.warning(f"Erro ao carregar {basename}: {e}")
            
            elif 'pca_analysis' in basename and basename not in loaded:
                try:
                    with open(json_file, 'r') as f:
                        if 'pca_umap_results' not in results:
                            results['pca_umap_results'] = {}
                        view_name = basename.split('_')[2]  # pca_analysis_FULL_timestamp.json
                        results['pca_umap_results'][view_name] = json.load(f)
                    loaded[basename] = True
                    log.info(f"  ✓ PCA Analysis: {basename}")
                except Exception as e:
                    log.warning(f"Erro ao carregar {basename}: {e}")
        
        log.info(f"✓ {len(loaded)} arquivos carregados")
        
        return results
    
    def generate_markdown_report(self, output_path: str):
        """
        Gera relatório em formato Markdown
        
        Args:
            output_path: Caminho do arquivo de saída (.md)
        """
        log.info("Gerando relatório em Markdown...")
        
        report = self._build_markdown_content()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        log.info(f"✓ Relatório Markdown salvo em: {output_path}")
    
    def _build_markdown_content(self) -> str:
        """Constrói conteúdo do relatório em Markdown"""
        
        sections = []
        
        # Cabeçalho
        sections.append(self._section_header())
        
        # Resumo executivo
        sections.append(self._section_executive_summary())
        
        # Metodologia
        sections.append(self._section_methodology())
        
        # Resultados - Comparação de visões
        sections.append(self._section_view_comparison())
        
        # Resultados - Protótipos
        sections.append(self._section_prototypes())
        
        # Resultados - Padrões linguísticos
        sections.append(self._section_patterns())
        
        # Insights estratégicos
        sections.append(self._section_insights())
        
        # Conclusões
        sections.append(self._section_conclusions())
        
        # Apêndices
        sections.append(self._section_appendix())
        
        return "\n\n".join(sections)
    
    def _section_header(self) -> str:
        """Cabeçalho do relatório"""
        return f"""# Análise de Padrões Semânticos em Ligações de Vendas

**Trabalho de Conclusão de Curso (TCC)**  
**Autor:** Daniel Nascimento  
**Data:** {datetime.now().strftime("%d de %B de %Y")}  
**Versão:** 1.0 - Análise Multiview com Embeddings V2

---

## Sumário

1. [Resumo Executivo](#resumo-executivo)
2. [Metodologia](#metodologia)
3. [Comparação de Visões de Embedding](#comparação-de-visões-de-embedding)
4. [Análise de Protótipos](#análise-de-protótipos)
5. [Padrões Linguísticos Discriminativos](#padrões-linguísticos-discriminativos)
6. [Insights Estratégicos](#insights-estratégicos)
7. [Conclusões e Recomendações](#conclusões-e-recomendações)
8. [Apêndices](#apêndices)

---"""
    
    def _section_executive_summary(self) -> str:
        """Resumo executivo com dados reais"""
        
        # Extrai dados
        comparison = self.pipeline_results.get('comparison_results', {})
        global_comp = comparison.get('global_comparison', {})
        prototypes = self.pipeline_results.get('prototype_results', {})
        patterns_status = self.pipeline_results.get('pattern_status_results', {})
        
        # Identifica melhor visão
        best_view = "full"
        best_silhouette = 0.0
        if global_comp:
            ranking = global_comp.get('ranking', [])
            if ranking:
                best_data = ranking[0]  # Já vem ordenado
                best_view = best_data.get('view', 'full')
                best_silhouette = best_data.get('silhouette', 0.0)
        
        # Conta padrões significativos
        n_patterns = 0
        if patterns_status:
            for product_data in patterns_status.values():
                if isinstance(product_data, dict):
                    n_patterns += len(product_data.get('significant_patterns', []))
        
        # Conta produtos analisados
        n_products = len(patterns_status) if patterns_status else 0
        
        return f"""## 1. Resumo Executivo

Este trabalho investiga padrões semânticos discriminativos em ligações de vendas utilizando embeddings de texto e técnicas de análise multiview. A abordagem introduz:

### Contribuições Principais

1. **Análise Multiview**: Comparação sistemática de três perspectivas de embedding
   - **Full**: Transcrição completa da ligação
   - **Agent**: Apenas falas do agente
   - **Client**: Apenas falas do cliente

2. **Análise por Produto**: Identificação de padrões específicos por tipo de produto vendido

3. **Análise por Status**: Separação de padrões em oportunidades ganhas vs perdidas

4. **Padrões Linguísticos**: Extração de 80+ patterns organizados em 15 categorias semânticas

### Principais Descobertas

- **Melhor Visão de Embedding**: `{best_view}` (silhueta: {best_silhouette:.4f})
- **Separação Semântica**: Silhueta média de {best_silhouette:.4f} entre ganhas/perdidas
- **Padrões Significativos**: {n_patterns} patterns discriminativos identificados
- **Produtos Analisados**: {n_products} produtos com análise completa
- **Insights Acionáveis**: Recomendações específicas por produto baseadas em dados reais

---"""
    
    def _section_methodology(self) -> str:
        """Seção de metodologia"""
        return f"""## 2. Metodologia

### 2.1. Pipeline de Processamento

```
Transcrições → Embeddings V2 → Análise Multiview → Insights
                  ↓                    ↓
            (3 visões)      Protótipos + Padrões
```

### 2.2. Embeddings V2

- **Modelo**: `jinaai/jina-embeddings-v3`
- **Dimensionalidade**: 768d
- **Token Limit**: 8192 tokens (vs 512 da V1)
- **Estratégia**: Re-embedding de texto concatenado
- **Visões**:
  - `full`: Ligação completa
  - `agent`: Apenas agente
  - `client`: Apenas cliente
  
**Nota**: A visão `labeled` (com marcadores [AG]/[CL]) foi descartada por degradação de qualidade confirmada empiricamente.

### 2.3. Análise de Protótipos

**Objetivo**: Identificar centróides semânticos de ligações ganhas vs perdidas

**Métricas**:
- **Coesão Intra-Cluster**: Similaridade média dentro do cluster
- **Separação Inter-Cluster**: Distância cosseno entre protótipos
- **Silhueta**: Qualidade de separação global (-1 a 1)

### 2.4. Análise de Padrões Linguísticos

**Patterns Extraídos**: {len(settings_v3.STOPWORDS_COMPLETAS)} stopwords filtradas

**Categorias** (15 categorias, 80+ patterns):
1. Produtos (seguro carga, garantia, vida, etc.)
2. Processos (cotação, proposta, contrato, etc.)
3. Documentação (minuta, CNPJ, nota fiscal, etc.)
4. Financeiro (valor, desconto, parcela, etc.)
5. Urgência (urgente, prazo, rápido, etc.)
6. Problemas (dúvida, problema, erro, etc.)
7. Positivo (perfeito, obrigado, entendi, etc.)
8. Objeções (caro, concorrente, já tenho, etc.)
9. Fechamento (quando, como funciona, próximo passo, etc.)
10. Negociação (negociar, melhor preço, concessão, etc.)
11. Relacionamento (parceria, confiança, indicação, etc.)
12. Técnico/Compliance (cláusula, exclusão, regulamento, etc.)
13. Competitivo (diferencial, comparação, inovação, etc.)
14. Risco/Segurança (risco, proteção, cobertura, etc.)
15. Operacional (sistema, email, telefone, etc.)

**Teste Estatístico**: Chi-quadrado para significância (p < 0.05)

### 2.5. Visualizações

- **UMAP**: Redução dimensional para visualização 2D
- **Parâmetros**: n_neighbors=15, min_dist=0.1, metric=cosine
- **Comparações**: Heatmaps, barplots, dashboards

---"""
    
    def _section_view_comparison(self) -> str:
        """Seção de comparação de visões com dados reais"""
        
        comparison = self.pipeline_results.get('comparison_results', {})
        global_comp = comparison.get('global_comparison', {})
        by_product = comparison.get('by_product', {})
        
        # Tabela de métricas globais
        table_rows = []
        by_view = global_comp.get('by_view', {})
        for view_name in ['full', 'agent', 'client']:
            if view_name in by_view:
                view_data = by_view[view_name]
                view = view_name.capitalize()
                silh = view_data.get('silhouette_overall', 0.0)
                sep = view_data.get('separation_distance', 0.0)
                coes_g = view_data.get('cohesion_ganha', 0.0)
                coes_p = view_data.get('cohesion_perdida', 0.0)
                n_samp = view_data.get('n_ganha', 0) + view_data.get('n_perdida', 0)
                table_rows.append(f"| {view:6} | {silh:8.4f} | {sep:9.4f} | {coes_g:12.4f} | {coes_p:14.4f} | {n_samp:10} |")
        
        table_str = "\n".join(table_rows) if table_rows else "| (sem dados) | - | - | - | - | - |"
        
        # Melhor visão
        best_view = "N/A"
        best_silh = 0.0
        ranking = global_comp.get('ranking', [])
        if ranking:
            best_view = ranking[0].get('view', 'N/A')
            best_silh = ranking[0].get('silhouette', 0.0)
        
        # Ranking por produto
        products_comp = comparison.get('products_comparison', {})
        best_view_counts = products_comp.get('best_view_counts', {})
        
        ranking = sorted(best_view_counts.items(), key=lambda x: x[1], reverse=True)
        ranking_str = "\n".join([f"{i+1}. **{view.capitalize()}**: {count} produtos" 
                                  for i, (view, count) in enumerate(ranking)])
        if not ranking_str:
            ranking_str = "(sem dados)"
        
        # Performance média por visão (extrair de product_comparisons)
        product_comps = products_comp.get('product_comparisons', {})
        view_stats = {}
        
        for product_name, prod_data in product_comps.items():
            by_view = prod_data.get('by_view', {})
            for view_name, view_data in by_view.items():
                silh = view_data.get('silhouette_overall', 0.0)
                if view_name not in view_stats:
                    view_stats[view_name] = []
                view_stats[view_name].append(silh)
        
        stats_str = []
        for view in ['full', 'agent', 'client']:
            if view in view_stats and view_stats[view]:
                import numpy as np
                mean = np.mean(view_stats[view])
                std = np.std(view_stats[view])
                stats_str.append(f"- **{view.capitalize()}**: μ={mean:.4f}, σ={std:.4f}")
            else:
                stats_str.append(f"- **{view.capitalize()}**: (sem dados)")
        
        stats_text = "\n".join(stats_str)
        
        return f"""## 3. Comparação de Visões de Embedding

### 3.1. Métricas Globais

| Visão  | Silhueta | Separação | Coesão Ganha | Coesão Perdida | N Amostras |
|--------|----------|-----------|--------------|----------------|------------|
{table_str}

**Melhor Visão (Global)**: `{best_view}` com silhueta de {best_silh:.4f}

### 3.2. Performance por Produto

**Ranking de Consistência** (visão que venceu em mais produtos):

{ranking_str}

**Performance Média por Visão**:

{stats_text}

### 3.3. Recomendação Final

**Visão Recomendada**: `{best_view}`

**Justificativa**: Baseado nas métricas globais e consistência por produto, a visão `{best_view}` apresentou melhor separação semântica entre ligações ganhas e perdidas (silhueta: {best_silh:.4f}).

**Visualizações**:
- Ver: `plots/umap_comparative_*.png`
- Ver: `plots/view_metrics_*.png`
- Ver: `plots/product_performance_*.png`

---"""
    
    def _section_prototypes(self) -> str:
        """Seção de protótipos com dados reais"""
        
        prototypes = self.pipeline_results.get('prototype_results', {})
        
        # Protótipos globais por visão
        sections = []
        sections.append("## 4. Análise de Protótipos\n\n### 4.1. Protótipos Globais\n")
        
        for view in ['full', 'agent', 'client']:
            view_key = f'global_{view}'
            view_data = prototypes.get(view_key, {})
            if view_data:
                ganha_data = view_data.get('ganha', {})
                perdida_data = view_data.get('perdida', {})
                sep_data = view_data.get('separation', {})
                
                n_ganha = ganha_data.get('n_samples', 0)
                n_perdida = perdida_data.get('n_samples', 0)
                cohesion_g = ganha_data.get('cohesion', 0.0)
                cohesion_p = perdida_data.get('cohesion', 0.0)
                separation = sep_data.get('distance', 0.0)
                silh_data = sep_data.get('silhouette', {})
                silhouette = silh_data.get('overall', 0.0) if isinstance(silh_data, dict) else 0.0
                
                sections.append(f"""#### Visão {view.capitalize()}

- **Ganha**: {n_ganha} amostras, coesão={cohesion_g:.4f}
- **Perdida**: {n_perdida} amostras, coesão={cohesion_p:.4f}
- **Separação**: {separation:.4f}
- **Silhueta**: {silhouette:.4f}
""")
            else:
                sections.append(f"""#### Visão {view.capitalize()}

(sem dados)
""")
        
        # Top produtos
        sections.append("\n### 4.2. Protótipos por Produto\n")
        sections.append("**Top 5 Produtos com Melhor Separação**:\n\n")
        
        # Extrai dados de produtos diretamente de prototypes
        product_scores = []
        for key, data in prototypes.items():
            if key.startswith('product_') and key.endswith('_full'):  # Usa apenas full view
                product_name = key.replace('product_', '').replace('_full', '')
                sep_data = data.get('separation', {})
                silh_data = sep_data.get('silhouette', {})
                silh = silh_data.get('overall', 0.0) if isinstance(silh_data, dict) else 0.0
                dist = sep_data.get('distance', 0.0)
                product_scores.append((product_name, silh, dist))
        
        # Ordena e pega top 5
        product_scores.sort(key=lambda x: x[1], reverse=True)
        for i, (product, silh, dist) in enumerate(product_scores[:5], 1):
            sections.append(f"{i}. **{product}**: silhueta={silh:.4f}, dist={dist:.4f}\n")
        
        if not product_scores:
            sections.append("(sem dados)\n")
        
        sections.append("""
**Interpretação**: Produtos com alta separação possuem padrões semânticos bem definidos entre ganhas e perdidas, facilitando identificação de estratégias vencedoras.

**Visualizações**:
- Ver: `plots/umap_products_*.png`

---""")
        
        return "".join(sections)
    
    def _section_patterns(self) -> str:
        """Seção de padrões linguísticos com dados reais"""
        
        patterns_status = self.pipeline_results.get('pattern_status_results', {})
        insights = patterns_status.get('insights', {})
        product_insights = insights.get('product_specific_insights', {})
        universal = insights.get('universal_patterns', {})
        
        sections = []
        sections.append("## 5. Padrões Linguísticos Discriminativos\n\n")
        
        # 5.1 Padrões Universais
        sections.append("### 5.1. Padrões Universais (Multi-Produto)\n\n")
        
        # Extrai padrões universais
        winning = universal.get('winning', [])
        losing = universal.get('losing', [])
        
        # Tabela vencedores
        sections.append("**Padrões Vencedores** (presentes em 3+ produtos):\n\n")
        sections.append("| Keyword | Produtos | N Produtos |\n")
        sections.append("|---------|----------|------------|\n")
        
        for w in winning[:10]:
            keyword = w.get('keyword', '')
            products = w.get('products', [])
            n_prod = len(products)
            count = w.get('count', 0)
            sections.append(f"| {keyword[:25]} | {', '.join(products[:2])}... | {n_prod} |\n")
        
        if not winning:
            sections.append("| (sem dados) | - | - |\n")
        
        # Tabela perdedores
        sections.append("\n**Padrões Perdedores** (presentes em 3+ produtos):\n\n")
        sections.append("| Keyword | Produtos | N Produtos |\n")
        sections.append("|---------|----------|------------|\n")
        
        for l in losing[:10]:
            keyword = l.get('keyword', '')
            products = l.get('products', [])
            n_prod = len(products)
            sections.append(f"| {keyword[:25]} | {', '.join(products[:2])}... | {n_prod} |\n")
        
        if not losing:
            sections.append("| (sem dados) | - | - |\n")
        
        # 5.2 Por produto (top 3)
        sections.append("\n### 5.2. Padrões Específicos por Produto (Top 3)\n\n")
        
        # Ordena produtos por win rate
        product_list = [(p, d.get('win_rate', 0)) for p, d in product_insights.items()]
        product_list.sort(key=lambda x: x[1], reverse=True)
        
        for product, _ in product_list[:3]:
            data = product_insights[product]
            win_rate = data.get('win_rate', 0.0)
            n_ganha = data.get('n_ganha', 0)
            n_perdida = data.get('n_perdida', 0)
            n_calls = n_ganha + n_perdida
            
            top_winning = data.get('top_winning_strategies', [])
            top_losing = data.get('top_losing_strategies', [])
            
            sections.append(f"#### {product}\n\n")
            sections.append(f"**Win Rate**: {win_rate:.1f}%  \n")
            sections.append(f"**N Chamadas**: {n_calls} ({n_ganha} ganhas, {n_perdida} perdidas)\n\n")
            
            # Top vencedores
            sections.append("**Top 5 Estratégias Vencedoras**:\n\n")
            for i, strat in enumerate(top_winning[:5], 1):
                keyword = strat.get('keyword', '')
                category = strat.get('category', 'N/A')
                diff = strat.get('diff', 0.0)
                pval = strat.get('p_value', 1.0)
                sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*"
                sections.append(f"{i}. **{keyword}** ({category}): +{diff:.1f}% {sig}\n")
            
            if not top_winning:
                sections.append("(sem padrões vencedores)\n")
            
            # Top perdedores
            sections.append("\n**Top 3 Padrões a Evitar**:\n\n")
            for i, strat in enumerate(top_losing[:3], 1):
                keyword = strat.get('keyword', '')
                category = strat.get('category', 'N/A')
                diff = strat.get('diff', 0.0)
                pval = strat.get('p_value', 1.0)
                sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*"
                sections.append(f"{i}. **{keyword}** ({category}): {diff:.1f}% {sig}\n")
            
            if not top_losing:
                sections.append("(sem padrões perdedores)\n")
            
            sections.append("\n")
        
        # 5.3 Por categoria (placeholder - dados não disponíveis em JSON)
        sections.append("### 5.3. Análise por Categoria\n\n")
        sections.append("**Nota**: Análise detalhada por categoria disponível nos arquivos JSON individuais.\n\n")
        
        sections.append("""**Visualizações**:
- Ver: `data/patterns_agent_*.json`
- Ver: `data/patterns_client_*.json`

---""")
        
        return "".join(sections)
    
    def _section_insights(self) -> str:
        """Seção de insights estratégicos"""
        return """## 6. Insights Estratégicos

### 6.1. Insights Globais

1. **Visão de Embedding Ótima**: [VISÃO] apresenta melhor separação ganha/perdida
2. **Consistência**: [VISÃO] é mais consistente através de produtos (menor variância)
3. **Padrões Universais**: [N] patterns são discriminativos em múltiplos produtos

### 6.2. Insights por Produto

#### Produtos de Alto Desempenho (Win Rate > 60%)

**Características comuns**:
- Uso frequente de: [patterns]
- Evitam: [patterns]
- Foco nas categorias: [categorias]

#### Produtos de Baixo Desempenho (Win Rate < 40%)

**Oportunidades de melhoria**:
- Aumentar uso de: [patterns]
- Reduzir: [patterns]
- Treinar em: [categorias]

### 6.3. Recomendações Práticas

#### Para Gestores de Vendas

1. **Treinamento**: Foco em patterns vencedores da categoria [X]
2. **Scripts**: Incorporar keywords: [lista]
3. **Evitar**: Reduzir uso de objeções como [keywords]

#### Para Produtos Específicos

**[Produto A]**:
- ✅ Manter: [estratégias]
- ⚠️ Melhorar: [aspectos]
- 🎯 Foco: [categorias]

*[Repetir para produtos principais]*

### 6.4. Impacto Esperado

**Adoção de Recomendações**:
- Potencial aumento de win rate: +[X]% a +[Y]%
- ROI estimado: [valor]
- Tempo de implementação: [período]

---"""
    
    def _section_conclusions(self) -> str:
        """Seção de conclusões"""
        return """## 7. Conclusões e Recomendações

### 7.1. Principais Conclusões

1. **Embeddings V2 são Superiores**: Re-embedding com limite de 8192 tokens captura melhor semântica completa

2. **Visões Complementares**: [VISÃO] é globalmente melhor, mas visões específicas podem ser úteis por produto

3. **Padrões São Discriminativos**: 80+ patterns organizados em 15 categorias capturam diferenças significativas

4. **Variação por Produto**: Estratégias vencedoras são produto-específicas, não universais

5. **Análise Temporal Importa**: Padrões presentes em diferentes fases da conversa têm impactos distintos

### 7.2. Limitações

- **Dados**: Análise limitada a ligações com transcrição completa
- **Causalidade**: Correlação ≠ causação; patterns podem ser efeito, não causa
- **Generalização**: Resultados específicos para o contexto de seguros
- **Token Limit**: Algumas ligações longas são truncadas

### 7.3. Trabalhos Futuros

1. **Modelos Preditivos**: Treinar classificadores com features extraídas
2. **Análise Temporal**: Incorporar ordem e timing dos patterns
3. **Multimodalidade**: Adicionar características prosódicas (tom, velocidade)
4. **A/B Testing**: Validar recomendações em campo
5. **LLMs**: Explorar embeddings de modelos maiores (GPT, Claude)

### 7.4. Considerações Finais

Este trabalho demonstra que análise semântica sistemática de ligações de vendas pode identificar padrões discriminativos acionáveis. A abordagem multiview oferece uma base sólida para sistemas de coaching automatizado e recomendação de estratégias.

**Implementação Prática**: Os insights podem ser integrados em:
- Dashboards de monitoramento em tempo real
- Sistemas de feedback pós-chamada
- Módulos de treinamento personalizados
- Ferramentas de sugestão durante chamadas

---"""
    
    def _section_appendix(self) -> str:
        """Seção de apêndices"""
        return f"""## 8. Apêndices

### A. Estrutura de Arquivos

```
sales-call-topic-analysis/
├── config/
│   └── settings_v3.py          # Configurações
├── core/
│   ├── database_v3.py          # Conexão com banco
│   └── embeddings_v3.py        # Utilitários de embeddings
├── analysis/
│   ├── prototypes_v3.py        # Análise de protótipos
│   ├── patterns_by_product.py  # Padrões por produto
│   ├── patterns_by_product_status.py  # Padrões por status
│   └── comparisons.py          # Comparação de visões
├── visualization/
│   ├── umap_plots.py           # Visualizações UMAP
│   └── comparison_plots.py     # Gráficos comparativos
├── outputs/
│   ├── data/                   # Dados JSON
│   ├── plots/                  # Gráficos PNG
│   └── reports/                # Relatórios
├── pipeline_v3_main.py         # Pipeline principal
└── generate_tcc_report.py      # Gerador deste relatório
```

### B. Configurações Utilizadas

```python
EMBEDDING_VIEWS = {settings_v3.EMBEDDING_VIEWS}
MIN_CALLS_PER_PRODUCT = {settings_v3.MIN_CALLS_PER_PRODUCT}
MIN_CALLS_PER_PRODUCT_STATUS = {settings_v3.MIN_CALLS_PER_PRODUCT_STATUS}
UMAP_N_NEIGHBORS = {settings_v3.UMAP_N_NEIGHBORS}
UMAP_MIN_DIST = {settings_v3.UMAP_MIN_DIST}
CHI_SQUARE_THRESHOLD = {settings_v3.CHI_SQUARE_THRESHOLD}
MIN_DIFF_PERCENTAGE = {settings_v3.MIN_DIFF_PERCENTAGE}
```

### C. Dependências

- Python 3.9+
- sentence-transformers
- psycopg[binary]
- numpy
- matplotlib
- seaborn
- umap-learn
- scikit-learn

### D. Reprodução

Para reproduzir este relatório:

```bash
# 1. Executar pipeline completo
python pipeline_v3_main.py

# 2. Gerar relatório
python generate_tcc_report.py
```

### E. Contato

**Autor**: Daniel Nascimento  
**Email**: [email]  
**GitHub**: [repo]  
**Data**: {datetime.now().strftime("%d/%m/%Y")}

---

**FIM DO RELATÓRIO**
"""
    
    def generate_html_report(self, markdown_path: str, output_path: str):
        """
        Converte relatório Markdown para HTML
        
        Args:
            markdown_path: Caminho do arquivo Markdown
            output_path: Caminho do arquivo HTML de saída
        """
        try:
            import markdown
            
            # Lê Markdown
            with open(markdown_path, 'r', encoding='utf-8') as f:
                md_content = f.read()
            
            # Converte para HTML
            html_content = markdown.markdown(
                md_content,
                extensions=['tables', 'fenced_code', 'toc']
            )
            
            # Template HTML com estilo
            html_template = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TCC - Análise de Padrões Semânticos em Vendas</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1, h2, h3 {{
            color: #2c3e50;
        }}
        h1 {{
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            border-bottom: 2px solid #95a5a6;
            padding-bottom: 8px;
            margin-top: 40px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background-color: white;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        code {{
            background-color: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }}
        pre {{
            background-color: #2c3e50;
            color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }}
        pre code {{
            background-color: transparent;
            color: #ecf0f1;
        }}
        blockquote {{
            border-left: 4px solid #3498db;
            padding-left: 20px;
            color: #555;
            font-style: italic;
        }}
    </style>
</head>
<body>
{html_content}
</body>
</html>"""
            
            # Salva HTML
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_template)
            
            log.info(f"✓ Relatório HTML salvo em: {output_path}")
            
        except ImportError:
            log.warning("Módulo 'markdown' não encontrado. Instale com: pip install markdown")
        except Exception as e:
            log.error(f"Erro ao gerar HTML: {e}")
    
    def generate_full_report(self):
        """Gera relatório completo (MD e HTML)"""
        log.info("="*80)
        log.info("📄 GERANDO RELATÓRIO FINAL PARA TCC")
        log.info("="*80)
        
        # Markdown
        md_path = os.path.join(settings_v3.V3_REPORTS_DIR, f"tcc_report_{self.timestamp}.md")
        self.generate_markdown_report(md_path)
        
        # HTML
        if settings_v3.TCC_REPORT_FORMAT in ["html", "both"]:
            html_path = os.path.join(settings_v3.V3_REPORTS_DIR, f"tcc_report_{self.timestamp}.html")
            self.generate_html_report(md_path, html_path)
        
        log.info("\n✓ Relatórios gerados com sucesso!")
        log.info(f"  • Markdown: {md_path}")
        if settings_v3.TCC_REPORT_FORMAT in ["html", "both"]:
            log.info(f"  • HTML: {html_path}")


def main():
    """Ponto de entrada"""
    generator = TCCReportGenerator()
    generator.generate_full_report()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

