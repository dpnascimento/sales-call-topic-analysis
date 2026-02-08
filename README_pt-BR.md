# Análise de Padrões Semânticos em Ligações de Vendas

**Versão:** 1.0.0  
**Autor:** Daniel Nascimento  
**Data:** Fevereiro 2026

[🇺🇸 English](README_en-US.md) | 🇧🇷 Português

---

## 📋 Índice

1. [Visão Geral](#visão-geral)
2. [Arquitetura](#arquitetura)
3. [Instalação](#instalação)
4. [Uso Rápido](#uso-rápido)
5. [Módulos](#módulos)
6. [Configuração](#configuração)
7. [Outputs](#outputs)
8. [Exemplos](#exemplos)
9. [FAQ](#faq)
10. [Contribuindo](#contribuindo)

---

## 🎯 Visão Geral

Solução  para análise de padrões semânticos discriminativos em ligações de vendas. 

### ✨ Principais Características

- **🔍 Análise Multiview**: Compara 3 perspectivas de embedding (full, agent, client)
- **📊 Análise por Produto**: Identifica padrões específicos por tipo de produto
- **📈 Análise por Status**: Separa padrões em oportunidades ganhas vs perdidas
- **🏷️ 80+ Patterns**: Organizados em 15 categorias semânticas
- **🔬 PCA Interpretável + UMAP**: Análise geométrica com interpretação matemática
- **📉 Visualizações**: UMAPs, PCAs, heatmaps, dashboards automáticos
- **📄 Relatórios TCC**: Geração automática de relatórios em MD/HTML

---

## 🏗️ Arquitetura

```
sales-call-topic-analysis/
├── 📁 config/                    # Configurações
│   ├── __init__.py
│   └── settings_v3.py           # Settings centralizados
│
├── 📁 core/                      # Utilitários core
│   ├── __init__.py
│   ├── database_v3.py           # DatabaseManager com métodos específicos
│   └── embeddings_v3.py         # Funções de manipulação de embeddings
│
├── 📁 analysis/                  # Módulos de análise
│   ├── __init__.py
│   ├── prototypes_v3.py         # Análise de protótipos (3 visões)
│   ├── patterns_by_product.py  # Padrões linguísticos por produto
│   ├── patterns_by_product_status.py  # Padrões por produto + status
│   ├── comparisons.py           # Comparação entre visões
│   └── embedding_geometry.py    # PCA interpretável, LDA, outliers
│
├── 📁 visualization/             # Módulos de visualização
│   ├── __init__.py
│   ├── umap_plots.py            # UMAPs por visão/produto
│   ├── comparison_plots.py      # Gráficos comparativos
│   └── pca_umap_plots.py        # Visualizações integradas PCA+UMAP
│
├── 📁 outputs/                   # Outputs gerados
│   ├── data/                    # JSONs com resultados
│   ├── plots/                   # Gráficos PNG
│   └── reports/                 # Relatórios MD/HTML
│
├── 📄 enhanced_patterns.py       # Definição dos 80+ patterns
├── 📄 pipeline_v3_main.py       # Pipeline principal
├── 📄 generate_tcc_report.py    # Gerador de relatórios
└── 📄 README.md                 # Hub de documentação
```

### 🔄 Fluxo de Dados

```
┌─────────────────────────────────────────────────────────┐
│  PostgreSQL (call_embeddings_v2)                        │
│  • embedding_full, embedding_agent, embedding_client    │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│  DatabaseManagerV3                                      │
│  • get_all_embeddings_by_view()                         │
│  • get_calls_by_product_and_outcome()                   │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│  Análises Paralelas                                     │
│  ├─► PrototypeAnalyzerV3     (protótipos por visão)     │
│  ├─► PatternsByProductAnalyzer (padrões linguísticos)   │
│  ├─► EmbeddingViewComparator (comparação de visões)     │
│  └─► PatternsByProductStatusAnalyzer (insights)         │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│  Visualizações                                          │
│  ├─► UMAPVisualizer         (redução dimensional)       │
│  └─► ComparisonPlotter      (gráficos comparativos)     │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│  Outputs                                                │
│  ├─► data/*.json          (resultados estruturados)     │
│  ├─► plots/*.png          (visualizações)               │
│  └─► reports/*.md|html    (relatórios TCC)              │
└─────────────────────────────────────────────────────────┘
```

---

## 📦 Instalação

### Pré-requisitos

- Python 3.9+
- PostgreSQL com extensão `pgvector`
- Embeddings V2 já gerados (via `generate_call_embeddings_v2.py`)

### Dependências

```bash
# Instalar dependências
pip install -r requirements.txt

# Dependências principais:
# - sentence-transformers
# - psycopg[binary]
# - numpy
# - matplotlib
# - seaborn
# - umap-learn
# - scikit-learn
# - markdown (opcional, para relatórios HTML)
```

### Configuração do Banco

Certifique-se de que a tabela `call_embeddings_v2` existe:

```sql
SELECT COUNT(*) FROM call_embeddings_v2;
SELECT COUNT(*) FROM call_embeddings_v2 WHERE full_valid = TRUE;
SELECT COUNT(*) FROM call_embeddings_v2 WHERE agent_valid = TRUE;
SELECT COUNT(*) FROM call_embeddings_v2 WHERE client_valid = TRUE;
```

---

## 🚀 Uso Rápido

### Executar Pipeline Completo

```bash
# Executar pipeline principal
python pipeline_v3_main.py
```

Isso irá:
1. ✅ Conectar ao banco
2. ✅ Analisar protótipos (3 visões)
3. ✅ Analisar padrões linguísticos (agente + cliente)
4. ✅ Analisar padrões por produto + status
5. ✅ Comparar visões de embedding
6. ✅ Gerar todas as visualizações
7. ✅ Salvar resultados em `outputs/`

### Gerar Relatório TCC

```bash
# Após executar o pipeline
python generate_tcc_report.py
```

Isso irá gerar:
- `reports/tcc_report_YYYYMMDD_HHMMSS.md` (Markdown)
- `reports/tcc_report_YYYYMMDD_HHMMSS.html` (HTML, se configurado)

---

## 📚 Módulos

### 1. Core

#### `database_v3.py`

```python
from core.database_v3 import DatabaseManagerV3

db = DatabaseManagerV3()
db.connect()

# Buscar produtos
products = db.get_products(min_calls=20)

# Buscar embeddings de um produto
calls = db.get_calls_by_product_and_outcome(
    product_name="Seguro Carga",
    outcome="ganha",
    embedding_view="full"
)

# Buscar transcrições
transcripts = db.get_product_transcripts_by_role(
    product_name="Seguro Carga",
    outcome="ganha",
    role="AGENTE"
)
```

#### `embeddings_v3.py`

```python
from core.embeddings_v3 import (
    from_pgvector, cosine_similarity,
    centroid, average_silhouette
)

# Parse embedding do banco
vec = from_pgvector("[0.1, 0.2, ...]")

# Calcular similaridade
sim = cosine_similarity(vec_a, vec_b)

# Calcular centróide de cluster
proto = centroid([vec1, vec2, vec3, ...])

# Calcular silhueta
silhouette = average_silhouette(cluster_a, cluster_b)
```

### 2. Analysis

#### `prototypes_v3.py`

```python
from analysis.prototypes_v3 import PrototypeAnalyzerV3

analyzer = PrototypeAnalyzerV3(db)

# Protótipos globais
global_protos = analyzer.compute_global_prototypes(embedding_view="full")

# Protótipos por produto
product_protos = analyzer.compute_product_prototypes(
    product_name="Seguro Carga",
    embedding_view="agent"
)

# Comparar separação entre produtos
comparison = analyzer.compare_products_separation(embedding_view="client")
```

#### `patterns_by_product.py`

```python
from analysis.patterns_by_product import PatternsByProductAnalyzer

analyzer = PatternsByProductAnalyzer(db)

# Analisar um produto
analysis = analyzer.analyze_product(
    product_name="Seguro Carga",
    role="AGENTE"
)

# Top patterns vencedores
winners = analyzer.get_top_winning_patterns(
    product_name="Seguro Carga",
    n=10
)

# Comparar produtos
comparison = analyzer.compare_products(
    product_a="Seguro Carga",
    product_b="Seguro Garantia"
)
```

#### `comparisons.py`

```python
from analysis.comparisons import EmbeddingViewComparator

comparator = EmbeddingViewComparator(db)

# Comparação global
global_comp = comparator.compare_views_global()

# Comparação por produtos
products_comp = comparator.compare_views_all_products()

# Recomendações
recommendations = comparator.generate_view_recommendations()
```

### 3. Visualization

#### `umap_plots.py`

```python
from visualization.umap_plots import UMAPVisualizer

visualizer = UMAPVisualizer(db)

# UMAP comparativo (3 visões)
comparative_umaps = visualizer.create_comparative_umap()
visualizer.plot_comparative_umaps(comparative_umaps, "output.png")

# UMAP por produto
product_umap = visualizer.create_umap_by_product(
    product_name="Seguro Carga",
    embedding_view="full"
)
visualizer.plot_umap(product_umap, "product_umap.png")

# Grid de produtos
grid = visualizer.create_product_grid_umap(embedding_view="agent")
visualizer.plot_product_grid(grid, "grid.png")
```

#### `comparison_plots.py`

```python
from visualization.comparison_plots import ComparisonPlotter

plotter = ComparisonPlotter(db)

# Métricas de visões
plotter.plot_view_comparison_metrics(comparison_data, "metrics.png")

# Heatmap de performance
plotter.plot_product_performance_by_view(products_comp, "heatmap.png")

# Win rate por produto
plotter.plot_win_rate_by_product(products, "winrate.png")

# Dashboard resumido
plotter.create_summary_dashboard(
    view_comparison, products, "dashboard.png"
)
```

---

## ⚙️ Configuração

### `settings_v3.py`

```python
# Visões de embedding
EMBEDDING_VIEWS = ["full", "agent", "client"]

# Filtros
MIN_CALLS_PER_PRODUCT = 20
MIN_CALLS_PER_PRODUCT_STATUS = 10

# Análise de protótipos
COMPUTE_PROTOTYPES_PER_VIEW = True
COMPUTE_PROTOTYPES_PER_PRODUCT = True

# Padrões
USE_ENHANCED_PATTERNS = True
CHI_SQUARE_THRESHOLD = 3.84  # p < 0.05
MIN_DIFF_PERCENTAGE = 3.0

# Visualizações
CREATE_UMAP_PER_VIEW = True
CREATE_UMAP_PER_PRODUCT = True
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
UMAP_SAMPLE_SIZE = 5000

# Relatórios TCC
GENERATE_TCC_REPORT = True
TCC_REPORT_FORMAT = "both"  # "markdown", "html", "both"
```

---

## 📊 Outputs

### Estrutura de Outputs

```
outputs/
├── data/
│   ├── prototypes_20251015_143022.json
│   ├── patterns_agent_20251015_143022.json
│   ├── patterns_client_20251015_143022.json
│   ├── patterns_status_20251015_143022.json
│   └── view_comparison_20251015_143022.json
│
├── plots/
│   ├── umap_comparative_20251015_143022.png
│   ├── umap_products_full_20251015_143022.png
│   ├── umap_products_agent_20251015_143022.png
│   ├── umap_products_client_20251015_143022.png
│   ├── view_metrics_20251015_143022.png
│   ├── product_performance_20251015_143022.png
│   ├── win_rate_20251015_143022.png
│   └── dashboard_20251015_143022.png
│
└── reports/
    ├── tcc_report_20251015_143022.md
    └── tcc_report_20251015_143022.html
```

### Descrição dos Outputs

| Arquivo | Descrição |
|---------|-----------|
| `prototypes_*.json` | Protótipos por visão e produto |
| `patterns_agent_*.json` | Padrões linguísticos do agente |
| `patterns_client_*.json` | Padrões linguísticos do cliente |
| `patterns_status_*.json` | Insights por produto + status |
| `view_comparison_*.json` | Comparação entre visões |
| `umap_comparative_*.png` | UMAPs lado a lado (3 visões) |
| `umap_products_*.png` | Grid de UMAPs por produto |
| `view_metrics_*.png` | Métricas comparativas |
| `product_performance_*.png` | Heatmap de silhueta |
| `win_rate_*.png` | Win rate por produto |
| `dashboard_*.png` | Dashboard resumido |
| `tcc_report_*.md` | Relatório completo (Markdown) |
| `tcc_report_*.html` | Relatório completo (HTML) |

---

## 💡 Exemplos

### Exemplo 1: Análise Rápida de um Produto

```python
from core.database_v3 import DatabaseManagerV3
from analysis.patterns_by_product import PatternsByProductAnalyzer

# Conectar
db = DatabaseManagerV3()
db.connect()

# Analisar produto
analyzer = PatternsByProductAnalyzer(db)
analysis = analyzer.analyze_product("Seguro Carga", role="AGENTE")

# Top 5 vencedores
winners = analyzer.get_top_winning_patterns("Seguro Carga", n=5)

for i, pattern in enumerate(winners, 1):
    print(f"{i}. [{pattern['category']}] {pattern['keyword']}: +{pattern['diff']:.1f}%")

db.close()
```

### Exemplo 2: Comparar Visões

```python
from analysis.comparisons import EmbeddingViewComparator

comparator = EmbeddingViewComparator(db)

# Comparação global
results = comparator.compare_views_global()

print(f"Melhor visão: {results['best_view']}")
for view, metrics in results['by_view'].items():
    print(f"  {view}: silhueta={metrics['silhouette_overall']:.4f}")
```

### Exemplo 3: Gerar Visualização Específica

```python
from visualization.umap_plots import UMAPVisualizer

visualizer = UMAPVisualizer(db)

# UMAP de um produto
umap_data = visualizer.create_umap_by_product(
    product_name="Seguro Garantia",
    embedding_view="agent"
)

visualizer.plot_umap(
    umap_data,
    output_path="seguro_garantia_agent.png",
    title="Seguro Garantia - Visão Agent"
)
```

---

## ❓ FAQ

### P: Por que 3 visões de embedding?

**R**: Diferentes perspectivas capturam diferentes aspectos semânticos:
- **Full**: Contexto completo da conversa
- **Agent**: Estratégias e abordagens do vendedor
- **Client**: Objeções e interesse do cliente

### P: O que aconteceu com a visão `labeled`?

**R**: A visão `labeled` (com marcadores [AG]/[CL]) foi descartada após análise empírica mostrar degradação de qualidade comparada às outras 3 visões.

### P: Como interpretar a silhueta?

**R**: Silhueta mede qualidade de separação:
- **> 0.5**: Separação excelente
- **0.3 - 0.5**: Separação boa
- **0.1 - 0.3**: Separação fraca
- **< 0.1**: Clusters sobrepostos

### P: Posso adicionar novos patterns?

**R**: Sim! Edite `enhanced_patterns.py` e adicione novos patterns em `get_enhanced_patterns()`. Não esqueça de categorizá-los em `categorize_patterns()`.

### P: Como filtrar por período?

**R**: Modifique as queries em `database_v3.py` adicionando filtros em `recorded_at`.

### P: Posso usar outros modelos de embedding?

**R**: Sim, mas requer:
1. Gerar novos embeddings com `generate_call_embeddings_v2.py`
2. Atualizar o nome da tabela diretamente em `database_v3.py` (buscar por `public.call_embeddings_v2`)

---

## 🤝 Contribuindo

### Estrutura de Commits

```
tipo(escopo): descrição curta

Descrição detalhada (opcional)
```

**Tipos**:
- `feat`: Nova funcionalidade
- `fix`: Correção de bug
- `docs`: Documentação
- `refactor`: Refatoração de código
- `test`: Testes
- `chore`: Manutenção

**Exemplo**:
```
feat(analysis): adicionar análise temporal de patterns

Implementa detecção de patterns em diferentes fases da conversa
(início, meio, fim) para identificar momentos críticos.
```

### Roadmap

- [ ] Análise temporal de patterns (fases da conversa)
- [ ] Integração com LLMs para explicações automáticas
- [ ] Dashboard interativo com Streamlit
- [ ] API REST para integração com sistemas externos
- [ ] Análise de prosódia (tom, velocidade, pausas)
- [ ] Modelo preditivo de win rate

---

## 📝 Licença

Este projeto faz parte de um Trabalho de Conclusão de Curso (TCC) e está disponível para fins acadêmicos.

---

## 📧 Contato

**Autor**: Daniel Nascimento  
**Email**: dpnascimento@gmail.com  
**LinkedIn**: https://www.linkedin.com/in/dpnascimento  
**GitHub**: https://www.github.com/dpnascimento

---

**Última Atualização**: Outubro 2025  
**Versão**: 3.0.0
