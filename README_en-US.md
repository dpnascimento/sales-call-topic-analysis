# Semantic Pattern Analysis in Sales Calls

**Version:** 1.0.0  
**Author:** Daniel Nascimento  
**Date:** February 2026

🇺🇸 English | [🇧🇷 Português](README_pt-BR.md)

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Modules](#modules)
6. [Configuration](#configuration)
7. [Outputs](#outputs)
8. [Examples](#examples)
9. [FAQ](#faq)
10. [Contributing](#contributing)

---

## 🎯 Overview

Solution for analyzing discriminative semantic patterns in sales calls.

### ✨ Key Features

- **🔍 Multiview Analysis**: Compares 3 embedding perspectives (full, agent, client)
- **📊 Product-Based Analysis**: Identifies specific patterns by product type
- **📈 Status-Based Analysis**: Separates patterns in won vs lost opportunities
- **🏷️ 80+ Patterns**: Organized into 15 semantic categories
- **🔬 Interpretable PCA + UMAP**: Geometric analysis with mathematical interpretation
- **📉 Visualizations**: UMAPs, PCAs, heatmaps, automatic dashboards
- **📄 Reports**: Automatic report generation in MD/HTML

---

## 🏗️ Architecture

```
sales-call-topic-analysis/
├── 📁 config/                    # Configurations
│   ├── __init__.py
│   └── settings_v3.py           # Centralized settings
│
├── 📁 core/                      # Core utilities
│   ├── __init__.py
│   ├── database_v3.py           # DatabaseManager with specific methods
│   └── embeddings_v3.py         # Embedding manipulation functions
│
├── 📁 analysis/                  # Analysis modules
│   ├── __init__.py
│   ├── prototypes_v3.py         # Prototype analysis (3 views)
│   ├── patterns_by_product.py  # Linguistic patterns by product
│   ├── patterns_by_product_status.py  # Patterns by product + status
│   ├── comparisons.py           # View comparison
│   └── embedding_geometry.py    # Interpretable PCA, LDA, outliers
│
├── 📁 visualization/             # Visualization modules
│   ├── __init__.py
│   ├── umap_plots.py            # UMAPs by view/product
│   ├── comparison_plots.py      # Comparative plots
│   └── pca_umap_plots.py        # Integrated PCA+UMAP visualizations
│
├── 📁 outputs/                   # Generated outputs
│   ├── data/                    # JSONs with results
│   ├── plots/                   # PNG plots
│   └── reports/                 # MD/HTML reports
│
├── 📄 enhanced_patterns.py       # Definition of 80+ patterns
├── 📄 pipeline_v3_main.py       # Main pipeline
├── 📄 generate_tcc_report.py    # Report generator
└── 📄 README.md                 # Documentation hub
```

### 🔄 Data Flow

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
│  Parallel Analyses                                      │
│  ├─► PrototypeAnalyzerV3     (prototypes by view)       │
│  ├─► PatternsByProductAnalyzer (linguistic patterns)    │
│  ├─► EmbeddingViewComparator (view comparison)          │
│  └─► PatternsByProductStatusAnalyzer (insights)         │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│  Visualizations                                         │
│  ├─► UMAPVisualizer         (dimensionality reduction)  │
│  └─► ComparisonPlotter      (comparative plots)         │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│  Outputs                                                │
│  ├─► data/*.json          (structured results)          │
│  ├─► plots/*.png          (visualizations)              │
│  └─► reports/*.md|html    (TCC reports)                 │
└─────────────────────────────────────────────────────────┘
```

---

## 📦 Installation

### Prerequisites

- Python 3.9+
- PostgreSQL with `pgvector` extension
- V2 Embeddings already generated (via `generate_call_embeddings_v2.py`)

### Dependencies

```bash
# Install dependencies
pip install -r requirements.txt

# Main dependencies:
# - sentence-transformers
# - psycopg[binary]
# - numpy
# - matplotlib
# - seaborn
# - umap-learn
# - scikit-learn
# - markdown (optional, for HTML reports)
```

### Database Configuration

Make sure the `call_embeddings_v2` table exists:

```sql
SELECT COUNT(*) FROM call_embeddings_v2;
SELECT COUNT(*) FROM call_embeddings_v2 WHERE full_valid = TRUE;
SELECT COUNT(*) FROM call_embeddings_v2 WHERE agent_valid = TRUE;
SELECT COUNT(*) FROM call_embeddings_v2 WHERE client_valid = TRUE;
```

---

## 🚀 Quick Start

### Run Complete Pipeline

```bash
# Execute main pipeline
python pipeline_v3_main.py
```

This will:
1. ✅ Connect to database
2. ✅ Analyze prototypes (3 views)
3. ✅ Analyze linguistic patterns (agent + client)
4. ✅ Analyze patterns by product + status
5. ✅ Compare embedding views
6. ✅ Generate all visualizations
7. ✅ Save results to `outputs/`

### Generate Report

```bash
# After running the pipeline
python generate_tcc_report.py
```

This will generate:
- `reports/tcc_report_YYYYMMDD_HHMMSS.md` (Markdown)
- `reports/tcc_report_YYYYMMDD_HHMMSS.html` (HTML, if configured)

---

## 📚 Modules

### 1. Core

#### `database_v3.py`

```python
from core.database_v3 import DatabaseManagerV3

db = DatabaseManagerV3()
db.connect()

# Fetch products
products = db.get_products(min_calls=20)

# Fetch embeddings for a product
calls = db.get_calls_by_product_and_outcome(
    product_name="Cargo Insurance",
    outcome="won",
    embedding_view="full"
)

# Fetch transcripts
transcripts = db.get_product_transcripts_by_role(
    product_name="Cargo Insurance",
    outcome="won",
    role="AGENT"
)
```

#### `embeddings_v3.py`

```python
from core.embeddings_v3 import (
    from_pgvector, cosine_similarity,
    centroid, average_silhouette
)

# Parse embedding from database
vec = from_pgvector("[0.1, 0.2, ...]")

# Calculate similarity
sim = cosine_similarity(vec_a, vec_b)

# Calculate cluster centroid
proto = centroid([vec1, vec2, vec3, ...])

# Calculate silhouette
silhouette = average_silhouette(cluster_a, cluster_b)
```

### 2. Analysis

#### `prototypes_v3.py`

```python
from analysis.prototypes_v3 import PrototypeAnalyzerV3

analyzer = PrototypeAnalyzerV3(db)

# Global prototypes
global_protos = analyzer.compute_global_prototypes(embedding_view="full")

# Prototypes by product
product_protos = analyzer.compute_product_prototypes(
    product_name="Cargo Insurance",
    embedding_view="agent"
)

# Compare separation between products
comparison = analyzer.compare_products_separation(embedding_view="client")
```

#### `patterns_by_product.py`

```python
from analysis.patterns_by_product import PatternsByProductAnalyzer

analyzer = PatternsByProductAnalyzer(db)

# Analyze a product
analysis = analyzer.analyze_product(
    product_name="Cargo Insurance",
    role="AGENT"
)

# Top winning patterns
winners = analyzer.get_top_winning_patterns(
    product_name="Cargo Insurance",
    n=10
)

# Compare products
comparison = analyzer.compare_products(
    product_a="Cargo Insurance",
    product_b="Guarantee Insurance"
)
```

#### `comparisons.py`

```python
from analysis.comparisons import EmbeddingViewComparator

comparator = EmbeddingViewComparator(db)

# Global comparison
global_comp = comparator.compare_views_global()

# Comparison by products
products_comp = comparator.compare_views_all_products()

# Recommendations
recommendations = comparator.generate_view_recommendations()
```

### 3. Visualization

#### `umap_plots.py`

```python
from visualization.umap_plots import UMAPVisualizer

visualizer = UMAPVisualizer(db)

# Comparative UMAP (3 views)
comparative_umaps = visualizer.create_comparative_umap()
visualizer.plot_comparative_umaps(comparative_umaps, "output.png")

# UMAP by product
product_umap = visualizer.create_umap_by_product(
    product_name="Cargo Insurance",
    embedding_view="full"
)
visualizer.plot_umap(product_umap, "product_umap.png")

# Product grid
grid = visualizer.create_product_grid_umap(embedding_view="agent")
visualizer.plot_product_grid(grid, "grid.png")
```

#### `comparison_plots.py`

```python
from visualization.comparison_plots import ComparisonPlotter

plotter = ComparisonPlotter(db)

# View metrics
plotter.plot_view_comparison_metrics(comparison_data, "metrics.png")

# Performance heatmap
plotter.plot_product_performance_by_view(products_comp, "heatmap.png")

# Win rate by product
plotter.plot_win_rate_by_product(products, "winrate.png")

# Summary dashboard
plotter.create_summary_dashboard(
    view_comparison, products, "dashboard.png"
)
```

---

## ⚙️ Configuration

### `settings_v3.py`

```python
# Embedding views
EMBEDDING_VIEWS = ["full", "agent", "client"]

# Filters
MIN_CALLS_PER_PRODUCT = 20
MIN_CALLS_PER_PRODUCT_STATUS = 10

# Prototype analysis
COMPUTE_PROTOTYPES_PER_VIEW = True
COMPUTE_PROTOTYPES_PER_PRODUCT = True

# Patterns
USE_ENHANCED_PATTERNS = True
CHI_SQUARE_THRESHOLD = 3.84  # p < 0.05
MIN_DIFF_PERCENTAGE = 3.0

# Visualizations
CREATE_UMAP_PER_VIEW = True
CREATE_UMAP_PER_PRODUCT = True
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
UMAP_SAMPLE_SIZE = 5000

# TCC Reports
GENERATE_TCC_REPORT = True
TCC_REPORT_FORMAT = "both"  # "markdown", "html", "both"
```

---

## 📊 Outputs

### Output Structure

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

### Output Descriptions

| File | Description |
|------|-------------|
| `prototypes_*.json` | Prototypes by view and product |
| `patterns_agent_*.json` | Agent linguistic patterns |
| `patterns_client_*.json` | Client linguistic patterns |
| `patterns_status_*.json` | Insights by product + status |
| `view_comparison_*.json` | View comparison |
| `umap_comparative_*.png` | Side-by-side UMAPs (3 views) |
| `umap_products_*.png` | Product UMAP grid |
| `view_metrics_*.png` | Comparative metrics |
| `product_performance_*.png` | Silhouette heatmap |
| `win_rate_*.png` | Win rate by product |
| `dashboard_*.png` | Summary dashboard |
| `tcc_report_*.md` | Complete report (Markdown) |
| `tcc_report_*.html` | Complete report (HTML) |

---

## 💡 Examples

### Example 1: Quick Product Analysis

```python
from core.database_v3 import DatabaseManagerV3
from analysis.patterns_by_product import PatternsByProductAnalyzer

# Connect
db = DatabaseManagerV3()
db.connect()

# Analyze product
analyzer = PatternsByProductAnalyzer(db)
analysis = analyzer.analyze_product("Cargo Insurance", role="AGENT")

# Top 5 winners
winners = analyzer.get_top_winning_patterns("Cargo Insurance", n=5)

for i, pattern in enumerate(winners, 1):
    print(f"{i}. [{pattern['category']}] {pattern['keyword']}: +{pattern['diff']:.1f}%")

db.close()
```

### Example 2: Compare Views

```python
from analysis.comparisons import EmbeddingViewComparator

comparator = EmbeddingViewComparator(db)

# Global comparison
results = comparator.compare_views_global()

print(f"Best view: {results['best_view']}")
for view, metrics in results['by_view'].items():
    print(f"  {view}: silhouette={metrics['silhouette_overall']:.4f}")
```

### Example 3: Generate Specific Visualization

```python
from visualization.umap_plots import UMAPVisualizer

visualizer = UMAPVisualizer(db)

# Product UMAP
umap_data = visualizer.create_umap_by_product(
    product_name="Guarantee Insurance",
    embedding_view="agent"
)

visualizer.plot_umap(
    umap_data,
    output_path="guarantee_insurance_agent.png",
    title="Guarantee Insurance - Agent View"
)
```

---

## ❓ FAQ

### Q: Why 3 embedding views?

**A**: Different perspectives capture different semantic aspects:
- **Full**: Complete conversation context
- **Agent**: Seller's strategies and approaches
- **Client**: Client's objections and interest

### Q: What happened to the `labeled` view?

**A**: The `labeled` view (with [AG]/[CL] markers) was discarded after empirical analysis showed quality degradation compared to the other 3 views.

### Q: How to interpret silhouette?

**A**: Silhouette measures separation quality:
- **> 0.5**: Excellent separation
- **0.3 - 0.5**: Good separation
- **0.1 - 0.3**: Weak separation
- **< 0.1**: Overlapping clusters

### Q: Can I add new patterns?

**A**: Yes! Edit `enhanced_patterns.py` and add new patterns in `get_enhanced_patterns()`. Don't forget to categorize them in `categorize_patterns()`.

### Q: How to filter by period?

**A**: Modify queries in `database_v3.py` by adding filters on `recorded_at`.

### Q: Can I use other embedding models?

**A**: Yes, but requires:
1. Generate new embeddings with `generate_call_embeddings_v2.py`
2. Update the table name directly in `database_v3.py` (search for `public.call_embeddings_v2`)

---

## 🤝 Contributing

### Commit Structure

```
type(scope): short description

Detailed description (optional)
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `refactor`: Code refactoring
- `test`: Tests
- `chore`: Maintenance

**Example**:
```
feat(analysis): add temporal pattern analysis

Implements pattern detection at different conversation phases
(beginning, middle, end) to identify critical moments.
```

### Roadmap

- [ ] Temporal pattern analysis (conversation phases)
- [ ] LLM integration for automatic explanations
- [ ] Interactive dashboard with Streamlit
- [ ] REST API for external system integration
- [ ] Prosody analysis (tone, speed, pauses)
- [ ] Predictive win rate model

---

## 📝 License

This project is part of a Bachelor's Thesis (TCC) and is available for academic purposes.

---

## 📧 Contact

**Author**: Daniel Nascimento  
**Email**: dpnascimento@gmail.com  
**LinkedIn**: https://www.linkedin.com/in/dpnascimento  
**GitHub**: https://www.github.com/dpnascimento

---

**Last Update**: October 2025  
**Version**: 3.0.0
