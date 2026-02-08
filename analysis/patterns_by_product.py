"""
Análise de padrões linguísticos por produto
Identifica patterns discriminativos entre ganha/perdida para cada produto
"""
import logging
import re
from collections import Counter, defaultdict
from typing import Dict, List
import numpy as np

from core.database_v3 import DatabaseManagerV3
from config import settings_v3
from enhanced_patterns import get_enhanced_patterns, categorize_patterns

log = logging.getLogger(__name__)


class PatternsByProductAnalyzer:
    """
    Analisa padrões linguísticos por produto
    Usa enhanced patterns (80+ patterns em 15 categorias)
    """
    
    def __init__(self, db: DatabaseManagerV3):
        self.db = db
        self.patterns = get_enhanced_patterns()
        self.categories = categorize_patterns()
    
    def extract_keywords(self, text: str) -> Dict[str, int]:
        """
        Extrai keywords de um texto usando patterns
        
        Args:
            text: Texto para análise
            
        Returns:
            Dict mapeando keyword -> contagem
        """
        text = text.lower()
        found = {}
        
        for key, pattern in self.patterns.items():
            count = len(re.findall(pattern, text))
            if count > 0:
                found[key] = count
        
        return found
    
    def chi_square_test(
        self, 
        n_ganha_com: int, 
        n_ganha_sem: int, 
        n_perdida_com: int, 
        n_perdida_sem: int
    ) -> tuple:
        """
        Teste qui-quadrado para significância estatística
        
        Returns:
            Tupla (chi2, p_value)
        """
        a, b = n_ganha_com, n_ganha_sem
        c, d = n_perdida_com, n_perdida_sem
        
        n = a + b + c + d
        
        if n == 0:
            return 0, 1.0
        
        # Expected values
        e_a = (a + b) * (a + c) / n
        e_b = (a + b) * (b + d) / n
        e_c = (c + d) * (a + c) / n
        e_d = (c + d) * (b + d) / n
        
        # Chi-square
        chi2 = 0
        for obs, exp in [(a, e_a), (b, e_b), (c, e_c), (d, e_d)]:
            if exp > 0:
                chi2 += (obs - exp) ** 2 / exp
        
        # Aproximação de p-value (1 grau de liberdade)
        if chi2 > 10.83:
            p_value = 0.001
        elif chi2 > 6.63:
            p_value = 0.01
        elif chi2 > 3.84:
            p_value = 0.05
        else:
            p_value = 1.0
        
        return chi2, p_value
    
    def analyze_product(
        self, 
        product_name: str,
        role: str = "AGENTE"
    ) -> Dict:
        """
        Analisa padrões para um produto específico
        
        Args:
            product_name: Nome do produto
            role: 'AGENTE' ou 'CLIENTE'
            
        Returns:
            Dict com análise completa por categoria
        """
        log.info(f"Analisando padrões - produto: {product_name}, papel: {role}")
        
        # Busca transcrições ganhas
        ganhas_rows = self.db.get_product_transcripts_by_role(
            product_name=product_name,
            outcome="ganha",
            role=role
        )
        
        # Busca transcrições perdidas
        perdidas_rows = self.db.get_product_transcripts_by_role(
            product_name=product_name,
            outcome="perdida",
            role=role
        )
        
        if not ganhas_rows or not perdidas_rows:
            log.warning(f"Produto '{product_name}': dados insuficientes (G={len(ganhas_rows)}, P={len(perdidas_rows)})")
            return {}
        
        # Extrai keywords
        ganhas_keywords = [self.extract_keywords(row['texto']) for row in ganhas_rows]
        perdidas_keywords = [self.extract_keywords(row['texto']) for row in perdidas_rows]
        
        # Agrega contagens
        ganha_counts = Counter()
        for kw_dict in ganhas_keywords:
            ganha_counts.update(kw_dict.keys())
        
        perdida_counts = Counter()
        for kw_dict in perdidas_keywords:
            perdida_counts.update(kw_dict.keys())
        
        # Analisa por categoria
        category_results = {}
        
        for cat_name, cat_patterns in self.categories.items():
            cat_keywords = set(cat_patterns.keys())
            relevant_keywords = cat_keywords & (set(ganha_counts.keys()) | set(perdida_counts.keys()))
            
            if not relevant_keywords:
                continue
            
            patterns = []
            for kw in relevant_keywords:
                n_ganha_com = ganha_counts[kw]
                n_ganha_sem = len(ganhas_keywords) - n_ganha_com
                n_perdida_com = perdida_counts[kw]
                n_perdida_sem = len(perdidas_keywords) - n_perdida_com
                
                freq_ganha = n_ganha_com / len(ganhas_keywords) * 100
                freq_perdida = n_perdida_com / len(perdidas_keywords) * 100
                diff = freq_ganha - freq_perdida
                
                chi2, p_value = self.chi_square_test(
                    n_ganha_com, n_ganha_sem,
                    n_perdida_com, n_perdida_sem
                )
                
                patterns.append({
                    'keyword': kw,
                    'freq_ganha': freq_ganha,
                    'freq_perdida': freq_perdida,
                    'diff': diff,
                    'chi2': chi2,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                })
            
            # Ordena por impacto
            category_results[cat_name] = sorted(patterns, key=lambda x: -abs(x['diff']))
        
        return {
            "product_name": product_name,
            "role": role,
            "n_ganha": len(ganhas_keywords),
            "n_perdida": len(perdidas_keywords),
            "win_rate": len(ganhas_keywords) / (len(ganhas_keywords) + len(perdidas_keywords)),
            "categories": category_results
        }
    
    def analyze_all_products(self, role: str = "AGENTE") -> Dict[str, Dict]:
        """
        Analisa padrões para todos os produtos
        
        Args:
            role: 'AGENTE' ou 'CLIENTE'
            
        Returns:
            Dict mapeando produto -> análise
        """
        log.info(f"Analisando padrões para todos os produtos - papel: {role}")
        
        # Busca produtos elegíveis
        products = self.db.get_products(min_calls=settings_v3.MIN_CALLS_PER_PRODUCT)
        
        results = {}
        
        for product in products:
            product_name = product['product_name']
            
            try:
                analysis = self.analyze_product(product_name, role)
                if analysis:
                    results[product_name] = analysis
            except Exception as e:
                log.error(f"Erro ao analisar produto '{product_name}': {e}")
        
        log.info(f"Análise concluída para {len(results)} produtos")
        
        return results
    
    def get_top_winning_patterns(
        self,
        product_name: str,
        n: int = 10,
        role: str = "AGENTE"
    ) -> List[Dict]:
        """
        Retorna top N padrões vencedores para um produto
        
        Args:
            product_name: Nome do produto
            n: Número de patterns a retornar
            role: Papel do speaker
            
        Returns:
            Lista de patterns ordenada por impacto
        """
        analysis = self.analyze_product(product_name, role)
        
        if not analysis or "categories" not in analysis:
            return []
        
        # Coleta todos os padrões significativos e positivos
        all_winners = []
        
        for cat_name, patterns in analysis["categories"].items():
            winners = [
                p for p in patterns 
                if p['diff'] > settings_v3.MIN_DIFF_PERCENTAGE 
                and p['significant']
            ]
            
            for w in winners:
                w['category'] = cat_name
                all_winners.append(w)
        
        # Ordena por diferença e retorna top N
        all_winners.sort(key=lambda x: -x['diff'])
        
        return all_winners[:n]
    
    def get_top_losing_patterns(
        self,
        product_name: str,
        n: int = 10,
        role: str = "AGENTE"
    ) -> List[Dict]:
        """
        Retorna top N padrões perdedores para um produto
        
        Args:
            product_name: Nome do produto
            n: Número de patterns a retornar
            role: Papel do speaker
            
        Returns:
            Lista de patterns ordenada por impacto negativo
        """
        analysis = self.analyze_product(product_name, role)
        
        if not analysis or "categories" not in analysis:
            return []
        
        # Coleta todos os padrões significativos e negativos
        all_losers = []
        
        for cat_name, patterns in analysis["categories"].items():
            losers = [
                p for p in patterns 
                if p['diff'] < -settings_v3.MIN_DIFF_PERCENTAGE 
                and p['significant']
            ]
            
            for l in losers:
                l['category'] = cat_name
                all_losers.append(l)
        
        # Ordena por diferença (mais negativo primeiro)
        all_losers.sort(key=lambda x: x['diff'])
        
        return all_losers[:n]
    
    def compare_products(
        self,
        product_a: str,
        product_b: str,
        role: str = "AGENTE"
    ) -> Dict:
        """
        Compara padrões entre dois produtos
        
        Args:
            product_a: Nome do primeiro produto
            product_b: Nome do segundo produto
            role: Papel do speaker
            
        Returns:
            Dict com diferenças significativas
        """
        log.info(f"Comparando produtos: {product_a} vs {product_b}")
        
        analysis_a = self.analyze_product(product_a, role)
        analysis_b = self.analyze_product(product_b, role)
        
        if not analysis_a or not analysis_b:
            return {}
        
        differences = []
        
        # Para cada categoria, compara patterns
        all_cats = set(analysis_a["categories"].keys()) | set(analysis_b["categories"].keys())
        
        for cat_name in all_cats:
            patterns_a = {p['keyword']: p for p in analysis_a["categories"].get(cat_name, [])}
            patterns_b = {p['keyword']: p for p in analysis_b["categories"].get(cat_name, [])}
            
            all_kw = set(patterns_a.keys()) | set(patterns_b.keys())
            
            for kw in all_kw:
                if kw in patterns_a and kw in patterns_b:
                    diff_a = patterns_a[kw]['diff']
                    diff_b = patterns_b[kw]['diff']
                    
                    # Se há grande diferença de impacto entre produtos
                    if abs(diff_a - diff_b) > 15:
                        differences.append({
                            'keyword': kw,
                            'category': cat_name,
                            'diff_a': diff_a,
                            'diff_b': diff_b,
                            'gap': abs(diff_a - diff_b),
                            'more_important_in': product_a if diff_a > diff_b else product_b
                        })
        
        # Ordena por gap
        differences.sort(key=lambda x: -x['gap'])
        
        return {
            "product_a": product_a,
            "product_b": product_b,
            "role": role,
            "differences": differences
        }
    
    def export_results(self, results: Dict, output_path: str):
        """
        Exporta resultados para JSON
        
        Args:
            results: Resultados da análise
            output_path: Caminho do arquivo de saída
        """
        import json
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        log.info(f"Resultados exportados para: {output_path}")

