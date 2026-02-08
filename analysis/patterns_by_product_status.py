"""
Análise de padrões por produto E status da oportunidade
Extensão da análise por produto com segmentação adicional por outcome
"""
import logging
from typing import Dict, List

from core.database_v3 import DatabaseManagerV3
from analysis.patterns_by_product import PatternsByProductAnalyzer
from config import settings_v3

log = logging.getLogger(__name__)


class PatternsByProductStatusAnalyzer:
    """
    Analisa padrões considerando produto E status da oportunidade
    Útil para entender patterns específicos de cada contexto
    """
    
    def __init__(self, db: DatabaseManagerV3):
        self.db = db
        self.base_analyzer = PatternsByProductAnalyzer(db)
    
    def analyze_product_by_status(
        self,
        product_name: str,
        role: str = "AGENTE"
    ) -> Dict:
        """
        Analisa padrões de um produto separando por status
        
        Args:
            product_name: Nome do produto
            role: 'AGENTE' ou 'CLIENTE'
            
        Returns:
            Dict com análise por status
        """
        log.info(f"Analisando produto por status - produto: {product_name}, papel: {role}")
        
        results = {
            "product_name": product_name,
            "role": role,
            "by_status": {}
        }
        
        for outcome in settings_v3.OPPORTUNITY_STATUSES:
            # Reutiliza lógica base mas para cada status
            analysis = self.base_analyzer.analyze_product(product_name, role)
            
            if analysis:
                results["by_status"][outcome] = {
                    "n_samples": analysis.get("n_ganha" if outcome == "ganha" else "n_perdida", 0),
                    # Aqui poderíamos adicionar análises específicas por status
                    # Por exemplo, patterns que aparecem APENAS em ganhas ou APENAS em perdidas
                }
        
        return results
    
    def find_status_specific_patterns(
        self,
        product_name: str,
        role: str = "AGENTE"
    ) -> Dict:
        """
        Identifica patterns que são exclusivos de cada status
        
        Args:
            product_name: Nome do produto
            role: Papel do speaker
            
        Returns:
            Dict com patterns exclusivos de cada status
        """
        log.info(f"Buscando patterns específicos por status - produto: {product_name}")
        
        # Analisa padrões completos
        full_analysis = self.base_analyzer.analyze_product(product_name, role)
        
        if not full_analysis or "categories" not in full_analysis:
            return {}
        
        # Separa patterns vencedores e perdedores
        winning_patterns = []
        losing_patterns = []
        
        for cat_name, patterns in full_analysis["categories"].items():
            for p in patterns:
                if p['significant']:
                    if p['diff'] > settings_v3.MIN_DIFF_PERCENTAGE:
                        p['category'] = cat_name
                        winning_patterns.append(p)
                    elif p['diff'] < -settings_v3.MIN_DIFF_PERCENTAGE:
                        p['category'] = cat_name
                        losing_patterns.append(p)
        
        return {
            "product_name": product_name,
            "role": role,
            "winning_exclusive": sorted(winning_patterns, key=lambda x: -x['diff'])[:10],
            "losing_exclusive": sorted(losing_patterns, key=lambda x: x['diff'])[:10]
        }
    
    def compare_status_patterns_across_products(
        self,
        role: str = "AGENTE"
    ) -> Dict:
        """
        Compara patterns de status através de produtos
        Identifica padrões universais vs específicos de produto
        
        Args:
            role: Papel do speaker
            
        Returns:
            Dict com padrões universais e específicos
        """
        log.info(f"Comparando patterns de status através de produtos - papel: {role}")
        
        # Busca produtos elegíveis
        products = self.db.get_products(min_calls=settings_v3.MIN_CALLS_PER_PRODUCT)
        
        # Coleta patterns de todos os produtos
        all_winning_patterns = []
        all_losing_patterns = []
        
        for product in products:
            product_name = product['product_name']
            
            try:
                specific = self.find_status_specific_patterns(product_name, role)
                
                if not specific:
                    continue
                
                for p in specific.get("winning_exclusive", []):
                    p['from_product'] = product_name
                    all_winning_patterns.append(p)
                
                for p in specific.get("losing_exclusive", []):
                    p['from_product'] = product_name
                    all_losing_patterns.append(p)
                
            except Exception as e:
                log.error(f"Erro ao processar produto '{product_name}': {e}")
        
        # Identifica patterns universais (aparecem em múltiplos produtos)
        from collections import Counter
        
        winning_keywords = Counter([p['keyword'] for p in all_winning_patterns])
        losing_keywords = Counter([p['keyword'] for p in all_losing_patterns])
        
        # Patterns que aparecem em 3+ produtos
        universal_winning = [kw for kw, count in winning_keywords.items() if count >= 3]
        universal_losing = [kw for kw, count in losing_keywords.items() if count >= 3]
        
        # Patterns específicos (aparecem em apenas 1 produto)
        specific_winning = {}
        specific_losing = {}
        
        for p in all_winning_patterns:
            if winning_keywords[p['keyword']] == 1:
                product = p['from_product']
                if product not in specific_winning:
                    specific_winning[product] = []
                specific_winning[product].append(p)
        
        for p in all_losing_patterns:
            if losing_keywords[p['keyword']] == 1:
                product = p['from_product']
                if product not in specific_losing:
                    specific_losing[product] = []
                specific_losing[product].append(p)
        
        return {
            "role": role,
            "universal_patterns": {
                "winning": [
                    {
                        "keyword": kw,
                        "count": winning_keywords[kw],
                        "products": list(set([p['from_product'] for p in all_winning_patterns if p['keyword'] == kw]))
                    }
                    for kw in universal_winning
                ],
                "losing": [
                    {
                        "keyword": kw,
                        "count": losing_keywords[kw],
                        "products": list(set([p['from_product'] for p in all_losing_patterns if p['keyword'] == kw]))
                    }
                    for kw in universal_losing
                ]
            },
            "product_specific_patterns": {
                "winning": specific_winning,
                "losing": specific_losing
            }
        }
    
    def generate_insights_by_product_status(
        self,
        role: str = "AGENTE"
    ) -> Dict:
        """
        Gera insights consolidados por produto e status
        
        Args:
            role: Papel do speaker
            
        Returns:
            Dict com insights estratégicos
        """
        log.info(f"Gerando insights por produto e status - papel: {role}")
        
        # Analisa todos os produtos
        all_products_analysis = self.base_analyzer.analyze_all_products(role)
        
        # Compara patterns universais vs específicos
        cross_product = self.compare_status_patterns_across_products(role)
        
        insights = {
            "role": role,
            "n_products_analyzed": len(all_products_analysis),
            "universal_patterns": cross_product["universal_patterns"],
            "product_specific_insights": {}
        }
        
        # Gera insights por produto
        for product_name, analysis in all_products_analysis.items():
            win_rate = analysis.get("win_rate", 0)
            
            # Top estratégias vencedoras
            top_winning = self.base_analyzer.get_top_winning_patterns(product_name, n=5, role=role)
            
            # Top estratégias perdedoras
            top_losing = self.base_analyzer.get_top_losing_patterns(product_name, n=5, role=role)
            
            insights["product_specific_insights"][product_name] = {
                "win_rate": win_rate,
                "n_ganha": analysis.get("n_ganha", 0),
                "n_perdida": analysis.get("n_perdida", 0),
                "top_winning_strategies": top_winning,
                "top_losing_strategies": top_losing,
                "recommendations": self._generate_recommendations(
                    product_name, 
                    win_rate, 
                    top_winning, 
                    top_losing
                )
            }
        
        return insights
    
    def _generate_recommendations(
        self,
        product_name: str,
        win_rate: float,
        top_winning: List[Dict],
        top_losing: List[Dict]
    ) -> List[str]:
        """Gera recomendações para um produto"""
        recommendations = []
        
        # Recomendações baseadas em win rate
        if win_rate < 0.4:
            recommendations.append(f"⚠️ Win rate baixo ({win_rate:.1%}): revisar estratégia de vendas")
        elif win_rate > 0.6:
            recommendations.append(f"✅ Win rate saudável ({win_rate:.1%}): manter estratégia atual")
        
        # Recomendações baseadas em winning patterns
        if top_winning:
            top_cat = max(set([p['category'] for p in top_winning]), 
                         key=lambda c: len([p for p in top_winning if p['category'] == c]))
            recommendations.append(f"🎯 Foco na categoria '{top_cat}': maior impacto positivo")
        
        # Alertas baseados em losing patterns
        if top_losing:
            worst = top_losing[0]
            recommendations.append(
                f"⚠️ Evitar '{worst['keyword']}' ({worst['category']}): impacto negativo de {worst['diff']:.1f}%"
            )
        
        return recommendations
    
    def export_results(self, results: Dict, output_path: str):
        """Exporta resultados para JSON"""
        import json
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        log.info(f"Resultados exportados para: {output_path}")

