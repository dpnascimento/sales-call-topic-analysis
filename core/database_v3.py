"""
DatabaseManager
Inclui métodos específicos para análise por produto e visões de embedding
"""
import logging
from typing import Any, List, Optional, Dict
from contextlib import contextmanager
import psycopg
from psycopg.rows import dict_row

from config import settings_v3

log = logging.getLogger(__name__)


class DatabaseManagerV3:
    """Gerencia conexões e operações de banco de dados"""
    
    def __init__(self):
        self.conn = None
    
    def connect(self):
        """Estabelece conexão com o banco"""
        try:
            self.conn = psycopg.connect(
                host=settings_v3.PG_HOST,
                port=settings_v3.PG_PORT,
                user=settings_v3.PG_USER,
                password=settings_v3.PG_PASS,
                dbname=settings_v3.PG_DB
            )
            self.conn.autocommit = False
            log.info("Conexão estabelecida com sucesso")
            return self.conn
        except psycopg.Error as e:
            log.error(f"Erro ao conectar: {e}")
            raise
    
    def close(self):
        """Fecha conexão"""
        if self.conn:
            self.conn.close()
            log.info("Conexão fechada")
    
    @contextmanager
    def get_cursor(self, row_factory=dict_row):
        """Context manager para cursor"""
        cursor = self.conn.cursor(row_factory=row_factory)
        try:
            yield cursor
        finally:
            cursor.close()
    
    def execute_safe(self, sql: str, params: Any = None, fetch: bool = True) -> Optional[List[Dict]]:
        """
        Executa SQL com tratamento de erro
        
        Args:
            sql: Query SQL
            params: Parâmetros
            fetch: Se True, retorna resultados
            
        Returns:
            Lista de dicionários se fetch=True, None caso contrário
        """
        # Log SQL para troubleshooting
        log.debug("=" * 80)
        log.debug("EXECUTANDO SQL:")
        log.debug(sql)
        if params:
            log.debug(f"PARÂMETROS: {params}")
        log.debug("=" * 80)
        
        try:
            with self.get_cursor() as cur:
                cur.execute(sql, params)
                if fetch:
                    results = cur.fetchall()
                    log.debug(f"✓ Query retornou {len(results)} registros")
                    return results
                self.conn.commit()
                log.debug("✓ Query executada com sucesso (sem fetch)")
                return None
        except psycopg.Error as e:
            self.conn.rollback()
            log.error(f"❌ Erro ao executar SQL: {e}")
            log.debug(f"SQL que falhou: {sql[:500]}...")
            if params:
                log.debug(f"Parâmetros: {params}")
            raise
        except Exception as e:
            self.conn.rollback()
            log.error(f"❌ Erro inesperado: {e}")
            raise
    
    def execute_batch(self, sql: str, params_list: List[tuple], batch_size: int = 1000):
        """
        Executa operações em batch
        
        Args:
            sql: Query SQL
            params_list: Lista de tuplas de parâmetros
            batch_size: Tamanho do batch
        """
        try:
            with self.get_cursor() as cur:
                for i in range(0, len(params_list), batch_size):
                    batch = params_list[i:i + batch_size]
                    cur.executemany(sql, batch)
                    self.conn.commit()
                    log.debug(f"Batch {i//batch_size + 1} executado ({len(batch)} registros)")
        except psycopg.Error as e:
            self.conn.rollback()
            log.error(f"Erro em batch: {e}")
            raise
    
    # =========================
    # MÉTODOS ESPECÍFICOS
    # =========================
    
    # =========================
    # REGRAS DE NEGÓCIO (explícitas no código)
    # =========================
    # 1. Chamadas válidas: duração >= 10 segundos
    # 2. Outcomes: 'ganha' ou 'perdida' (de call_outcomes)
    # 3. Produtos: nome do pipeline (de pipelines)
    # 4. Todas as chamadas válidas (não apenas 1 por deal)
    #    - Vendas complexas contribuem mais (apropriado para patterns)
    
    def get_products(self, min_calls: int = None) -> List[Dict]:
        """
        Retorna lista de produtos com número de chamadas
        
        Regras de negócio explícitas:
        - Apenas chamadas com duração >= 10s
        - Com outcome definido (ganha/perdida)
        - Agrupadas por produto (pipeline)
        
        Args:
            min_calls: Filtrar por mínimo de chamadas
            
        Returns:
            Lista de dicts com {product_name, n_calls, n_wins, win_rate}
        """
        min_filter = f"HAVING COUNT(DISTINCT cr.call_id) >= {min_calls}" if min_calls else ""
        
        sql = f"""
        SELECT 
            p.name AS product_name,
            COUNT(DISTINCT cr.call_id) as n_calls,
            COUNT(DISTINCT CASE WHEN co.outcome = 'ganha' THEN cr.call_id END) as n_wins,
            COUNT(DISTINCT CASE WHEN co.outcome = 'perdida' THEN cr.call_id END) as n_losses
        FROM public.call_records cr
        JOIN public.call_outcomes co ON co.deal_id = cr.deal_id
        LEFT JOIN public.deals d ON cr.deal_id = d.deal_id
        LEFT JOIN public.pipelines p ON d.pipeline = p.pipeline_id
        WHERE p.name IS NOT NULL
          AND cr.call_duration >= 10  -- Regra: chamadas válidas
          AND co.outcome IN ('ganha', 'perdida')  -- Regra: outcomes conhecidos
        GROUP BY p.name
        {min_filter}
        ORDER BY n_calls DESC
        """
        
        rows = self.execute_safe(sql, fetch=True)
        
        for row in rows:
            row['win_rate'] = row['n_wins'] / row['n_calls'] if row['n_calls'] > 0 else 0.0
        
        return rows
    
    def get_calls_by_product_and_outcome(
        self, 
        product_name: str, 
        outcome: str, 
        embedding_view: str = "full"
    ) -> List[Dict]:
        """
        Retorna chamadas de um produto com outcome específico
        
        Args:
            product_name: Nome do produto
            outcome: 'ganha' ou 'perdida'
            embedding_view: 'full', 'agent' ou 'client'
            
        Returns:
            Lista de chamadas com embeddings
        """
        emb_col = settings_v3.EMBEDDING_COLUMN_MAP[embedding_view]
        valid_col = settings_v3.EMBEDDING_VALID_MAP[embedding_view]
        
        sql = f"""
        SELECT DISTINCT ON (e.call_id)
            e.call_id,
            e.{emb_col}::text AS embedding_text,
            co.outcome,
            p.name AS product_name,
            cr.sales_rep_id,
            cr.recorded_at
        FROM public.call_embeddings_v2 e
        JOIN public.call_records cr ON cr.call_id = e.call_id
        JOIN public.call_outcomes co ON co.deal_id = cr.deal_id
        LEFT JOIN public.deals d ON cr.deal_id = d.deal_id
        LEFT JOIN public.pipelines p ON d.pipeline = p.pipeline_id
        WHERE p.name = %(product_name)s
          AND co.outcome = %(outcome)s
          AND cr.call_duration >= 10  -- Regra: chamadas válidas
          AND e.{valid_col} = TRUE
          AND e.{emb_col} IS NOT NULL
        ORDER BY e.call_id, co.outcome_date DESC NULLS LAST, cr.recorded_at
        """
        
        return self.execute_safe(sql, {"product_name": product_name, "outcome": outcome}, fetch=True)
    
    def get_product_transcripts_by_role(
        self,
        product_name: str,
        outcome: str,
        role: str = "AGENTE"
    ) -> List[Dict]:
        """
        Retorna transcrições agregadas por chamada para um produto
        
        Args:
            product_name: Nome do produto
            outcome: 'ganha' ou 'perdida'
            role: 'AGENTE' ou 'CLIENTE'
            
        Returns:
            Lista com call_id, outcome, texto agregado
        """
        sql = """
        WITH latest_outcomes AS (
            SELECT DISTINCT ON (deal_id)
                deal_id,
                outcome
            FROM public.call_outcomes
            ORDER BY deal_id, outcome_date DESC NULLS LAST
        )
        SELECT 
            t.call_id,
            co.outcome,
            p.name AS product_name,
            string_agg(
                t.text,
                ' '
                ORDER BY t.start_time ASC
            ) AS texto
        FROM public.call_transcripts_deepgram t
        JOIN public.call_records cr ON cr.call_id = t.call_id
        JOIN latest_outcomes co ON co.deal_id = cr.deal_id
        LEFT JOIN public.deals d ON cr.deal_id = d.deal_id
        LEFT JOIN public.pipelines p ON d.pipeline = p.pipeline_id
        LEFT JOIN public.call_speakers cs ON cs.call_id = t.call_id AND cs.speaker_id = t.speaker_id
        WHERE p.name = %(product_name)s
          AND co.outcome = %(outcome)s
          AND cr.call_duration >= 10  -- Regra: chamadas válidas
          AND COALESCE(cs.role, CASE WHEN t.speaker_id = 0 THEN 'AGENTE' ELSE 'CLIENTE' END) = %(role)s
          AND LENGTH(TRIM(t.text)) > 0
        GROUP BY t.call_id, co.outcome, p.name
        HAVING LENGTH(string_agg(t.text, ' ')) >= 200
        """
        
        return self.execute_safe(
            sql, 
            {"product_name": product_name, "outcome": outcome, "role": role}, 
            fetch=True
        )
    
    def get_all_embeddings_by_view(
        self,
        embedding_view: str = "full",
        outcome: str = None,
        product_name: str = None,
        limit: int = None
    ) -> List[Dict]:
        """
        Retorna todos os embeddings para uma visão específica
        
        Args:
            embedding_view: 'full', 'agent' ou 'client'
            outcome: Filtrar por outcome (opcional)
            product_name: Filtrar por produto (opcional)
            limit: Limite de registros (opcional)
            
        Returns:
            Lista de embeddings com metadados
        """
        emb_col = settings_v3.EMBEDDING_COLUMN_MAP[embedding_view]
        valid_col = settings_v3.EMBEDDING_VALID_MAP[embedding_view]
        
        # Regras de negócio base (sempre aplicadas)
        filters = [
            f"e.{valid_col} = TRUE",
            f"e.{emb_col} IS NOT NULL",
            "cr.call_duration >= 10",  # Regra: chamadas válidas
            "co.outcome IN ('ganha', 'perdida')"  # Regra: outcomes conhecidos
        ]
        params = {}
        
        if outcome:
            filters.append("co.outcome = %(outcome)s")
            params["outcome"] = outcome
        
        if product_name:
            filters.append("p.name = %(product_name)s")
            params["product_name"] = product_name
        
        where_clause = " AND ".join(filters)
        limit_clause = f"LIMIT %(limit)s" if limit else ""
        if limit:
            params["limit"] = limit
        
        sql = f"""
        SELECT DISTINCT ON (e.call_id)
            e.call_id,
            e.{emb_col}::text AS embedding_text,
            co.outcome,
            p.name AS product_name,
            cr.sales_rep_id,
            cr.recorded_at
        FROM public.call_embeddings_v2 e
        JOIN public.call_records cr ON cr.call_id = e.call_id
        JOIN public.call_outcomes co ON co.deal_id = cr.deal_id
        LEFT JOIN public.deals d ON cr.deal_id = d.deal_id
        LEFT JOIN public.pipelines p ON d.pipeline = p.pipeline_id
        WHERE {where_clause}
        ORDER BY e.call_id, co.outcome_date DESC NULLS LAST, cr.recorded_at
        {limit_clause}
        """
        
        return self.execute_safe(sql, params, fetch=True)

