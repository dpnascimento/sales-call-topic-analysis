"""
Configurações centralizadas

Todas as configurações são lidas de variáveis de ambiente ou valores padrão.
"""
import os
from pathlib import Path
from typing import Set

# =========================
# DIRETÓRIOS
# =========================
ROOT_DIR = Path(__file__).parent.parent

# =========================
# BANCO DE DADOS
# =========================
PG_HOST = os.getenv("PGHOST", "localhost")
PG_PORT = int(os.getenv("PGPORT", "5432"))
PG_USER = os.getenv("PGUSER", "postgres")
PG_PASS = os.getenv("PGPASSWORD", "postgres")
PG_DB = os.getenv("PGDATABASE", "tcc")

# =========================
# EMBEDDINGS V2
# =========================
# Visões disponíveis (labeled foi descartada por baixa qualidade)
EMBEDDING_VIEWS = ["full", "agent", "client"]

# Mapeia nome da visão para coluna no banco
EMBEDDING_COLUMN_MAP = {
    "full": "embedding_full",
    "agent": "embedding_agent", 
    "client": "embedding_client"
}

# Mapeia nome da visão para flag de validade
EMBEDDING_VALID_MAP = {
    "full": "full_valid",
    "agent": "agent_valid",
    "client": "client_valid"
}

# =========================
# ANÁLISE POR PRODUTO
# =========================
# Produtos a analisar (None = todos)
PRODUCTS_TO_ANALYZE = None  # None para todos, ou lista: ["Produto A", "Produto B"]

# Mínimo de chamadas por produto para análise
MIN_CALLS_PER_PRODUCT = 20

# Mínimo de chamadas por produto + status
MIN_CALLS_PER_PRODUCT_STATUS = 10

# Status de oportunidade a analisar
OPPORTUNITY_STATUSES = ["ganha", "perdida"]

# =========================
# ANÁLISE DE PROTÓTIPOS
# =========================
# Calcular protótipos para cada visão
COMPUTE_PROTOTYPES_PER_VIEW = True

# Calcular protótipos por produto
COMPUTE_PROTOTYPES_PER_PRODUCT = True

# Calcular protótipos por produto + status
COMPUTE_PROTOTYPES_PER_PRODUCT_STATUS = True

# Otimizações de performance
COMPUTE_SILHOUETTE = True  # Calcular silhueta (lento para datasets grandes)
SILHOUETTE_SAMPLE_SIZE = 2000  # Amostragem para silhoueta (None = todos os dados)

# =========================
# PADRÕES SEMÂNTICOS
# =========================
# Usar enhanced patterns (80+ patterns em 15 categorias)
USE_ENHANCED_PATTERNS = True

# Chi-square threshold para significância
CHI_SQUARE_THRESHOLD = 3.84  # p < 0.05

# Diferença mínima percentual para considerar relevante
MIN_DIFF_PERCENTAGE = 3.0

# =========================
# VISUALIZAÇÕES
# =========================
# Criar UMAPs para cada visão
CREATE_UMAP_PER_VIEW = True

# Criar UMAPs por produto
CREATE_UMAP_PER_PRODUCT = True

# Parâmetros UMAP
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
UMAP_METRIC = "cosine"

# Tamanho de amostra para UMAP (None = todos)
UMAP_SAMPLE_SIZE = 5000

# =========================
# ANÁLISE DE TÓPICOS (BERTOPIC)
# =========================
# Habilitar análise de tópicos
DO_TOPICS = os.getenv("DO_TOPICS", "true").lower() == "true"

# Usar ligação completa (True) ou enunciados (False)
TOPICS_USE_FULL_CALL = True

# Máximo de documentos a processar
TOPICS_MAX_DOCUMENTS = int(os.getenv("TOPICS_MAX_DOCUMENTS", "10000"))

# Parâmetros UMAP para BERTopic
TOPICS_UMAP_N_NEIGHBORS = int(os.getenv("TOPICS_UMAP_N_NEIGHBORS", "15"))
TOPICS_UMAP_MIN_DIST = float(os.getenv("TOPICS_UMAP_MIN_DIST", "0.0"))

# Parâmetros HDBSCAN
TOPICS_MIN_CLUSTER_SIZE = int(os.getenv("TOPICS_MIN_CLUSTER_SIZE", "30"))
TOPICS_MIN_SAMPLES = int(os.getenv("TOPICS_MIN_SAMPLES", "10"))

# Parâmetros CountVectorizer
TOPICS_MIN_DF = int(os.getenv("TOPICS_MIN_DF", "5"))

# Gerar visualizações HTML
TOPICS_GENERATE_PLOTS = os.getenv("TOPICS_GENERATE_PLOTS", "true").lower() == "true"

# Gerar análise temporal (Topics over Time)
TOPICS_GENERATE_OVERTIME = os.getenv("TOPICS_GENERATE_OVERTIME", "true").lower() == "true"

# Número de bins temporais para análise over time
TOPICS_OVERTIME_BINS = int(os.getenv("TOPICS_OVERTIME_BINS", "20"))

# =========================
# OUTPUTS
# =========================
V3_OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")
V3_DATA_DIR = os.path.join(V3_OUTPUT_DIR, "data")
V3_PLOTS_DIR = os.path.join(V3_OUTPUT_DIR, "plots")
V3_REPORTS_DIR = os.path.join(V3_OUTPUT_DIR, "reports")
V3_TOPICS_DIR = os.path.join(V3_OUTPUT_DIR, "topics")

# Criar diretórios se não existirem
os.makedirs(V3_DATA_DIR, exist_ok=True)
os.makedirs(V3_PLOTS_DIR, exist_ok=True)
os.makedirs(V3_REPORTS_DIR, exist_ok=True)
os.makedirs(V3_TOPICS_DIR, exist_ok=True)

# =========================
# RELATÓRIO TCC
# =========================
# Gerar relatório consolidado para TCC
GENERATE_TCC_REPORT = True

# Formato do relatório (markdown, html, both)
TCC_REPORT_FORMAT = "both"

# Incluir gráficos no relatório
TCC_INCLUDE_PLOTS = True

# Incluir estatísticas detalhadas
TCC_INCLUDE_DETAILED_STATS = True

# =========================
# LOGGING
# =========================
LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG")  # DEBUG para troubleshooting SQL
LOG_FILE = os.path.join(V3_OUTPUT_DIR, "v3_pipeline.log")

# =========================
# STOPWORDS PORTUGUÊS
# =========================
PT_STOPWORDS: Set[str] = {
    'a', 'o', 'os', 'as', 'um', 'uma', 'de', 'da', 'do', 'das', 'dos', 
    'em', 'no', 'na', 'nos', 'nas', 'para', 'por', 'com', 'sem', 'sobre',
    'entre', 'e', 'ou', 'mas', 'que', 'se', 'eu', 'tu', 'ele', 'ela',
    'nós', 'vós', 'eles', 'elas', 'me', 'te', 'lhe', 'nos', 'vos', 'lhes',
    'este', 'esta', 'isso', 'aquele', 'aquela', 'aquilo', 'também', 'já',
    'não', 'sim', 'como', 'quando', 'onde', 'porque', 'porquê', 'pois',
    'bom', 'dia', 'boa', 'tarde', 'noite', 'oi', 'olá', 'tchau',
    'obrigado', 'obrigada', 'valeu', 'blz', 'beleza', 'ok', 'certo',
    'tá', 'tá bom', 'uhum', 'aham', 'né', 'tudo', 'bem', 'até',
    'até mais', 'até logo', 'só', 'tipo', 'cotação', 'seguro', 'seguro de carga', 
    'seguro de transporte', 'seguro garantia', 'seguro de vida', 'aí', 'entendi', 
    'ah', 'então', 'alguma', 'tem', 'mais', 'assim', 'valor', 'entendeu', 'mil', 
    'tô', 'termo', 'execuo', 'arthur', 'obra', 'caminhes', 'carregar', 'carros', 
    'contrato', 'tenho', 'sabe', 'exata', 'pedir', 'passar', 'giovana', 'prefeitura', 
    'mandando', 'demanda', 'seguros', 'solicitação', 'cotao', 'hoje', 'amanhã', 
    'amanha', 'depois', 'antes', 'encerramento', 'perfeito'
}

STOPWORDS_CONVERSACAO: Set[str] = {
    # Artigos e contrações
    'a', 'o', 'as', 'os', 'um', 'uma', 'uns', 'umas', 'ao', 'à', 'aos', 'às', 
    'do', 'da', 'dos', 'das', 'no', 'na', 'nos', 'nas', 'pelo', 'pela', 'pelos', 'pelas',
    
    # Pronomes
    'eu', 'tu', 'ele', 'ela', 'nós', 'vós', 'eles', 'elas', 'me', 'mim', 'comigo', 
    'te', 'ti', 'contigo', 'se', 'si', 'consigo', 'lhe', 'nos', 'vos', 'lhes', 
    'meu', 'minha', 'meus', 'minhas', 'teu', 'tua', 'teus', 'tuas', 'seu', 'sua', 
    'seus', 'suas', 'nosso', 'nossa', 'nossos', 'nossas', 'vosso', 'vossa', 'vossos', 'vossas',
    
    # Preposições
    'de', 'em', 'por', 'para', 'com', 'sem', 'sob', 'sobre', 'entre', 'até', 
    'desde', 'contra', 'perante', 'através', 'além', 'dentro', 'fora', 'perto', 
    'longe', 'durante',
    
    # Conjunções
    'e', 'mas', 'ou', 'porque', 'pois', 'que', 'se', 'como', 'quando', 'embora', 
    'porém', 'todavia', 'contudo', 'então', 'também', 'apesar', 'caso', 'além disso', 
    'portanto', 'logo', 'ainda que', 'a fim de', 'com o objetivo de',
    
    # Verbos auxiliares e comuns
    'ser', 'estar', 'ter', 'haver', 'ir', 'vir', 'fazer', 'poder', 'dever', 
    'querer', 'saber', 'está', 'estão', 'tem', 'tenho', 'é',
    
    # Demonstrativos
    'esse', 'essa', 'isso', 'este', 'esta', 'isto', 'aquele', 'aquela', 'aquilo', 
    'aqueles', 'aquelas', 'deste', 'desta', 'destes', 'destas', 'disso', 'daquilo', 
    'nisto', 'naquilo',
    
    # Localizadores
    'lá', 'aqui', 'ali', 'onde', 'aonde',
    
    # Interjeições e expressões comuns
    'ah', 'oh', 'ei', 'oi', 'olá', 'opa', 'eita', 'nossa', 'caramba', 'poxa', 
    'uau', 'xi', 'ih', 'ué', 'hein', 'que que é isso', 'quê',
    
    # Gírias e expressões coloquiais
    'cara', 'mano', 'mina', 'véi', 'velho', 'brother', 'meu', 'meu chapa', 
    'bacana', 'maneiro', 'massa', 'show', 'top', 'legal', 'beleza', 
    'joia', 'firmeza', 'daora', 'dahora', 'da hora', 'irado', 'animal', 'monstro', 'mito', 'lenda',
    
    # Abreviações e siglas comuns
    'pq', 'tb', 'tbm', 'vc', 'cê', 'ce', 'c', 'mt', 'mto', 'mta', 'td', 'tdo', 
    'tda', 'hj', 'amg', 'amigo', 'amiga', 'bjs', 'vlw', 'flw', 'fvr', 'por favor', 
    'abs', 'vdd', 'aff', 'pfv', 'pfvr', 'sla', 'slc', 'sdds', 'tmj', 'tamo junto', 
    'fds', 'blz', 'falou', 'fmz', 'mds', 'oxe', 'oxi', 'vey', 'véi', 'vix', 'vish', 
    'vcs', 'tá', 'tô', 'tamos', 'tamo', 'cadê', 'd', 'pro', 'pra', 'q', 'qualé', 
    'tipo', 'né', 'qnd', 'aki', 'vamo', 'vambora', 'partiu', 'bora', 'falô', 'blza', 
    'obg', 'ñ', 'num', 'msm', 'sdd', 'agr', 'poxa', 'po', 'uai', 'eita',
    
    # Regionalismos
    'bão', 'ocê', 'oxente', 'égua', 'bah', 'tchê', 'aham', 'ahã', 
    'orra meu', 'home', 'homi', 'muié', 'muler', 'visse', 'tu', 'num é', 'nera',
    
    # Marcadores de discurso
    'tipo assim', 'aí', 'pois é', 'quer dizer', 'sabe', 'entende', 'sacou', 
    'tá ligado', 'ok', 'okay', 'tranquilo', 'suave', 'de boa', 'fechou', 
    'combinado', 'entendido',
    
    # Expressões de afirmação e negação
    's', 'n', 'yep', 'nop', 'ahan', 'nops', 'de jeito nenhum', 'com certeza', 
    'claro', 'óbvio', 'lógico', 'evidente', 'sem dúvida', 'quem sabe', 'vai ver', 
    'pode ser', 'anos', 'possível',
    
    # Expressões de quantidade e intensidade
    'bastante', 'pra caramba', 'pra cacete', 'pra dedéu', 'pácas', 
    'um monte', 'um bocado', 'um tiquinho', 'um cadinho', 'um tico',
    
    # Expressões de concordância
    'tá bom', 'tá bem', 'tá certo', 'pode crer', 'tô dentro', 'topo', 'vamos', 
    'bora lá', 'tá de boa',
    
    # Expressões de discordância
    'tá nada', 'que nada', 'nem a pau', 'nem ferrando', 'nem morto', 
    'tá louco', 'tá doido', 'nem pensar',
    
    # Expressões de despedida
    'tchau', 'até logo', 'até mais', 'fui', 'to indo', 'té mais', 'inté', 
    'bye', 'xau', 'beijo', 'abraço', 'abç', 'fique com Deus', 'vai com Deus', 
    'se cuida', 'se cuide', 'fica bem', 'fique bem',
    
    # Expressões de saudação
    'e aí', 'fala', 'fala aí', 'fala tu', 'salve', 'quali', 'como vai', 
    'como tá', 'tudo bem', 'tudo bom', 'eae', 'coé', 'alô',
    
    # Expressões de agradecimento
    'valeu', 'obrigado', 'obrigada', 'brigado', 'brigadão', 'muito obrigado', 
    'grato', 'gratidão', 'thanks', 'thx',
    
    # Termos de internet
    'rs', 'kkkk', 'haha', 'hehe', 'lol', 'risos', 'kkk', 'hahaha',

    # Expressões expandidas (mais informais)
    'você', 'vocês', 'porquê', 'entaum', 'bls', 'dexa', 'tá ligado', 'tá sabendo', 
    'tá com', 'tá ok', 'cê tá', 'ocê', 'ocês', 'nois', 'nois é', 'é nois', 
    'na moral', 'escuta só', 'né não', 'nu', 'rlx', 'susto', 'tru', 'brodi', 
    'eu to', 'eu tô', 'nois tá', 'nóis tá', 'tu tá', 'tu vai', 'c vai', 'tu vamo', 
    'tá suave', 'tá de boaça', 'ô',
    
    # === NOMES PRÓPRIOS (funcionários, clientes) ===
    'tiani', 'tiane', 'ana', 'vanessa', 'tatiana', 'mirella', 'mirela', 'ludiane', 'nicolás', 'nicolas',
    'anny', 'gabriel', 'rodrigo', 'fernanda', 'matheus', 'julio', 'flavio', 'flávio', 'gabriela',
    'aline', 'paulo', 'carlos', 'maria', 'viviane', 'giovana', 'rafael', 'bárbara', 'anderson',
    'guilherme', 'bruna', 'andré', 'jorge', 'antônio', 'mateus', 'gustavo', 'francisco', 'fernando',
    'ricardo', 'edson', 'henrique', 'felipe', 'daniel', 'thiago', 'william', 'lucas', 'rogério',
    'bruno', 'emerson', 'diadora', 'bianca', 'nádia', 'luciane', 'sandra', 'eliane', 'alexandre',
    'rafaela', 'anos', 'pesquisa', 'tiana',
    
    # === NOMES DE EMPRESAS/MARCAS ===
    'mutuseguros', 'mutu', 'mutuus', 'mutus', 'muta', 'damuto', 'sou mutos', 'multos', 'multa',
    'multiseguros', 'multisseguros', 'motosseguros', 'motoseguros', 'sura', 'tóquio', 'namuto', 'amuto',
    
    # === FRASES DE CONTATO/SAUDAÇÃO ===
    'motivo contato', 'motivo contato respeito', 'contato respeito', 'contato respeito plataforma',
    'entrando contato respeito', 'entrando contato carga', 'contato carga', 'ótimo entrando contato',
    'ótimo entrando', 'estou ligando', 'chama gente', 'chamo rafael', 'ficar contato', 'contato viu',
    'faça contato', 'agradeço contato', 'contar gente',
    
    # === FRASES DE DISPOSIÇÃO/CORTESIA ===
    'fico disposição', 'vai entrar', 'gente recebeu', 'gente conversar', 'gente verificou',
    'possível fará contato', 'possível fará', 'fará contato', 'fará', 'dou retorno',
    'agradeço informações', 'pegar algumas informações', 'pegar algumas', 'graças deus minutinho',
    'deus minutinho', 'qual melhor', 'agradeço qualquer', 'agradeço qualquer forma', 'qualquer forma',
    'agradeço atenção', 'gente fica', 'gente fica disposição', 'entrando gente', 'entrando gente recebeu',
    'gente passando', 'fico aguardo', 'retorno viu', 'possível viu', 'viu agradeço',
    'aguardar', 'vai seguir', 'gente sempre', 'fez gente', 'garantia fez', 'garantia fez gente',
    'necessidade ajudando', 'visite', 'site garantia', 'via', 'entrando', 'ótimo entrando',
    'algumas informações nesse', 'motivo', 'plataforma', 'carro', 'entendo', 'erro',
    
    # === TERMOS DE CARGA/PRODUTO (genéricos sem valor semântico) ===
    'interesse carga', 'carga queria', 'carga estou', 'carga precisa', 'carga  precisa', 'carga seria',
    'carga mesmo', 'carga mesmo precisa', 'carga foi', 'dono carga', 'pedido carga', 'houve pedido carga',
    'carga carga', 'tiane carga', 'sou carga', 'ótimo carga', 'houve carga', 'cargo mesmo precisa', 'carga',
    'meses',
    
    # === TERMOS DE SEGURO/PROCESSO (muito genéricos) ===
    'vida seria', 'vida grupo', 'tinha vida', 'solicitou vida', 'multa vida', 'multa tinha vida',
    'mandar multa', 'multas', 'multa jóia', 'multa tinha', 'seguradora vai', 'vai cobrar', 'seria cobertura',
    'data pólice', 'referente garantia', 'respeito plataforma', 'respeito carga', 'nome empresa',
    'entender melhor produto', 'melhor produto', 'quetia entender melhor', 'material interesse',
    'acessou material', 'pessoa física', 'pessoa jurídica', 'pessoa física jurídica', 'física jurídica',
    'física', 'jurídica', 'seria data', 'qual seria data', 'qual seria', 'seria benefício',
    'algumas cotações', 'realizar algumas', 'nesse seria', 'ótimo vida', 'seria processo',
    'plataforma vida', 'motivo plataforma', 'atendimento', 'necessidade',
    
    # === VERBOS/AÇÕES DE CONVERSAÇÃO ===
    'falo', 'recebi', 'prefere', 'mande', 'ligue', 'vou', 'podendo', 'falar', 'pode', 'oizinho',
    'dar', 'ligar', 'apresentar', 'faça', 'posso', 'chamar', 'escutando', 'procede', 'tens',
    'passar', 'lembra', 'ligo', 'achou', 'participar', 'aguarde', 'encontra', 'cortando',
    'cobre', 'verificou', 'iniciar', 'solicitado', 'mostrar', 'caiu', 'considerar', 'assinar',
    'renovar', 'colher', 'trazer', 'busca', 'dirigindo', 'confirma', 'carrega',
    'ajudando', 'tratando', 'resolver', 'enviar', 'ligando', 'peço',
    
    # === ADVÉRBIOS/CONECTORES CONVERSACIONAIS ===
    'agora', 'exemplo', 'daqui', 'minutos', 'época', 'segundo', 'ainda', 'pé', 'mesmo', 'normalmente',
    'algum', 'alguns', 'quantas', 'melhor', 'disponível', 'prioridade', 'rápidas', 'primeira vez',
    'vez', 'algumas informações', 'informações seria', 'aproximadamente', 'informar', 'realizar',
    'urgência contratar', 'urgência urgência', 'preço', 'cotando', 'seguimento',
    'rápido', 'quantos', 'interesse', 'continuar', 'interesse continuar', 'deseja', 'deseja continuar',
    'futuro', 'respeito', 'entrando respeito', 'caminhar', 'coletar', 'errado', 'veio',
    'ninguém', 'creio', 'celular', 'fechei', 'deixa ver', 'novo', 'olhada', 'fim',
    'ajuste', 'bela', 'dou', 'acha', 'mandei', 'passei', 'ruim', 'débito', 'boleto', 'fevereiro',
    'data', 'pegar',
    
    # === SUBSTANTIVOS GENÉRICOS ===
    'whats', 'telefone', 'whatsapp', 'contato', 'encaminhar', 'barra', 'referências', 'passadas',
    'verdade', 'queria', 'hora', 'meia', 'meia hora', 'tempinho', 'almoço', 'especialista', 'estagiário',
    'motoristas', 'motorista', 'cargas', 'juiz', 'obra obra', 'vidas', 'grupo', 'requisitos',
    'companhia', 'senhor', 'algum prazo', 'diz coisa', 'reforma', 'ativa', 'viagens', 'registro',
    'médio', 'avulso', 'milhões', 'milhão', 'desejo', 'sanada', 'referência', 'ouvindo ouvindo',
    'ouvindo', 'sou', 'bola', 'ntt', 'antt', 'horário', 'melhor horário', 'pedido', 'ordens',
    'colaboradores', 'prazo', 'dono', 'próprias empresa', 'terceiros próprias', 'quem quem',
    'viu nada', 'rua', 'cobrança', 'olhadinha', 'data nascimento', 'juros', 'motos',
    'outras cotações', 'algumas cotações', 'mês vem', 'condições', 'todos meses', 'tinha vida',
    'rio janeiro', 'material', 'precisa', 'estou', 'possui', 'faz', 'seriam', 'compreendi',
    'tinha soliticado', 'mesmo precisa', 'dispensa', 'licitação', 'relação', 'entender pouquinho',
    'outros', 'civil', 'ótima semana', 'semana', 'horas', 'respeito vida',
    'belo horizonte', 'horizonte', 'enviar mail', 'andar muito',
    'apresentação', 'preciso entender', 'solicitar', 'são pessoas', 'conversou',
}

# Stopwords completas (união de todas)
STOPWORDS_COMPLETAS = PT_STOPWORDS | STOPWORDS_CONVERSACAO

# Stop words para tópicos (usa TODAS as stopwords - alinhado com V2)
TOPICS_STOPWORDS = STOPWORDS_COMPLETAS  # Alinhado com visualize_bertopic_v2.py

