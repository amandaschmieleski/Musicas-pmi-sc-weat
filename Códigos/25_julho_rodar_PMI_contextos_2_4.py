# ============================================================
# SCRIPT STANDALONE OTIMIZADO - PMI PARA TODOS OS GÊNEROS E JANELAS
#
# Este arquivo incorpora o conteúdo de COMPLETO_146k_pmi_alfabetico.py
# e o executor em lote. Não precisa importar outro arquivo .py do projeto.
#
# Removido a pedido:
# - *_comparacao_contextos_com_categoria_por_100.csv
# - *_grafico_01_contextos_com_categoria_por_100.png
# - *_diagnostico_colisoes_alvos.csv
# - *_auditoria_palavras.csv
# - *_grafico_04_top_aparencia_por_genero.png
# - *_auditoria_contextos_aparencia.csv
# - *_contextos.csv
# - *_comparacao_pmi_medio_palavras.csv
# - *_comparacao_pmi_ponderado_por_ocorrencia.csv
# - colunas contextos_com_categoria_por_100 e ocorrencias_categoria_por_100_contextos
#   no arquivo *_categorias.csv e nos consolidados.
#
# Otimização desta versão:
# - tokeniza cada corpus uma única vez;
# - reaproveita os tokens para janela 4 e janela 2;
# - evita processar o mesmo CSV inteiro duas vezes com spaCy.
# ============================================================

import math
import re
import unicodedata
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURAÇÕES
# ============================================================

CAMINHO_CSV = "merged_df.txt"
COLUNA_TEXTO = "Letra da Música"
COLUNA_MUSICA = "Nome da Música"
COLUNA_ARTISTA = "Artista"

USAR_PRIMEIRAS_N_MUSICAS = None
JANELA_CONTEXTO = 4
REMOVER_ALVO_DO_CONTEXTO = True

# NÃO deduplicar contextos: cada ocorrência de alvo gera um contexto.
DEDUPLICAR_CONTEXTOS = False

EPSILON = 0.5
MIN_OCORRENCIA_ATRIBUTO = 1

MODELO_SPACY = "pt_core_news_sm"
PREFIXO_SAIDA = "COMPLETO_146k_pmi_14_junho"

# ============================================================
# LISTAS
# ============================================================

ALVOS = {
    'Feminino': [
        'alcione',
        'alice',
        'aline',
        'avó',  # traduzido Chen
        'avós',
        'avozinha',
        'babi',
        'baby',
        'baiana',
        'beth',
        'brasileira',
        'carolina',
        'claudia',
        'dama',  # traduzido Chen
        'damas',
        'daminha',
        'dela',  # traduzido Chen
        'delas',
        'dolores',
        'dona',
        'donas',
        'dora',
        'ela',  # traduzido Chen
        'elas',
        'elka',
        'erika',
        'esposa',  # traduzido Chen
        'esposas',
        'esposinha',
        'estrangeira',
        'fabiana',
        'fêmea',  # traduzido Chen
        'fêmeas',
        'filha',  # traduzido Chen
        'filhas',
        'filhinha',
        'filhona',
        'gabriela',
        'garota',  # traduzido Chen
        'garotas',
        'garotinha',
        'garotona',
        'inara',
        'iolanda',
        'irene',
        'irmã',  # traduzido Chen
        'irmãs',
        'irmãzinha',
        'jéssica',
        'julieta',
        'leandra',
        'leonor',
        'ludmilla',
        'luzia',
        'madame',  # traduzido Chen
        'madames',
        'mãe',  # traduzido Chen
        'mães',
        'mãezinha',
        'mainha',
        'mamãe',  # traduzido Chen
        'mamães',
        'mamãezinha',
        'mana',  # traduzido Chen
        'manas',
        'maninha',
        'marcelly',
        'maria',
        'marilia',
        'menina',  # traduzido Chen
        'meninas',
        'menininha',
        'mina',
        'minas',
        'minazinha',
        'moça',  # traduzido Chen
        'moças',
        'mocinha',
        'mulata',
        'mulatinha',
        'mulher',  # traduzido Chen
        'mulherada',
        'mulheres',
        'mulherona',
        'mulherzinha',
        'namorada',  # traduzido Chen
        'namoradas',
        'namoradinha',
        'nega',
        'negona',
        'neguinha',
        'nora',  # traduzido Chen
        'noras',
        'norinha',
        'novinha',
        'pagodeira',
        'pagodeiras',
        'rainha',  # traduzido Chen
        'rainhas',
        'rita',
        'rosa',
        'rosalina',
        'sabrina',
        'sara',
        'senhora',  # traduzido Chen
        'senhoras',
        'senhorinha',
        'senhorita',  # traduzido Chen
        'senhoritas',
        'sobrinha',  # traduzido Chen
        'sobrinhas',
        'solteira',
        'solteiras',
        'solteirinha',
        'solteirona',
        'tia',  # traduzido Chen
        'tias',
        'tiazinha',
        'titia',  # traduzido Chen
        'titias',
        'vizinha',
        'vizinhas',
        'vovó',  # traduzido Chen
        'vovós',
        'vovozinha',
    ],
    'Masculino': [
        'abel',
        'adão',
        'alexandre',
        'almir',
        'anderson',
        'anthony',
        'arlindo',
        'augusto',
        'avô',  # traduzido Chen
        'avôs',
        'avozinho',
        'bebeto',
        'beto',
        'bill',
        'bob',
        'bruninho',
        'bruno',
        'caetano',
        'cantor',
        'cantores',
        'cantorzinho',
        'cara',  # traduzido Chen
        'carão',
        'caras',
        'carinha',
        'carlos',
        'charlie',
        'chico',
        'cláudio',
        'companheirinho',
        'companheiro',
        'companheiros',
        'cosme',
        'cumpade',
        'daniel',
        'dele',  # traduzido Chen
        'deles',
        'dilsinho',
        'dono',
        'donos',
        'doutor',
        'doutores',
        'doutorzinho',
        'ele',  # traduzido Chen
        'eles',
        'emanuel',
        'filhão',
        'filhinho',
        'filho',  # traduzido Chen
        'filhos',
        'gabriel',
        'garotão',
        'garotinho',
        'garoto',  # traduzido Chen
        'garotos',
        'genro',  # traduzido Chen
        'genros',
        'guilherme',
        'gustavo',
        'homão',
        'homem',  # traduzido Chen
        'homens',
        'homenzinho',
        'ícaro',
        'irmão',  # traduzido Chen
        'irmãos',
        'irmãozão',
        'irmãozinho',
        'iuri',
        'james',
        'jhonny',
        'joão',
        'jorge',
        'josé',
        'junior',
        'leandro',
        'léo',
        'leonardo',
        'luan',
        'luiz',
        'machão',
        'machinho',
        'macho',  # traduzido Chen
        'machos',
        'manão',
        'mané',
        'manés',
        'manezinho',
        'maninho',
        'mano',
        'manoel',
        'manos',
        'marcelo',
        'marcinho',
        'maridão',
        'maridinho',
        'marido',  # traduzido Chen
        'maridos',
        'mario',
        'marquinho',
        'marquinhos',
        'meninão',
        'menininho',
        'menino',  # traduzido Chen
        'meninos',
        'mestrão',
        'mestre',
        'mestres',
        'miguel',
        'mino',
        'mocinho',
        'moço',
        'moços',
        'molecão',
        'moleque',
        'moleques',
        'molequinho',
        'mulato',
        'namoradinho',
        'namorado',  # traduzido Chen
        'namorados',
        'nathan',
        'negão',
        'nego',
        'neguinho',
        'netinho',
        'novo',
        'pablo',
        'pagodeiro',
        'pagodeiros',
        'pai',  # traduzido Chen
        'painho',
        'pais',
        'paizão',
        'paizinho',
        'papai',  # traduzido Chen
        'papais',
        'papaizão',
        'papaizinho',
        'patrão',
        'patrãozinho',
        'patrões',
        'pedro',
        'poeta',
        'poetas',
        'poetinha',
        'primão',
        'priminho',
        'primo',
        'primos',
        'rafael',
        'rapagão',
        'rapaz',  # traduzido Chen
        'rapazes',
        'rapaziada',
        'rapazinho',
        'rei',  # traduzido Chen
        'reis',
        'reizinho',
        'ricardo',
        'rodolfinho',
        'rodriguinho',
        'romeu',
        'romeus',
        'rubem',
        'safadão',
        'safadinho',
        'safado',
        'safados',
        'senhor',  # traduzido Chen
        'senhores',
        'senhorzinho',
        'sergio',
        'sobrinho',  # traduzido Chen
        'sobrinhos',
        'sogrão',
        'sogro',  # traduzido Chen
        'sogros',
        'solteirão',
        'solteirinho',
        'solteiro',
        'solteiros',
        'suel',
        'thiago',
        'thiaguinho',
        'tião',
        'tio',  # traduzido Chen
        'tios',
        'tiozão',
        'tiozinho',
        'titio',
        'vavá',
        'vinícius',
        'vitinho',
        'vizinho',
        'vizinhos',
        'vovô',  # traduzido Chen
        'vovozinho',
        'zé',
        'zeca',
    ],
}

# ALVOS e ATRIBUTOS iguais ao XLSX "Atributos_e_alvos_listas_Chen_Amanda_25_julho_ordenado_corrigido.xlsx".
ATRIBUTOS = {
    'Agradável': [
        'abraçar',
        'abraço',
        'adorar',
        'alegre',
        'alegria',  # traduzido Chen
        'amado',
        'amar',  # traduzido Chen
        'amigo',  # traduzido Chen
        'amor',  # traduzido Chen
        'animação',  # traduzido Chen
        'anjo',
        'apaixonar',
        'apegado',
        'arco-íris',  # traduzido Chen
        'bacana',
        'beijar',
        'beijo',
        'bem',
        'bom',
        'brilho',
        'brincalhão',
        'carinho',  # traduzido Chen
        'carinhoso',
        'carisma',
        'certo',
        'céu',  # traduzido Chen
        'cheirosa',
        'companheira',
        'confiar',
        'contente',
        'coração',
        'cuidar',
        'curar',
        'curtir',
        'decente',
        'desejo',
        'deusa',
        'diamante',  # traduzido Chen
        'diploma',  # traduzido Chen
        'disposição',
        'doce',
        'elogiar',
        'empolgada',
        'escolhida',
        'especial',
        'espetacular',
        'estrela',
        'faceiro',
        'família',  # traduzido Chen
        'famosa',
        'felicidade',
        'feliz',  # traduzido Chen
        'férias',  # traduzido Chen
        'fiel',  # traduzido Chen
        'flor',
        'gentil',  # traduzido Chen
        'honesto',  # traduzido Chen
        'honra',  # traduzido Chen
        'incrível',
        'inocente',
        'joia',
        'lealdade',
        'legal',
        'liberdade',  # traduzido Chen
        'ligado',
        'luz',
        'maneira',
        'manso',
        'maravilha',
        'maravilhoso',  # traduzido Chen
        'medicinal',
        'milagre',  # traduzido Chen
        'moderna',
        'nascer',
        'nascer do sol',  # traduzido Chen
        'nobre',
        'nota 10',
        'ouro',
        'paixão',
        'paraíso',  # traduzido Chen
        'parceiro',
        'paz',  # traduzido Chen
        'perfeição',
        'perfeito',
        'prazer',  # traduzido Chen
        'presente',  # traduzido Chen
        'puro',
        'radiante',
        'remédio',
        'respeito',
        'responsa',
        'rico',
        'riqueza',
        'rir',
        'riso',  # traduzido Chen
        'romance',
        'saudade',
        'saúde',  # traduzido Chen
        'sensacional',
        'simpatia',
        'simpático',
        'sincero',
        'sonhar',
        'sonho',
        'sorrir',
        'sorriso',
        'sorte',  # traduzido Chen
        'ternura',
        'união',
        'verdadeiro',
    ],
    'Aparência': [
        'alto',
        'atlético',  # traduzido Chen
        'atraente',  # traduzido Chen
        'baixo',
        'barriga',
        'boca',
        'bombado',  # traduzido Chen
        'bonito',  # traduzido Chen
        'branco',
        'bumbum',
        'buzanfão',
        'cabelo',
        'calor',
        'careca',  # traduzido Chen
        'cheia',  # traduzido Chen
        'cheiro',
        'cigana',
        'cigano',
        'cinderela',
        'comprida',
        'corado',  # traduzido Chen
        'corpo',
        'donzela',
        'elegante',
        'estilosa',  # traduzido Chen
        'excitar',
        'feio',  # traduzido Chen
        'formosa',
        'forte',  # traduzido Chen
        'fraco',  # traduzido Chen
        'franzino',  # traduzido Chen
        'gata',
        'gatinha',
        'gato',
        'gordo',  # traduzido Chen
        'gostoso',
        'lindo',  # traduzido Chen
        'loiro',
        'magra',  # traduzido Chen
        'molhada',
        'morena',
        'musa',
        'negro',
        'nua',
        'olhar',
        'olhos',
        'peituda',
        'pelada',
        'pele',
        'perfume',
        'popotão',
        'pretinho',
        'preto',
        'princesa',
        'provocar',
        'quente',
        'rebolar',
        'roupa',
        'saliente',
        'saudável',  # traduzido Chen
        'sedução',
        'sedutor',  # traduzido Chen
        'seduzir',
        'sensual',  # traduzido Chen
        'sereia',
        'sexy',
        'simples',
        'tanajura',
        'tentação',
        'tesão',
        'turbinada',
    ],
    'Desagradável': [
        'abandonar',
        'abusar',  # traduzido Chen
        'acidente',  # traduzido Chen
        'adversidade',
        'agonia',  # traduzido Chen
        'agressão',  # traduzido Chen
        'amargo',
        'assassinato',  # traduzido Chen
        'bagaceira',
        'bagunça',
        'bandido',
        'barraqueira',
        'barreira',
        'bêbado',
        'bipolar',
        'bomba',  # traduzido Chen
        'brigar',
        'burro',
        'cafajeste',
        'câncer',  # traduzido Chen
        'castigo',
        'chato',
        'chifrudo',
        'chumbinho',
        'ciumenta',
        'colisão',  # traduzido Chen
        'congelar',
        'conspirador',
        'crime',
        'criminoso',
        'cruel',
        'delator',
        'desastre',  # traduzido Chen
        'desconfiar',
        'desobediente',
        'difícil',
        'divórcio',  # traduzido Chen
        'doença',  # traduzido Chen
        'doer',
        'doido',
        'dor',
        'droga',
        'enganar',
        'errado',
        'exagerada',
        'fácil',
        'falso',
        'fanático',
        'fatal',
        'fedor',  # traduzido Chen
        'feio',  # traduzido Chen
        'ferir',
        'fracasso',  # traduzido Chen
        'frio',
        'fuleiro',
        'fúria',
        'gelado',
        'granada',
        'guerra',  # traduzido Chen
        'horrível',  # traduzido Chen
        'implicar',
        'indigesta',
        'infiel',
        'ingrata',
        'inimigo',
        'invejar',
        'irresponsável',
        'ladrão',
        'libidinosa',
        'ligeira',
        'lobo',
        'luto',  # traduzido Chen
        'machucar',
        'magoar',
        'mal',  # traduzido Chen
        'malandro',
        'maldosa',
        'maloqueiro',
        'maltratar',
        'maluco',
        'malvada',
        'mandona',
        'manhosa',
        'maroto',
        'matar',  # traduzido Chen
        'mau',  # traduzido Chen
        'mentir',
        'mentira',
        'morrer',
        'morte',  # traduzido Chen
        'nojento',  # traduzido Chen
        'ódio',  # traduzido Chen
        'ofender',
        'patricinha',
        'pavor',
        'perversa',
        'péssimo',  # traduzido Chen
        'piranha',
        'piriguete',
        'pobre',
        'pobreza',  # traduzido Chen
        'podre',  # traduzido Chen
        'poluir',  # traduzido Chen
        'prisão',  # traduzido Chen
        'problema',
        'puta',
        'rancor',
        'revoltado',
        'ruim',
        'safada',
        'sapeca',
        'sofrer',
        'sofrimento',
        'solidão',
        'soltinha',
        'sombrio',
        'sozinho',
        'sujeira',  # traduzido Chen
        'terrível',  # traduzido Chen
        'tormento',
        'tragédia',  # traduzido Chen
        'trair',
        'triste',
        'tristeza',
        'vacilona',
        'vagabundo',
        'veneno',  # traduzido Chen
        'vômito',  # traduzido Chen
        'vulgar',
        'xingar',
        'zangado',
    ],
    'Força': [
        'afirmar',  # traduzido Chen
        'alto',  # traduzido Chen
        'atitude',
        'campeão',
        'comandar',  # traduzido Chen
        'competidor',
        'confiança',
        'confiante',  # traduzido Chen
        'consciente',
        'controle',
        'coragem',
        'dominar',  # traduzido Chen
        'força',
        'forte',  # traduzido Chen
        'grito',  # traduzido Chen
        'guerreiro',
        'independente',
        'justo',
        'liderança',  # traduzido Chen
        'lutar',
        'maduro',
        'ousadia',
        'ousado',  # traduzido Chen
        'potência',  # traduzido Chen
        'proteger',
        'protetor',
        'reprodutor',
        'seguro',
        'valente',
        'vencedor',  # traduzido Chen
        'vencer',  # traduzido Chen
        'vitória',  # traduzido Chen
    ],
    'Fraqueza': [
        'ansioso',
        'cabisbaixo',
        'carente',
        'ceder',  # traduzido Chen
        'chorar',
        'covarde',
        'defeito',
        'delicado',  # traduzido Chen
        'dependente',
        'deprimido',
        'derrota',
        'dificuldade',
        'doente',
        'errar',
        'fracasso',  # traduzido Chen
        'fraco',  # traduzido Chen
        'frágil',  # traduzido Chen
        'fraqueza',  # traduzido Chen
        'fudido',
        'inseguro',
        'medo',  # traduzido Chen
        'nervoso',
        'otário',
        'perdedor',  # traduzido Chen
        'perder',  # traduzido Chen
        'quieto',
        'recuar',  # traduzido Chen
        'seguir',  # traduzido Chen
        'timidez',  # traduzido Chen
        'tolo',
        'tonto',
        'vulnerável',  # traduzido Chen
    ],
    'Inteligência': [
        'adaptar',  # traduzido Chen
        'analítico',  # traduzido Chen
        'aprender',
        'apto',  # traduzido Chen
        'atualizado',
        'brilhante',  # traduzido Chen
        'compreender',
        'conhecer',
        'curioso',  # traduzido Chen
        'educado',
        'engenhoso',  # traduzido Chen
        'engraçado',
        'ensinar',
        'entender',
        'esperto',  # traduzido Chen
        'estudar',
        'estudioso',
        'gênio',  # traduzido Chen
        'habilidoso',  # traduzido Chen
        'imaginar',  # traduzido Chen
        'inteligente',  # traduzido Chen
        'intuitivo',  # traduzido Chen
        'inventar',  # traduzido Chen
        'investigativo',  # traduzido Chen
        'lógico',  # traduzido Chen
        'pensar',
        'perceber',
        'ponderado',  # traduzido Chen
        'potente',
        'precoce',  # traduzido Chen
        'refletir',  # traduzido Chen
        'sábio',  # traduzido Chen
        'sagaz',  # traduzido Chen
        'sensato',  # traduzido Chen
        'venerável',  # traduzido Chen
    ],
}

ORDEM_CATEGORIAS = ['Agradável', 'Desagradável', 'Aparência', 'Inteligência', 'Força', 'Fraqueza']

# Palavras em que o lema automático do spaCy costuma criar ruído semântico.
# Ex.: nega -> negar; mulherão -> mulher; linda -> lindar; lindo -> lir.
ATRIBUTOS_SEM_LEMA_AUTOMATICO = {
  #  "nega", "negão", "nego", "negona", "negra", "negro", "neguinha", "neguinho",
   # "preta", "pretinha", "pretinho", "preto",
  #  "mulata", "mulatinha", "mulato",
    #"linda", "lindo", "loira", "loiro", "morena",
    "gata", "gatinha", "gato", "musa", "sereia", "cigana", "cigano"
}

_NLP = None

# ============================================================
# NORMALIZAÇÃO
# ============================================================

def carregar_spacy():
    global _NLP
    if _NLP is not None:
        return _NLP
    try:
        import spacy
    except ImportError as exc:
        raise RuntimeError(
            "spaCy não está instalado. Instale com:\n"
            "  pip install spacy\n"
            "  python -m spacy download pt_core_news_sm"
        ) from exc
    try:
        _NLP = spacy.load(MODELO_SPACY, disable=["parser", "ner"])
    except OSError as exc:
        raise RuntimeError(
            f"Modelo spaCy '{MODELO_SPACY}' não encontrado. Instale com:\n"
            f"  python -m spacy download {MODELO_SPACY}"
        ) from exc
    return _NLP


def limpar_texto_para_spacy(texto: str) -> str:
    texto = str(texto).lower()
    texto = re.sub(r"[^a-zA-ZÀ-ÖØ-öø-ÿ0-9\s]", " ", texto)
    texto = re.sub(r"\d+", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


def remover_acentos(texto: str) -> str:
    texto = unicodedata.normalize("NFKD", str(texto).lower())
    return "".join(c for c in texto if not unicodedata.combining(c))


def normalizar_alvo(texto: str) -> str:
    """
    Normalização dos ALVOS.
    Preserva acentos para não misturar pares como avó/avô e vovó/vovô.
    Não usa lema.
    """
    texto = str(texto).lower().strip()
    texto = re.sub(r"[^a-zà-öø-ÿ0-9]", "", texto)
    return texto


def normalizar_atributo(texto: str) -> str:
    """
    Normalização dos ATRIBUTOS.
    Remove acentos porque, para atributos, queremos aproximar saúde/saude,
    paixão/paixao etc. A distinção avó/avô não se aplica aqui.
    """
    texto = remover_acentos(texto)
    texto = re.sub(r"[^a-z0-9]", "", texto)
    return texto.strip()


def tokenizar_com_spacy(texto: str) -> list[dict]:
    """
    Retorna tokens com duas normalizações:
    - alvo_norm: para ALVOS, sem lematização e com acentos preservados;
    - attr_raw_norm/attr_lemma_norm: para ATRIBUTOS.
    """
    texto_limpo = limpar_texto_para_spacy(texto)
    if not texto_limpo:
        return []
    nlp = carregar_spacy()
    doc = nlp(texto_limpo)
    saida = []
    for tok in doc:
        if tok.is_space or tok.is_punct:
            continue
        raw = tok.text.lower()
        lema = tok.lemma_.lower() if tok.lemma_ else raw
        alvo_norm = normalizar_alvo(raw)
        attr_raw_norm = normalizar_atributo(raw)
        attr_lemma_norm = normalizar_atributo(lema)
        if alvo_norm or attr_raw_norm or attr_lemma_norm:
            saida.append({
                "raw": raw,
                "lemma": lema,
                "alvo_norm": alvo_norm,
                "attr_raw_norm": attr_raw_norm,
                "attr_lemma_norm": attr_lemma_norm,
            })
    return saida


def lematizar_palavra_atributo(palavra: str) -> str:
    pares = tokenizar_com_spacy(palavra)
    if not pares:
        return normalizar_atributo(palavra)
    return pares[0]["attr_lemma_norm"] or pares[0]["attr_raw_norm"]

# ============================================================
# CACHE DAS LISTAS PREPARADAS
# ============================================================

_ALVOS_NORM_CACHE = None
_ATRIBUTOS_NORM_CACHE = None

# ============================================================
# PREPARAÇÃO DAS LISTAS
# ============================================================

def preparar_alvos() -> dict[str, set[str]]:
    """
    ALVOS SEM LEMATIZAÇÃO.
    A lista já contém singular, plural, diminutivo e aumentativo.
    """
    resultado = {}
    for genero, palavras in ALVOS.items():
        formas = set()
        for palavra in palavras:
            forma = normalizar_alvo(palavra)
            if forma:
                formas.add(forma)
        resultado[genero] = formas
    return resultado


def preparar_atributos() -> dict[str, dict[str, set[str]]]:
    """
    ATRIBUTOS com lematização segura.
    Atributos problemáticos ficam com correspondência por forma original normalizada.
    """
    resultado = {}
    for categoria, palavras in ATRIBUTOS.items():
        resultado[categoria] = {}
        for palavra in palavras:
            canonica = normalizar_atributo(palavra)
            variantes = {canonica}
            if canonica not in ATRIBUTOS_SEM_LEMA_AUTOMATICO:
                lema = lematizar_palavra_atributo(palavra)
                if lema:
                    variantes.add(lema)
            resultado[categoria][canonica] = variantes
    return resultado



def obter_listas_preparadas() -> tuple[dict[str, set[str]], dict[str, dict[str, set[str]]]]:
    """
    Prepara alvos e atributos uma única vez por execução do script.
    Isso evita repetir a lematização das listas a cada CSV/janela.
    """
    global _ALVOS_NORM_CACHE, _ATRIBUTOS_NORM_CACHE

    if _ALVOS_NORM_CACHE is None:
        _ALVOS_NORM_CACHE = preparar_alvos()

    if _ATRIBUTOS_NORM_CACHE is None:
        _ATRIBUTOS_NORM_CACHE = preparar_atributos()

    return _ALVOS_NORM_CACHE, _ATRIBUTOS_NORM_CACHE

# ============================================================
# EXTRAÇÃO DE CONTEXTOS
# ============================================================

def extrair_contextos(df: pd.DataFrame, alvos_norm: dict[str, set[str]]) -> list[dict]:
    contextos = []
    for idx_linha, row in df.iterrows():
        tokens = tokenizar_com_spacy(row.get(COLUNA_TEXTO, ""))
        if not tokens:
            continue

        musica = row.get(COLUNA_MUSICA, "")
        artista = row.get(COLUNA_ARTISTA, "")

        for i, token in enumerate(tokens):
            token_alvo = token["alvo_norm"]

            for genero, alvos_genero in alvos_norm.items():
                if token_alvo not in alvos_genero:
                    continue

                ini = max(0, i - JANELA_CONTEXTO)
                fim = min(len(tokens), i + JANELA_CONTEXTO + 1)

                if REMOVER_ALVO_DO_CONTEXTO:
                    tokens_contexto = tokens[ini:i] + tokens[i + 1:fim]
                else:
                    tokens_contexto = tokens[ini:fim]

                if not tokens_contexto:
                    continue

                contextos.append({
                    "idx_linha_csv": idx_linha,
                    "genero": genero,
                    "alvo": token["raw"],
                    "alvo_norm": token_alvo,
                    "alvo_lema_apenas_auditoria": token["lemma"],
                    "contexto": " ".join(t["raw"] for t in tokens_contexto),
                    "contexto_lematizado": " ".join(t["lemma"] for t in tokens_contexto),
                    "contexto_attr_raw": [t["attr_raw_norm"] for t in tokens_contexto if t["attr_raw_norm"]],
                    "contexto_attr_lemas": [t["attr_lemma_norm"] for t in tokens_contexto if t["attr_lemma_norm"]],
                    "musica": musica,
                    "artista": artista,
                })

    if not DEDUPLICAR_CONTEXTOS:
        return contextos

    # Mantido só por segurança; por configuração, não é usado.
    vistos = set()
    unicos = []
    for c in contextos:
        chave = (c["genero"], c["alvo_norm"], c["contexto"], c["musica"], c["artista"])
        if chave in vistos:
            continue
        vistos.add(chave)
        unicos.append(c)
    return unicos

# ============================================================
# CONTAGEM DE ATRIBUTOS
# ============================================================

def contar_variantes_no_contexto(contexto: dict, variantes: set[str]) -> int:
    """
    Conta ocorrências no contexto SEM deduplicar contextos e SEM transformar
    a janela em conjunto.

    Correção importante:
    - cada token da janela conta no máximo 1 vez para a mesma palavra canônica;
    - isso evita duplicação técnica quando token bruto e lema são iguais
      (ex.: gostoso/gostoso, fogo/fogo);
    - repetições reais em tokens diferentes continuam contando.
    """
    total = 0
    raws = contexto.get("contexto_attr_raw", [])
    lemas = contexto.get("contexto_attr_lemas", [])

    for raw, lema in zip(raws, lemas):
        if raw in variantes or lema in variantes:
            total += 1

    return total


def contexto_tem_variantes(contexto: dict, variantes: set[str]) -> bool:
    return contar_variantes_no_contexto(contexto, variantes) > 0


def pmi_suavizada(n_total: int, n_genero: int, n_attr_total: int, n_attr_genero: int) -> float:
    p_y_dado_x = (n_attr_genero + EPSILON) / (n_genero + 2 * EPSILON)
    p_y = (n_attr_total + EPSILON) / (n_total + 2 * EPSILON)
    return math.log2(p_y_dado_x / p_y)


def media_ponderada(valores: pd.Series, pesos: pd.Series) -> float:
    tmp = pd.DataFrame({"valor": valores, "peso": pesos}).dropna()
    if tmp.empty or tmp["peso"].sum() == 0:
        return np.nan
    return float((tmp["valor"] * tmp["peso"]).sum() / tmp["peso"].sum())

# ============================================================
# GRÁFICOS
# ============================================================

def grafico_barras(df: pd.DataFrame, coluna: str, nome: str, titulo: str, ylabel: str):
    pivot = df.pivot(index="categoria", columns="genero", values=coluna).reindex(ORDEM_CATEGORIAS)
    x = np.arange(len(ORDEM_CATEGORIAS))
    largura = 0.35
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - largura / 2, pivot.get("Masculino", pd.Series(index=ORDEM_CATEGORIAS, dtype=float)), largura, label="Masculino")
    ax.bar(x + largura / 2, pivot.get("Feminino", pd.Series(index=ORDEM_CATEGORIAS, dtype=float)), largura, label="Feminino")
    if "pmi" in coluna:
        ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_title(titulo)
    ax.set_xlabel("Categoria")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(ORDEM_CATEGORIAS, rotation=45, ha="right")
    ax.legend(title="Gênero")
    plt.tight_layout()
    plt.savefig(nome, dpi=200)
    plt.close()



# ============================================================
# TOKENIZAÇÃO REAPROVEITÁVEL E CÁLCULO DE PMI
# ============================================================

NLP_PIPE_BATCH_SIZE = 500


def tokens_from_doc(doc) -> list[dict]:
    """
    Converte um Doc do spaCy na mesma estrutura usada pelo cálculo de PMI.

    Importante:
    - alvo_norm preserva acentos e NÃO usa lema;
    - attr_raw_norm e attr_lemma_norm são usados para atributos.
    """
    saida = []

    for tok in doc:
        if tok.is_space or tok.is_punct:
            continue

        raw = tok.text.lower()
        lema = tok.lemma_.lower() if tok.lemma_ else raw

        alvo_norm = normalizar_alvo(raw)
        attr_raw_norm = normalizar_atributo(raw)
        attr_lemma_norm = normalizar_atributo(lema)

        if alvo_norm or attr_raw_norm or attr_lemma_norm:
            saida.append({
                "raw": raw,
                "lemma": lema,
                "alvo_norm": alvo_norm,
                "attr_raw_norm": attr_raw_norm,
                "attr_lemma_norm": attr_lemma_norm,
            })

    return saida


def carregar_dataframe_csv(caminho_csv: Path) -> pd.DataFrame:
    """Lê o CSV uma única vez para o corpus atual."""
    encoding = detectar_encoding(caminho_csv)
    print(f"Encoding usado: {encoding}")

    df = pd.read_csv(
        caminho_csv,
        encoding=encoding,
        dtype=str,
        low_memory=False
    )

    if COLUNA_TEXTO not in df.columns:
        raise ValueError(
            f"Coluna '{COLUNA_TEXTO}' não encontrada. Colunas: {list(df.columns)}"
        )

    if USAR_PRIMEIRAS_N_MUSICAS is not None:
        df = df.head(USAR_PRIMEIRAS_N_MUSICAS).copy()

    return df


def tokenizar_dataframe_uma_vez(df: pd.DataFrame) -> list[dict]:
    """
    Tokeniza todas as letras do corpus uma única vez.

    Essa é a principal otimização desta versão: os tokens gerados aqui são
    reaproveitados para janela 4 e janela 2. Assim, o spaCy não precisa
    processar o mesmo corpus duas vezes.
    """
    nlp = carregar_spacy()

    textos_limpos = []
    metadados = []

    for idx_linha, row in df.iterrows():
        texto_limpo = limpar_texto_para_spacy(row.get(COLUNA_TEXTO, ""))

        textos_limpos.append(texto_limpo)
        metadados.append({
            "idx_linha_csv": idx_linha,
            "musica": row.get(COLUNA_MUSICA, ""),
            "artista": row.get(COLUNA_ARTISTA, ""),
        })

    registros_tokenizados = []

    print("Tokenizando corpus uma única vez com spaCy...")

    for meta, doc in zip(
        metadados,
        nlp.pipe(textos_limpos, batch_size=NLP_PIPE_BATCH_SIZE)
    ):
        tokens = tokens_from_doc(doc)

        if not tokens:
            continue

        registros_tokenizados.append({
            "idx_linha_csv": meta["idx_linha_csv"],
            "musica": meta["musica"],
            "artista": meta["artista"],
            "tokens": tokens,
        })

    print(f"Registros com tokens: {len(registros_tokenizados)} de {len(df)}")

    return registros_tokenizados


def extrair_contextos_de_tokens(
    registros_tokenizados: list[dict],
    alvos_norm: dict[str, set[str]],
    janela_contexto: int
) -> list[dict]:
    """
    Extrai contextos a partir dos tokens já calculados.

    A diferença para a versão anterior é que esta função NÃO chama spaCy.
    Ela só muda a janela usada ao redor do alvo.
    """
    contextos = []

    for registro in registros_tokenizados:
        tokens = registro["tokens"]
        musica = registro["musica"]
        artista = registro["artista"]
        idx_linha = registro["idx_linha_csv"]

        for i, token in enumerate(tokens):
            token_alvo = token["alvo_norm"]

            for genero, alvos_genero in alvos_norm.items():
                if token_alvo not in alvos_genero:
                    continue

                ini = max(0, i - janela_contexto)
                fim = min(len(tokens), i + janela_contexto + 1)

                if REMOVER_ALVO_DO_CONTEXTO:
                    tokens_contexto = tokens[ini:i] + tokens[i + 1:fim]
                else:
                    tokens_contexto = tokens[ini:fim]

                if not tokens_contexto:
                    continue

                contextos.append({
                    "idx_linha_csv": idx_linha,
                    "genero": genero,
                    "alvo": token["raw"],
                    "alvo_norm": token_alvo,
                    "alvo_lema_apenas_auditoria": token["lemma"],
                    "contexto": " ".join(t["raw"] for t in tokens_contexto),
                    "contexto_lematizado": " ".join(t["lemma"] for t in tokens_contexto),
                    "contexto_attr_raw": [
                        t["attr_raw_norm"] for t in tokens_contexto if t["attr_raw_norm"]
                    ],
                    "contexto_attr_lemas": [
                        t["attr_lemma_norm"] for t in tokens_contexto if t["attr_lemma_norm"]
                    ],
                    "musica": musica,
                    "artista": artista,
                })

    if not DEDUPLICAR_CONTEXTOS:
        return contextos

    vistos = set()
    unicos = []

    for c in contextos:
        chave = (c["genero"], c["alvo_norm"], c["contexto"], c["musica"], c["artista"])
        if chave in vistos:
            continue
        vistos.add(chave)
        unicos.append(c)

    return unicos


def gerar_diagnostico_alvos(contextos: list[dict], prefixo_saida: str):
    """Gera o diagnóstico de alvos encontrados sem salvar *_contextos.csv."""
    contagem = {}

    for c in contextos:
        chave = (c["genero"], c["alvo_norm"], c["alvo"])
        contagem[chave] = contagem.get(chave, 0) + 1

    linhas = [
        {
            "genero": genero,
            "alvo_norm": alvo_norm,
            "alvo": alvo,
            "qtd_contextos": qtd,
        }
        for (genero, alvo_norm, alvo), qtd in contagem.items()
    ]

    diag_alvos = pd.DataFrame(linhas)

    if diag_alvos.empty:
        diag_alvos = pd.DataFrame(columns=["genero", "alvo_norm", "alvo", "qtd_contextos"])
    else:
        diag_alvos = diag_alvos.sort_values(
            ["genero", "qtd_contextos"],
            ascending=[True, False]
        )

    diag_alvos.to_csv(
        f"{prefixo_saida}_diagnostico_alvos_encontrados.csv",
        index=False,
        encoding="utf-8-sig"
    )


def calcular_pmi_e_gerar_outputs(
    contextos: list[dict],
    qtd_musicas: int,
    alvos_norm: dict[str, set[str]],
    atributos_norm: dict[str, dict[str, set[str]]],
    prefixo_saida: str,
    janela_contexto: int,
):
    """Calcula PMI e gera os outputs mantidos na versão enxuta."""
    if not contextos:
        raise ValueError(
            "Nenhum contexto foi encontrado. Confira a coluna de letras e as listas de alvos."
        )

    colisoes = sorted(alvos_norm.get("Feminino", set()) & alvos_norm.get("Masculino", set()))

    n_total_contextos = len(contextos)
    contextos_por_genero = {g: [c for c in contextos if c["genero"] == g] for g in ALVOS}
    n_genero = {g: len(lst) for g, lst in contextos_por_genero.items()}

    print(f"Músicas analisadas: {qtd_musicas}")
    print(f"Janela de contexto: {janela_contexto}")
    print(f"Contextos usados: {n_total_contextos}")
    print(f"Contextos por gênero: {n_genero}")
    print(f"Colisões entre alvos normalizados: {len(colisoes)}")

    if colisoes:
        print("ATENÇÃO: há alvos iguais nos dois gêneros após normalização:", colisoes)

    gerar_diagnostico_alvos(contextos, prefixo_saida)

    linhas_palavras = []

    for categoria, palavras in atributos_norm.items():
        print(f"Contando categoria: {categoria}")

        for palavra_canonica, variantes in palavras.items():
            cont_total = sum(contar_variantes_no_contexto(c, variantes) for c in contextos)
            ctx_total = sum(1 for c in contextos if contexto_tem_variantes(c, variantes))
            entra = cont_total >= MIN_OCORRENCIA_ATRIBUTO

            for genero, lista_ctx in contextos_por_genero.items():
                cont_genero = sum(contar_variantes_no_contexto(c, variantes) for c in lista_ctx)
                ctx_genero = sum(1 for c in lista_ctx if contexto_tem_variantes(c, variantes))

                pmi = (
                    pmi_suavizada(n_total_contextos, n_genero[genero], ctx_total, ctx_genero)
                    if entra
                    else np.nan
                )

                linhas_palavras.append({
                    "genero": genero,
                    "categoria": categoria,
                    "palavra_canonica": palavra_canonica,
                    "variantes_consideradas": "; ".join(sorted(variantes)),
                    "n_total_contextos": n_total_contextos,
                    "n_contextos_genero": n_genero[genero],
                    "contextos_com_atributo_baseline": ctx_total,
                    "contextos_com_atributo_genero": ctx_genero,
                    "ocorrencias_atributo_baseline": cont_total,
                    "ocorrencias_atributo_genero": cont_genero,
                    "p_contexto_atributo_baseline": (
                        ctx_total / n_total_contextos if n_total_contextos else np.nan
                    ),
                    "p_contexto_atributo_genero": (
                        ctx_genero / n_genero[genero] if n_genero[genero] else np.nan
                    ),
                    "ocorrencias_por_100_contextos_genero": (
                        cont_genero / n_genero[genero] * 100 if n_genero[genero] else np.nan
                    ),
                    "pmi_por_contexto_binario": pmi,
                    "usada_na_media": entra,
                    "motivo": "ok" if entra else "nao_ocorre_no_baseline",
                })

    df_palavras = pd.DataFrame(linhas_palavras)
    df_palavras.to_csv(
        f"{prefixo_saida}_palavras.csv",
        index=False,
        encoding="utf-8-sig"
    )

    linhas_cat = []

    for categoria in ORDEM_CATEGORIAS:
        for genero in ["Masculino", "Feminino"]:
            grupo = df_palavras[
                (df_palavras["categoria"] == categoria)
                & (df_palavras["genero"] == genero)
                & (df_palavras["usada_na_media"])
            ]

            lista_ctx = contextos_por_genero[genero]
            palavras_cat = atributos_norm[categoria]

            ocorrencias_categoria_genero = 0
            contextos_com_categoria_genero = 0

            for c in lista_ctx:
                total_c = sum(
                    contar_variantes_no_contexto(c, variantes)
                    for variantes in palavras_cat.values()
                )
                ocorrencias_categoria_genero += total_c
                if total_c > 0:
                    contextos_com_categoria_genero += 1

            ocorrencias_categoria_total = 0
            contextos_com_categoria_total = 0

            for c in contextos:
                total_c = sum(
                    contar_variantes_no_contexto(c, variantes)
                    for variantes in palavras_cat.values()
                )
                ocorrencias_categoria_total += total_c
                if total_c > 0:
                    contextos_com_categoria_total += 1

            linhas_cat.append({
                "genero": genero,
                "categoria": categoria,
                "n_contextos_genero": n_genero[genero],
                "contextos_com_categoria_genero": contextos_com_categoria_genero,
                "ocorrencias_categoria_genero": ocorrencias_categoria_genero,
                "pmi_categoria_contexto_binario": pmi_suavizada(
                    n_total_contextos,
                    n_genero[genero],
                    contextos_com_categoria_total,
                    contextos_com_categoria_genero,
                ),
                "pmi_medio_palavras": (
                    grupo["pmi_por_contexto_binario"].mean() if not grupo.empty else np.nan
                ),
                "pmi_ponderado_por_ocorrencia": (
                    media_ponderada(
                        grupo["pmi_por_contexto_binario"],
                        grupo["ocorrencias_atributo_genero"] + EPSILON,
                    )
                    if not grupo.empty
                    else np.nan
                ),
                "qtd_palavras_categoria": len(palavras_cat),
                "qtd_palavras_usadas": (
                    int(grupo["palavra_canonica"].nunique()) if not grupo.empty else 0
                ),
            })

    df_cat = pd.DataFrame(linhas_cat)
    df_cat["categoria"] = pd.Categorical(
        df_cat["categoria"],
        categories=ORDEM_CATEGORIAS,
        ordered=True
    )
    df_cat = df_cat.sort_values(["categoria", "genero"])
    df_cat.to_csv(
        f"{prefixo_saida}_categorias.csv",
        index=False,
        encoding="utf-8-sig"
    )

    grafico_barras(
        df_cat,
        "pmi_medio_palavras",
        f"{prefixo_saida}_grafico_02_pmi_medio_palavras.png",
        "PMI médio por atributo",
        "PMI médio",
    )

    grafico_barras(
        df_cat,
        "pmi_ponderado_por_ocorrencia",
        f"{prefixo_saida}_grafico_03_pmi_ponderado_por_ocorrencia.png",
        "PMI ponderado por frequência dos atributos",
        "PMI ponderado",
    )


    print("\nResumo por categoria:")
    print(df_cat.to_string(index=False))

    print("\nArquivos gerados:")
    for nome in [
        f"{prefixo_saida}_palavras.csv",
        f"{prefixo_saida}_categorias.csv",
        f"{prefixo_saida}_diagnostico_alvos_encontrados.csv",
        f"{prefixo_saida}_grafico_02_pmi_medio_palavras.png",
        f"{prefixo_saida}_grafico_03_pmi_ponderado_por_ocorrencia.png",
    ]:
        print("-", nome)

# ============================================================
# EXECUÇÃO EM LOTE: TODOS OS CSVs E JANELAS
# ============================================================

PASTA = Path(r"D:\Downloads")

PASTA_SAIDA_LOTE = PASTA / "resultados_pmi_todos_generos_janelas"
PASTA_SAIDA_LOTE.mkdir(parents=True, exist_ok=True)

ARQUIVOS_ENTRADA = {
    "merged_df_minusculo": PASTA / "merged_df_minusculo_sem_duplicadas.csv",
    "pagode": PASTA / "pagode_tudo_um_csv_com_letras.csv",
    "sertanejo": PASTA / "sertanejo_tudo_um_csv_com_letras.csv",
    "forro": PASTA / "forro_tudo_um_csv_com_letras.csv",
    "funk": PASTA / "funk_tudo_um_csv_com_letras.csv",
    "mpb": PASTA / "mpb_tudo_um_csv_com_letras.csv",
}

JANELAS_CONTEXTO = [4, 2]


def normalizar_nome_coluna(valor):
    texto = str(valor).strip().lower()
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    texto = texto.replace(" ", "")
    return texto


def detectar_encoding(caminho_csv):
    encodings = ["utf-8-sig", "utf-8", "cp1252", "latin1"]

    for enc in encodings:
        try:
            pd.read_csv(caminho_csv, encoding=enc, nrows=0)
            return enc
        except UnicodeDecodeError:
            continue
        except Exception:
            return enc

    return "latin1"


def achar_coluna_por_nome(caminho_csv, nome_desejado):
    encoding = detectar_encoding(caminho_csv)

    df_header = pd.read_csv(
        caminho_csv,
        encoding=encoding,
        nrows=0
    )

    alvo = normalizar_nome_coluna(nome_desejado)

    for coluna in df_header.columns:
        if normalizar_nome_coluna(coluna) == alvo:
            return coluna

    raise ValueError(
        f"Não encontrei a coluna '{nome_desejado}' no arquivo {caminho_csv.name}. "
        f"Colunas encontradas: {list(df_header.columns)}"
    )


def configurar_colunas_para_csv(caminho_csv):
    global COLUNA_TEXTO, COLUNA_MUSICA, COLUNA_ARTISTA

    COLUNA_TEXTO = achar_coluna_por_nome(caminho_csv, "Letra da Música")
    COLUNA_MUSICA = achar_coluna_por_nome(caminho_csv, "Nome da Música")
    COLUNA_ARTISTA = achar_coluna_por_nome(caminho_csv, "Artista")


def montar_prefixo_saida(nome_csv, janela):
    return PASTA_SAIDA_LOTE / f"{nome_csv}_janela{janela}"


def ler_categorias_geradas(prefixo_saida, nome_csv, janela):
    caminho_categorias = Path(f"{prefixo_saida}_categorias.csv")

    if not caminho_categorias.exists():
        return None

    df = pd.read_csv(caminho_categorias, encoding="utf-8-sig")

    df.insert(0, "csv_origem", nome_csv)
    df.insert(1, "janela_contexto", janela)

    return df


def gerar_comparacao_janelas(df_consolidado):
    """
    Compara janela 2 vs janela 4.

    Removido a pedido:
    - contextos_com_categoria_por_100
    - arquivo *_comparacao_contextos_com_categoria_por_100.csv
    - gráfico *_grafico_01_contextos_com_categoria_por_100.png
    - arquivo *_diagnostico_colisoes_alvos.csv
    - arquivo *_auditoria_palavras.csv
    - arquivo *_auditoria_contextos_aparencia.csv
    - gráfico *_grafico_04_top_aparencia_por_genero.png
    - arquivo *_contextos.csv
    - arquivo *_comparacao_pmi_medio_palavras.csv
    - arquivo *_comparacao_pmi_ponderado_por_ocorrencia.csv
    - colunas contextos_com_categoria_por_100 e ocorrencias_categoria_por_100_contextos
      em *_categorias.csv e nos consolidados.
    """
    metricas = [
        "pmi_categoria_contexto_binario",
        "pmi_medio_palavras",
        "pmi_ponderado_por_ocorrencia",
    ]

    linhas = []

    for metrica in metricas:
        if metrica not in df_consolidado.columns:
            continue

        pivot = df_consolidado.pivot_table(
            index=["csv_origem", "genero", "categoria"],
            columns="janela_contexto",
            values=metrica,
            aggfunc="first"
        ).reset_index()

        if 2 not in pivot.columns or 4 not in pivot.columns:
            continue

        for _, row in pivot.iterrows():
            valor_janela_2 = row[2]
            valor_janela_4 = row[4]

            linhas.append({
                "csv_origem": row["csv_origem"],
                "genero": row["genero"],
                "categoria": row["categoria"],
                "metrica": metrica,
                "valor_janela_2": valor_janela_2,
                "valor_janela_4": valor_janela_4,
                "diferenca_janela_4_menos_2": valor_janela_4 - valor_janela_2
            })

    return pd.DataFrame(linhas)


def limpar_outputs_removidos(prefixo_saida):
    """
    Remove sobras antigas dos outputs removidos, caso já existam de execuções anteriores.
    Assim a pasta final não fica com arquivo velho dando impressão de que foi gerado de novo.
    """
    sufixos_removidos = [
        "_comparacao_contextos_com_categoria_por_100.csv",
        "_grafico_01_contextos_com_categoria_por_100.png",
        "_diagnostico_colisoes_alvos.csv",
        "_auditoria_palavras.csv",
        "_auditoria_contextos_aparencia.csv",
        "_grafico_04_top_aparencia_por_genero.png",
        "_contextos.csv",
        "_comparacao_pmi_medio_palavras.csv",
        "_comparacao_pmi_ponderado_por_ocorrencia.csv",
    ]

    for sufixo in sufixos_removidos:
        caminho = Path(f"{prefixo_saida}{sufixo}")
        if caminho.exists():
            caminho.unlink()


def main_lote():
    global CAMINHO_CSV, JANELA_CONTEXTO, PREFIXO_SAIDA
    global USAR_PRIMEIRAS_N_MUSICAS, REMOVER_ALVO_DO_CONTEXTO, DEDUPLICAR_CONTEXTOS

    resultados_categorias = []
    relatorio_execucoes = []

    print("=" * 80)
    print("RODANDO PMI PARA TODOS OS CSVs E JANELAS")
    print("Script standalone otimizado: tokeniza cada corpus uma vez e reaproveita para janela 4 e 2")
    print("=" * 80)

    print("\nArquivos que serão processados:")
    for nome_csv, caminho_csv in ARQUIVOS_ENTRADA.items():
        print(f"- {nome_csv}: {caminho_csv}")

    print("\nJanelas de contexto:")
    for janela in JANELAS_CONTEXTO:
        print(f"- {janela}")

    total_execucoes_previstas = len(ARQUIVOS_ENTRADA) * len(JANELAS_CONTEXTO)
    print(f"\nTotal de saídas por janela previstas: {total_execucoes_previstas}")
    print("Observação: o spaCy será executado uma vez por CSV, não uma vez por CSV+janela.")

    print("\nPreparando listas de alvos e atributos uma única vez...")
    alvos_norm, atributos_norm = obter_listas_preparadas()

    for nome_csv, caminho_csv in ARQUIVOS_ENTRADA.items():
        if not caminho_csv.exists():
            print(f"\n[AVISO] Arquivo não encontrado: {caminho_csv}")

            for janela in JANELAS_CONTEXTO:
                relatorio_execucoes.append({
                    "csv_origem": nome_csv,
                    "caminho_csv": str(caminho_csv),
                    "janela_contexto": janela,
                    "status": "arquivo_nao_encontrado",
                    "erro": ""
                })

            continue

        try:
            print("\n" + "#" * 80)
            print(f"CSV: {nome_csv}")
            print(f"Arquivo: {caminho_csv}")
            print("#" * 80)

            configurar_colunas_para_csv(caminho_csv)

            CAMINHO_CSV = str(caminho_csv)
            USAR_PRIMEIRAS_N_MUSICAS = None
            REMOVER_ALVO_DO_CONTEXTO = True
            DEDUPLICAR_CONTEXTOS = False

            df = carregar_dataframe_csv(caminho_csv)
            registros_tokenizados = tokenizar_dataframe_uma_vez(df)

            if not registros_tokenizados:
                raise ValueError("Nenhum token foi gerado para este CSV. Confira a coluna de letras.")

        except Exception as erro_csv:
            print(f"\n[ERRO] Falhou ao preparar/tokenizar o CSV: {nome_csv}")
            print(str(erro_csv))
            traceback.print_exc()

            for janela in JANELAS_CONTEXTO:
                relatorio_execucoes.append({
                    "csv_origem": nome_csv,
                    "caminho_csv": str(caminho_csv),
                    "janela_contexto": janela,
                    "status": "erro_tokenizacao_csv",
                    "erro": str(erro_csv)
                })

            continue

        for janela in JANELAS_CONTEXTO:
            print("\n" + "=" * 80)
            print(f"CSV: {nome_csv}")
            print(f"Janela de contexto: {janela}")
            print("=" * 80)

            prefixo_saida = montar_prefixo_saida(nome_csv, janela)

            try:
                JANELA_CONTEXTO = janela
                PREFIXO_SAIDA = str(prefixo_saida)

                print("Extraindo contextos a partir dos tokens já calculados...")
                contextos = extrair_contextos_de_tokens(
                    registros_tokenizados,
                    alvos_norm,
                    janela_contexto=janela
                )

                calcular_pmi_e_gerar_outputs(
                    contextos=contextos,
                    qtd_musicas=len(df),
                    alvos_norm=alvos_norm,
                    atributos_norm=atributos_norm,
                    prefixo_saida=str(prefixo_saida),
                    janela_contexto=janela,
                )

                limpar_outputs_removidos(prefixo_saida)

                df_cat = ler_categorias_geradas(
                    prefixo_saida=prefixo_saida,
                    nome_csv=nome_csv,
                    janela=janela
                )

                if df_cat is not None:
                    resultados_categorias.append(df_cat)

                relatorio_execucoes.append({
                    "csv_origem": nome_csv,
                    "caminho_csv": str(caminho_csv),
                    "janela_contexto": janela,
                    "status": "ok",
                    "erro": ""
                })

                print(f"\n[OK] Finalizado: {nome_csv} | janela {janela}")

            except Exception as erro:
                print(f"\n[ERRO] Falhou: {nome_csv} | janela {janela}")
                print(str(erro))
                traceback.print_exc()

                relatorio_execucoes.append({
                    "csv_origem": nome_csv,
                    "caminho_csv": str(caminho_csv),
                    "janela_contexto": janela,
                    "status": "erro",
                    "erro": str(erro)
                })

    caminho_relatorio = PASTA_SAIDA_LOTE / "relatorio_execucoes.csv"

    pd.DataFrame(relatorio_execucoes).to_csv(
        caminho_relatorio,
        index=False,
        encoding="utf-8-sig"
    )

    if resultados_categorias:
        df_consolidado = pd.concat(resultados_categorias, ignore_index=True)

        caminho_consolidado = (
            PASTA_SAIDA_LOTE / "resumo_consolidado_categorias_todos_csvs_janelas.csv"
        )

        df_consolidado.to_csv(
            caminho_consolidado,
            index=False,
            encoding="utf-8-sig"
        )

        df_comparacao = gerar_comparacao_janelas(df_consolidado)

        caminho_comparacao = (
            PASTA_SAIDA_LOTE / "comparacao_janela4_vs_janela2_todos_csvs.csv"
        )

        df_comparacao.to_csv(
            caminho_comparacao,
            index=False,
            encoding="utf-8-sig"
        )

        print("\n" + "=" * 80)
        print("ARQUIVOS CONSOLIDADOS GERADOS")
        print("=" * 80)
        print(f"Resumo consolidado: {caminho_consolidado}")
        print(f"Comparação janela 4 vs 2: {caminho_comparacao}")

    else:
        print("\nNenhum resultado de categoria foi gerado.")

    print("\n" + "=" * 80)
    print("PROCESSAMENTO FINALIZADO")
    print("=" * 80)
    print(f"Relatório de execuções: {caminho_relatorio}")
    print(f"Pasta de saída: {PASTA_SAIDA_LOTE}")


if __name__ == "__main__":
    main_lote()
