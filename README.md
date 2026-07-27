Análise de viés de gênero em letras de músicas brasileiras

Este repositório reúne códigos, dados, listas lexicais e resultados de uma pesquisa sobre viés de gênero em letras de músicas brasileiras.

A análise utiliza duas abordagens:

PMI, para medir associações entre alvos e atributos em contextos textuais;

SC-WEAT, para medir associações semânticas em modelos de embeddings.

Os experimentos consideram o corpus completo e recortes dos gêneros pagode, sertanejo, forró, funk e MPB.

Estrutura

Códigos: scripts de pré-processamento, contagem, PMI e SC-WEAT;

Dados: arquivos do corpus e recortes por gênero musical;

Listas de alvos e atributos: planilhas usadas nas análises;

Resultados: saídas geradas pelos experimentos.

Requisitos principais

Python 3

pandas

numpy

scipy

gensim

spaCy

openpyxl
