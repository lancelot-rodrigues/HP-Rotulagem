# Processamento e Rotulagem de Dados para Detecção de Pirataria de Produtos HP

## 1. Introdução

Este repositório documenta as etapas de tratamento de dados, o processo de feature engineering e a rotulagem heurística que conduzimos com o intuito de construir um dataset estruturado. Este dataset servirá como alicerce para o desenvolvimento futuro de uma solução de Machine Learning, visando a identificação de anúncios potencialmente fraudulentos ou de pirataria de produtos HP em plataformas de e-commerce. O foco aqui é demonstrar a transformação dos dados brutos coletados em um conjunto de informações enriquecido e preparado para modelagem, refletindo as decisões e aprendizados do grupo durante esta jornada.

## 2. Coleta de Dados (Web Scraping)

Os dados primários para este projeto foram obtidos através de um processo de Web Scraping, que se concentrou na varredura da plataforma de e-commerce Mercado Livre em busca de anúncios de produtos HP. Nesta etapa, foram coletadas informações como descrições de produtos, preços, nomes de vendedores, avaliações, entre outros atributos relevantes para a análise.

É importante destacar que a metodologia detalhada, as ferramentas empregadas e os desafios encontrados durante a fase de coleta de dados foram extensivamente documentados em um projeto anterior. Este trabalho de scraping foi desenvolvido durante o Challenge Sprint 1 de RPA. Para um aprofundamento completo nesta etapa, recomendamos a consulta ao repositório dedicado: [https://github.com/lancelot-rodrigues/HP-Scraping](https://github.com/lancelot-rodrigues/HP-Scraping).

## 3. Features (Variáveis) do Dataset

O dataset utilizado neste trabalho é composto por features originais (diretamente coletadas) e por novas features desenvolvidas através do processo de feature engineering. O objetivo do feature engineering foi enriquecer a análise e fornecer subsídios mais robustos para o futuro modelo de Machine Learning.

### 3.1. Features Originais (Coletadas do CSV de Entrada)

As seguintes features foram consideradas como dados de entrada (provenientes do arquivo `ml_produtos_hp_processado_amostra.csv`):

* `link_anuncio`: URL do anúncio do produto.
* `titulo`: Título do anúncio.
* `preco`: Preço de venda do produto.
* `vendedor`: Nome do vendedor/loja.
* `avaliacao_nota`: Nota média de avaliação.
* `avaliacao_numero`: Quantidade de avaliações.
* `descrição`: Texto descritivo do produto.
* `marca`: Marca do produto.
* `modelo`: Modelo específico do produto.
* `cor`: Cor do produto.
* `tipo_cartucho`: Tipo específico do cartucho (ex: "Tinta", "Toner"), usado para categorização.
* `originalidade`: Alegação do vendedor sobre a originalidade (ex: "Original", "Compatível").
* `rendimento_paginas`: Estimativa de rendimento.
* `impressoras_compativeis`: Modelos de impressoras compatíveis.
* `volume_ml_ou_peso_g`: Volume ou peso do produto.
* `estado_produto`: Condição do produto (ex: "Novo").
* `nota_qualidade`: Avaliação numérica da qualidade (presente no input).
* `justificativa_qualidade`: Justificativa textual para a `nota_qualidade` (presente no input).

### 3.2. Novas Features (Resultado de Feature Engineering)

Para aumentar o poder preditivo do dataset, o trabalho de feature engineering resultou na criação das seguintes features:

* **`seller_trust_level` (Nível de Confiança do Vendedor):**
    * **Descrição:** Um score numérico (1: Suspeito, 2: Neutro/Avaliar Internamente, 3: Confiável Alto).
    * **Lógica:** Construída a partir de pesquisa manual do grupo em fontes como Reclame Aqui e percepção geral da reputação dos vendedores. Este é um score inicial que se beneficiará de validações futuras e dados da HP.

* **`preco_medio_categoria` (Preço Médio da Categoria):**
    * **Descrição:** O preço médio dos produtos dentro de uma mesma categoria (definida por `tipo_cartucho`).
    * **Lógica:** Média dos preços limpos por categoria. Implementamos "caps" (limites) para preços individuais e para as médias de categoria calculadas, convertendo valores implausivelmente altos para NaN, garantindo robustez.

* **`desvio_preco_media_categoria` (Desvio do Preço em Relação à Média da Categoria):**
    * **Descrição:** Desvio percentual do preço de um produto em relação à média de sua categoria.
    * **Lógica:** `(preco - preco_medio_categoria) / preco_medio_categoria`. Indica se um produto está mais caro ou mais barato que a média de sua categoria.

## 4. Processo de Rotulagem Heurística (`label_heuristico_calculado`)

Criamos a feature alvo `label_heuristico_calculado` usando um sistema de regras heurísticas, dado que não possuíamos um dataset previamente rotulado.

**Decisões e Lógica da Rotulagem:**

A rotulagem considerou múltiplas features para classificar os anúncios.
* **Alegação de `originalidade`:**
    * Produtos declarados "Compatível" receberam o rótulo `compativel`.
    * Declarações como "Falso" ou "Não Original" levaram ao rótulo `nao_original_declarado`.
* **Ceticismo com Alegações de "Original":** A alegação "Original" foi cruzada com outros fatores:
    * `seller_trust_level`: Alegações de originalidade de vendedores suspeitos (`trust == 1`) foram tratadas com maior escrutínio. Se o preço também fosse baixo, a tendência foi para `pirata_provavel_...`.
    * `desvio_preco_media_categoria`: Produtos alegadamente "Originais" com preços muito abaixo da média da categoria foram sinalizados, especialmente se de vendedores suspeitos.
    * `nota_qualidade`: Baixas notas, quando disponíveis, também influenciaram negativamente.
* **Estrutura e Prioridade das Regras:** As regras foram ordenadas para que indicadores fortes de fraude (ex: preço muito baixo + vendedor suspeito) pudessem sobrepor uma alegação de originalidade.
* **Rótulos de `avaliar_manual_...`:** Usados para casos ambíguos, indicando necessidade de análise humana ou aprendizado mais sutil pelo modelo de ML.

Este conjunto de heurísticas é um ponto de partida e pode ser refinado com mais dados e insights.

## 5. Desafios e Decisões Chave no Processamento

Alguns desafios foram cruciais para as decisões tomadas:

* **Limpeza da Coluna `preco`:** Foi essencial um tratamento robusto para converter diversos formatos de texto de preço em valores numéricos e aplicar um "cap" para valores extremos.
* **Cálculo e Interpretação de `preco_medio_categoria`:** Este foi o ponto mais complexo.
    1.  A dificuldade inicial com médias astronômicas no CSV foi rastreada até a interpretação do separador decimal pelo Excel. **Solução:** Especificar `decimal=','` ao salvar o CSV com Pandas.
    2.  Implementamos "caps" para as médias de categoria e normalizamos a coluna de categoria para garantir consistência.
* **Confiança do Vendedor (`seller_trust_level`):** Esta feature reflete a pesquisa atual do grupo e é reconhecidamente subjetiva nesta fase, aguardando enriquecimento com dados da HP.

## 6. Como Executar o Código (Exemplo)

O script principal (`app.py` ou o nome que você deu) realiza o pipeline de processamento.

1.  **Requisitos:**
    * Python 3.x
    * Pandas: `pip install pandas`
    * NumPy: `pip install numpy` (geralmente instalado com Pandas)
2.  **Entrada:** O script espera um arquivo chamado `ml_produtos_hp_processado_amostra.csv` (ou o nome definido na variável `input_filename`) no mesmo diretório, com separador `;`.
3.  **Execução:**
    ```bash
    python app.py
    ```
4.  **Saída:** Um novo arquivo CSV (ex: `ml_produtos_hp_final_com_features_CSV_CORRIGIDO_v2.csv`) será gerado com as novas features e os rótulos calculados, formatado com `;` como separador de colunas e `,` como separador decimal.

## 7. Conclusão e Próximos Passos

O processo descrito resultou na criação de um dataset estruturado e rotulado heuristicamente, preparando o terreno para o treinamento de modelos de Machine Learning.

**Próximos Passos:**
* Treinar e avaliar modelos de Machine Learning com este dataset.
* Refinar as heurísticas de rotulagem.
* Incorporar futuros dados e validações da HP, especialmente para a confiança do vendedor.
* Expandir o conjunto de features, possivelmente com análise textual.

Acreditamos que este trabalho é um passo importante para uma solução automatizada de auxílio à HP na detecção de produtos suspeitos.
