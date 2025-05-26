import pandas as pd
import numpy as np

# ETAPA 1: Definições Iniciais e Constantes Globais

# Mapeamento de confiança para vendedores conhecidos
# Níveis: 3 = Confiável Alto, 2 = Avaliar Internamente/Neutro, 1 = Suspeito Alto
SELLER_TRUST_MAP = {
    "CARREFOUR": 3, "OBERO INFORMATICA": 3, "HP OFICIAL": 3, 
    "MAGAZINE LUIZA": 3, "AMERICANAS": 3, "KABUM": 3,
    "INKCOR": 1, "CASAPRINT SPEED": 1, "PARK ECOM": 1, "JRWIMPORTACAO": 1, 
    "TONER SHOPS": 1, "VANMASTERCOMERCIO": 1, "ESHOP": 1, 
    "OUTLET PREÇOBAIXO": 1, "SARAIVA COMERCIO": 1, "SCOTCH": 1,
    "SIAD8238404": 2 
}
DEFAULT_SELLER_TRUST = 1 # Nível de confiança padrão para vendedores não mapeados

# Limites para capping de preços durante os cálculos
MAX_PLAUSIBLE_PRICE_INDIVIDUAL = 20000.00 # Para um item individual
MAX_PLAUSIBLE_PRICE_AVERAGE = 5000.00  # Para a média de uma categoria

# --- ETAPA 2: Definição das Funções de Processamento e Criação de Features ---

def clean_price_column(df, column_name='preco', max_plausible_price=MAX_PLAUSIBLE_PRICE_INDIVIDUAL):
    """
    Limpa e converte a coluna de preço para formato numérico.
    Remove 'R$', espaços, trata separadores de milhar/decimal e aplica um cap para valores extremos.
    """
    if column_name not in df.columns:
        return df

    if df[column_name].dtype == 'object': # Processa apenas se for string/objeto
        df[column_name] = (df[column_name].astype(str)
                           .str.replace(r'R\$', '', regex=True)
                           .str.replace(r'\s+', '', regex=True)
                           .str.replace(r'\.(?=.*\d{3}(?:,|$))', '', regex=True) # Remove . de milhar
                           .str.replace(',', '.', regex=True)) # Substitui , decimal por .
    
    df[column_name] = pd.to_numeric(df[column_name], errors='coerce')

    # Aplica cap para valores individuais implausivelmente altos
    if pd.api.types.is_numeric_dtype(df[column_name]):
        extreme_mask = df[column_name] > max_plausible_price
        if extreme_mask.sum() > 0:
            print(f"Info (clean_price_column): {extreme_mask.sum()} preço(s) individuais > {max_plausible_price} em '{column_name}'. Convertidos para NaN.")
            df.loc[extreme_mask, column_name] = np.nan
    return df

def create_seller_trust_level(df, seller_name_col='vendedor', trust_map=None, default_trust=DEFAULT_SELLER_TRUST):
    """
    Cria a feature 'seller_trust_level' baseada no SELLER_TRUST_MAP.
    Normaliza os nomes dos vendedores para correspondência.
    """
    if trust_map is None: trust_map = {}
    def get_trust(seller_name):
        if pd.isna(seller_name): return default_trust
        return trust_map.get(str(seller_name).strip().upper(), default_trust)
    
    df['seller_trust_level'] = df[seller_name_col].apply(get_trust)
    return df

def create_price_features(df_input, price_col='preco', category_col='tipo_cartucho', max_plausible_avg_price=MAX_PLAUSIBLE_PRICE_AVERAGE):
    """
    Cria as features 'preco_medio_categoria' e 'desvio_preco_media_categoria'.
    Normaliza a coluna de categoria e aplica cap nas médias calculadas.
    """
    df = df_input.copy() 

    if price_col not in df.columns or not pd.api.types.is_numeric_dtype(df[price_col]):
        df['preco_medio_categoria'] = np.nan
        df['desvio_preco_media_categoria'] = 0.0
        return df

    # Normaliza a coluna de categoria antes do groupby e merge
    if category_col in df.columns:
        df[category_col] = df[category_col].astype(str).str.strip()
        df.loc[df[category_col].isin(['', 'nan', 'None', 'NAN', 'NONE']), category_col] = 'CATEGORIA_AUSENTE_OU_INVALIDA'
    else: # Se a coluna de categoria não existir, usa média geral
        geral_mean_price = df[df[price_col].notna()][price_col].mean() if df[price_col].notna().any() else np.nan
        if pd.notna(geral_mean_price) and (geral_mean_price > max_plausible_avg_price or geral_mean_price <=0):
            geral_mean_price = np.nan # Aplica cap
        df['preco_medio_categoria'] = geral_mean_price
        df['desvio_preco_media_categoria'] = (df[price_col] - df['preco_medio_categoria']) / df['preco_medio_categoria']
        df['desvio_preco_media_categoria'] = df['desvio_preco_media_categoria'].replace([np.inf, -np.inf, np.nan], 0)
        return df

    # Agrupa APENAS com linhas que têm preço e categoria válidos
    valid_for_grouping = df.dropna(subset=[price_col, category_col])
    preco_medio_calc_series = pd.Series(dtype='float64') 
    
    if not valid_for_grouping.empty and valid_for_grouping[price_col].notna().any():
        preco_medio_calc_series = valid_for_grouping.groupby(category_col)[price_col].mean()
    
    # Deleta colunas alvo preexistentes para garantir valores recalculados
    if 'preco_medio_categoria' in df.columns: del df['preco_medio_categoria']
    temp_avg_col = '_temp_avg_cat_' # Nome temporário para coluna de média
    if temp_avg_col in df.columns: del df[temp_avg_col]

    # Merge das médias calculadas
    if not preco_medio_calc_series.empty:
        df = df.merge(preco_medio_calc_series.rename(temp_avg_col), on=category_col, how='left')
    else:
        df[temp_avg_col] = np.nan

    # Aplica cap na coluna de média de categoria e atribui à coluna final
    if temp_avg_col in df.columns:
        problematic_avg_mask = (df[temp_avg_col] > max_plausible_avg_price) | (df[temp_avg_col] <= 0)
        if problematic_avg_mask.sum() > 0:
            print(f"Info (create_price_features): {problematic_avg_mask.sum()} 'preco_medio_categoria' calculada fora do intervalo (0 < média <= {max_plausible_avg_price}). Convertendo para NaN.")
            df.loc[problematic_avg_mask, temp_avg_col] = np.nan
        df['preco_medio_categoria'] = df[temp_avg_col]
        del df[temp_avg_col] # Remove coluna temporária
    else: # Se a coluna temporária não foi criada (ex: preco_medio_calc_series vazio)
        df['preco_medio_categoria'] = np.nan

    # Garante que a coluna exista como float
    if 'preco_medio_categoria' not in df.columns: df['preco_medio_categoria'] = np.nan

    # Calcula desvio
    df['desvio_preco_media_categoria'] = (df[price_col] - df['preco_medio_categoria']) / df['preco_medio_categoria']
    df['desvio_preco_media_categoria'] = df['desvio_preco_media_categoria'].replace([np.inf, -np.inf, np.nan], 0)
    return df

def create_heuristic_label(df):
    """
    Cria a feature 'label_heuristico_calculado' com base em um conjunto de regras heurísticas,
    considerando originalidade alegada, confiança do vendedor, desvio de preço e nota de qualidade.
    """
    if 'originalidade' in df.columns:
        df['originalidade_norm'] = df['originalidade'].astype(str).str.strip().str.lower()
    else: 
        df['originalidade_norm'] = 'desconhecido'

    if 'nota_qualidade' in df.columns: # Normaliza nota_qualidade se existir
        if not pd.api.types.is_numeric_dtype(df['nota_qualidade']):
            df['nota_qualidade'] = pd.to_numeric(df['nota_qualidade'], errors='coerce')
    else: 
        df['nota_qualidade'] = np.nan # Garante que a coluna exista como float

    def apply_rules(row):
        originalidade_claim = row['originalidade_norm']
        desvio = row['desvio_preco_media_categoria'] if pd.notna(row['desvio_preco_media_categoria']) else 0
        trust = row['seller_trust_level']
        nota_q = row['nota_qualidade'] if pd.notna(row['nota_qualidade']) else 3 # Default neutro se NaN

        # Nível 1: Alegações explícitas de NÃO ser original
        if originalidade_claim == 'compatível': return 'compativel'
        if originalidade_claim in ['falso', 'não original', 'pirata', 'suspeito', 'alternativo']:
            return 'nao_original_declarado'
        
        # Nível 2: Fortes indicadores de ser PIRATA (podem sobrepor alegação de "original")
        if trust == 1: # Vendedor MUITO suspeito
            if desvio < -0.40: return 'pirata_provavel_preco_vendedor_ruim'
            if pd.notna(row['nota_qualidade']) and nota_q < 2.5: return 'pirata_provavel_nota_vendedor_ruim'
            # Se alega original, mas vendedor é ruim e preço não é tão baixo (outros sinais podem ser necessários)
            if originalidade_claim in ['original', 'verdadeiro', 'genuíno'] and desvio >= -0.20 :
                 return 'avaliar_manual_alegado_original_vendedor_suspeito'
        
        if trust == 2: # Vendedor Neutro
            if desvio < -0.60: return 'pirata_provavel_preco_muito_baixo_vendedor_neutro'
            # Se alega original, vendedor neutro, mas preço baixo
            if originalidade_claim in ['original', 'verdadeiro', 'genuíno'] and desvio < -0.30 :
                return 'avaliar_manual_alegado_original_vendedor_neutro_preco_baixo'

        # Nível 3: Avaliando alegação de "ORIGINAL"
        if originalidade_claim in ['original', 'verdadeiro', 'genuíno']:
            if trust == 3: # Vendedor Confiável Alto
                if desvio >= -0.30: return 'original' # Preço normal ou pouca promoção
                elif desvio < -0.30 and desvio >= -0.50: # Preço um pouco baixo (promoção?)
                    return 'avaliar_manual_original_preco_baixo_vendedor_confiavel'
                else: # Preço MUITO baixo para vendedor confiável
                    return 'avaliar_manual_original_preco_MUITO_baixo_vendedor_confiavel'
            
            elif trust == 2 and desvio >= -0.20 : # Vendedor neutro, alegado original, preço normal
                return 'avaliar_manual_alegado_original_vendedor_neutro'
            # Caso de trust == 1 e originalidade_claim == 'original' já foi parcialmente coberto.
            # Para maior segurança, se ainda não classificado:
            elif trust == 1 : return 'avaliar_manual_alegado_original_vendedor_suspeito'

        # Nível 4: Originalidade Desconhecida/Não especificada
        if originalidade_claim in ['desconhecido', 'nan', 'na', 'categoria_ausente_ou_invalida']:
            if trust == 3:
                if desvio >= -0.30: return 'original_provavel_orig_desconhecida_vendedor_confiavel'
                else: return 'avaliar_manual_orig_desconhecida_vendedor_confiavel_preco_baixo'
            elif trust == 2: return 'avaliar_manual_orig_desconhecida_vendedor_neutro'
            elif trust == 1:
                 if desvio < -0.30 :
                     return 'pirata_provavel_orig_desconhecida_vendedor_suspeito_preco_baixo'
                 return 'avaliar_manual_orig_desconhecida_vendedor_suspeito'

        return 'avaliar_manual_geral_sem_classificacao_clara' # Fallback

    df['label_heuristico_calculado'] = df.apply(apply_rules, axis=1)
    if 'originalidade_norm' in df.columns: # Limpa coluna auxiliar
        df.drop(columns=['originalidade_norm'], inplace=True, errors='ignore')
    return df

# --- ETAPA 3: Execução Principal do Script ---
if __name__ == "__main__":
    # Define nomes de arquivos e colunas principais
    input_filename = "ml_produtos_hp_processado_amostra.csv"
    output_filename = "ml_produtos_hp_final_features_target.csv" # Novo nome para esta versão
    category_column_name = 'tipo_cartucho' # Coluna usada para agrupar produtos por categoria

    # Carrega o dataset de entrada
    try:
        df = pd.read_csv(input_filename, sep=';')
        print(f"Dataset '{input_filename}' carregado com sucesso. Shape: {df.shape}")
    except FileNotFoundError: 
        print(f"Erro: Arquivo '{input_filename}' não encontrado."); exit()
    except Exception as e: 
        print(f"Erro ao carregar o CSV: {e}"); exit()

    df_processed = df.copy() # Trabalha com uma cópia do DataFrame

    # Aplica as funções de processamento e criação de features em sequência
    df_processed = clean_price_column(df_processed, column_name='preco')
    df_processed = create_seller_trust_level(df_processed, seller_name_col='vendedor', trust_map=SELLER_TRUST_MAP)
    df_processed = create_price_features(df_processed, price_col='preco', category_col=category_column_name)
    df_processed = create_heuristic_label(df_processed)
    
    # Exibe informações sobre o resultado da rotulagem
    print("\nFeature 'label_heuristico_calculado' criada.")
    print("Contagem por 'label_heuristico_calculado':\n", df_processed['label_heuristico_calculado'].value_counts())

    # Exibe uma amostra do DataFrame final
    print("\n--- Amostra do DataFrame Processado Final (colunas selecionadas - primeiras 5 linhas) ---")
    cols_to_show = ['titulo', 'preco', 'vendedor', 'seller_trust_level', category_column_name, 
                    'preco_medio_categoria', 'desvio_preco_media_categoria', 'originalidade', 
                    'nota_qualidade', 'label_heuristico_calculado']
    if 'label_heuristico' in df_processed.columns: # Adiciona label original, se existir
        cols_to_show.append('label_heuristico')
    
    existing_cols_to_show = [col for col in cols_to_show if col in df_processed.columns]
    print(df_processed[existing_cols_to_show].head())
    
    # Salva o DataFrame processado em um novo arquivo CSV, usando vírgula como separador decimal
    try:
        df_processed.to_csv(output_filename, sep=';', index=False, encoding='utf-8-sig', decimal=',')
        print(f"\nDataset processado salvo como '{output_filename}' (com separador decimal VÍRGULA).")
        print("Por favor, verifique este novo arquivo no Excel ou outro visualizador de CSV.")
    except Exception as e: 
        print(f"\nErro ao salvar o CSV final: {e}")