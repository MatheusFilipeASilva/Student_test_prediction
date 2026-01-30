def feature_changer(df):
    """Faz as transformações necessárias para o df. Recebe um pd.DataFrame, 
    e devolve o mesmo DataFrame com as transformações feitas em suas variáveis."""
    df['study_sleeping_range'] = (df['study_hours'] * df['sleep_hours']/8)
    df['sleep_quality_ord'] = df['sleep_quality'].map({
        'poor': 1,
        'average': 2,
        'good': 3})
    
    df['facility_ord'] = df['facility_rating'].map({
        'low': 1,
        'medium': 2,
        'high': 3
        })
    df['exam_difficulty_order'] = df['exam_difficulty'].map({
        'easy': 1,
        'moderate': 2,
        'hard': 3
    })
        
    return df


def loadmodel(modelo = "model_new_features.cbm"):
    """Carrega o modelo e seus parametros. Ou seja, carrega o modelo (por padrão, o mais recente) da linha de produção"""
    
    model = CatBoostRegressor()
    model.load_model(modelo)
    params = model.get_params()
    return model, params


def get_dfs(df):
    """Toma o df completo, e entrega seus splits (X e y) já com os tratamentos indispensáveis"""
    X_full = df.drop(['exam_score', 'id'], axis=1)
    y_full = df['exam_score']
    cat_vars = X_full.select_dtypes(exclude="number").columns.tolist()
    return X_full, y_full, cat_vars

def start_model(X, y, cat_features, params):
    """Toma X, y, as variáveis categoricas de X, e os parametros de treinamento da função, e retorna
    o modelo já treinado com os dados fornecidos (X e y)"""
    final_model = CatBoostRegressor(**params)
    final_model.fit(X, y, verbose=100, cat_features=cat_vars)
    return final_model