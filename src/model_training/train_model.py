# src/model_training/train_model.py

import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib
import os

def train_model(symbol, strategy_name):
    print(f"--- Iniciando entrenamiento del modelo para {symbol} y estrategia {strategy_name} ---")

    # 1. Cargar datos
    features_path = f"data/features/{symbol}_features.parquet"
    if not os.path.exists(features_path):
        print(f"Archivo de features no encontrado: {features_path}")
        return

    df = pd.read_parquet(features_path)
    df_trades = df[df['target'] != 0].copy()
    df_trades['target_class'] = (df_trades['target'] > 0).astype(int)

    # 2. Definir Features (X) y Target (y)
    excluded_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Spread', 'target', 'target_class']
    features = [col for col in df_trades.columns if col not in excluded_cols]
    X = df_trades[features]
    y = df_trades['target_class']

    if len(df_trades) < 100:
        print(f"No hay suficientes datos ({len(df_trades)} operaciones) para entrenar un modelo para {symbol}/{strategy_name}.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # 4. Escalar y entrenar
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = lgb.LGBMClassifier(objective='binary', random_state=42)
    model.fit(X_train_scaled, y_train)

    # 5. Guardar los artefactos con el nombre del símbolo
    model_dir = f'models/{strategy_name}'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # --- INICIO DE LA CORRECCIÓN ---
    # Se añade el 'symbol' al nombre de los archivos guardados.
    model_path = os.path.join(model_dir, f'{symbol}_confirmation_model.pkl')
    scaler_path = os.path.join(model_dir, f'{symbol}_feature_scaler.pkl')
    features_path = os.path.join(model_dir, f'{symbol}_model_features.joblib')
    # --- FIN DE LA CORRECCIÓN ---

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(features, features_path)
    
    print(f"Modelo para '{symbol}' guardado en: {model_path}")