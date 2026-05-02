"""
catalyst_model.py
CATALYST: Context-Aware Temporal Attention LSTM for Freeway Anomaly Detection

Extension of: Aslam & Mahfuz (2025), Procedia Computer Science 265, 326-333.
"""

# ── Imports ──────────────────────────────────────────────────────────────────
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    recall_score, precision_score,
    classification_report, confusion_matrix, roc_curve
)
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Bidirectional, LSTM,
    Dense, Dropout, BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from attention_layer import BahdanauAttention

# ── Config ───────────────────────────────────────────────────────────────────
DATA_PATH = '/kaggle/input/datasets/salmanthecodepro/freeway-anomaly-detection/nashville_freeway_anomaly.txt'
BATCH_SIZE = 64
EPOCHS     = 100
SEED       = 42


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Load Data
# ─────────────────────────────────────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded: {df.shape}")
    print(df['human_label'].value_counts())
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Preprocessing
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_data(df: pd.DataFrame):
    # Timestamp features
    df['datetime']    = pd.to_datetime(df['unix_time'], unit='s')
    df['hour']        = df['datetime'].dt.hour
    df['minute']      = df['datetime'].dt.minute
    df['day_of_week'] = df['datetime'].dt.dayofweek

    # Missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].interpolate(method='linear')

    # Normalize base features
    base_cols = [
        'lane1_speed', 'lane2_speed', 'lane3_speed', 'lane4_speed',
        'lane1_volume','lane2_volume','lane3_volume','lane4_volume',
        'lane1_occ',   'lane2_occ',   'lane3_occ',   'lane4_occ',
        'hour', 'minute', 'day_of_week'
    ]
    scaler = MinMaxScaler()
    df[base_cols] = scaler.fit_transform(df[base_cols])
    return df, scaler


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df['avg_speed']       = df[['lane1_speed','lane2_speed','lane3_speed','lane4_speed']].mean(axis=1)
    df['avg_volume']      = df[['lane1_volume','lane2_volume','lane3_volume','lane4_volume']].mean(axis=1)
    df['avg_occ']         = df[['lane1_occ','lane2_occ','lane3_occ','lane4_occ']].mean(axis=1)
    df['speed_change']    = df['avg_speed'].diff().abs()
    df['occ_vol_ratio']   = df['avg_occ'] / (df['avg_volume'] + 1)
    df['speed_roll_mean'] = df['avg_speed'].rolling(5, min_periods=1).mean()
    df['speed_roll_std']  = df['avg_speed'].rolling(5, min_periods=1).std().fillna(0)
    df['speed_vs_avg']    = df['avg_speed'] - df.groupby('unix_time')['avg_speed'].transform('mean')
    df['volume_change']   = df['avg_volume'].diff()
    df['is_peak_hour']    = (((df['hour']>=7)&(df['hour']<=9))|
                              ((df['hour']>=16)&(df['hour']<=18))).astype(int)
    df['is_night']        = ((df['hour']>=22)|(df['hour']<=5)).astype(int)
    return df.ffill().fillna(0)


FEATURE_COLS = [
    'lane1_speed','lane2_speed','lane3_speed','lane4_speed',
    'lane1_volume','lane2_volume','lane3_volume','lane4_volume',
    'lane1_occ','lane2_occ','lane3_occ','lane4_occ',
    'hour','minute','day_of_week',
    'avg_speed','avg_volume','avg_occ',
    'speed_change','occ_vol_ratio',
    'speed_roll_mean','speed_roll_std',
    'speed_vs_avg','volume_change',
    'is_peak_hour','is_night'
]  # 26 features


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Class Balancing (RUS + SMOTE)
# ─────────────────────────────────────────────────────────────────────────────
def balance_data(X, y):
    X_rus, y_rus = RandomUnderSampler(sampling_strategy=0.5, random_state=SEED).fit_resample(X, y)
    print(f"After RUS   — Normal: {sum(y_rus==0)}, Anomaly: {sum(y_rus==1)}")
    X_bal, y_bal = SMOTE(sampling_strategy=1.0, random_state=SEED).fit_resample(X_rus, y_rus)
    print(f"After SMOTE — Normal: {sum(y_bal==0)}, Anomaly: {sum(y_bal==1)}")
    return X_bal, y_bal


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: Train/Val/Test Split & Reshape
# ─────────────────────────────────────────────────────────────────────────────
def prepare_splits(X, y):
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=SEED, stratify=y_temp)

    # Reshape for LSTM: (samples, timesteps=1, features)
    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_val   = X_val.reshape(X_val.shape[0],   1, X_val.shape[1])
    X_test  = X_test.reshape(X_test.shape[0],  1, X_test.shape[1])

    print(f"X_train: {X_train.shape} | X_val: {X_val.shape} | X_test: {X_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: Baseline BiLSTM Model
# ─────────────────────────────────────────────────────────────────────────────
def build_baseline_model(input_shape):
    inputs = Input(shape=input_shape)
    x = Bidirectional(LSTM(128, return_sequences=True))(inputs)
    x = BatchNormalization()(x); x = Dropout(0.2)(x)
    x = LSTM(64, return_sequences=True)(x)
    x = BatchNormalization()(x); x = Dropout(0.2)(x)
    x = LSTM(32)(x); x = Dropout(0.2)(x)
    x = Dense(16, activation='relu')(x); x = Dropout(0.2)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    return Model(inputs, outputs, name='Baseline_BiLSTM')


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7: CATALYST Enhanced Model
# ─────────────────────────────────────────────────────────────────────────────
def build_catalyst_model(input_shape):
    inputs = Input(shape=input_shape)

    # BiLSTM (same as original)
    x = Bidirectional(LSTM(128, return_sequences=True))(inputs)
    x = BatchNormalization()(x); x = Dropout(0.2)(x)

    # Bahdanau Attention (NEW)
    context, _ = BahdanauAttention(units=64)(x)

    # LSTM branch
    x2 = LSTM(64, return_sequences=True)(x)
    x2 = BatchNormalization()(x2); x2 = Dropout(0.2)(x2)
    x3 = LSTM(32)(x2); x3 = Dropout(0.2)(x3)

    # Combine context + LSTM output
    combined = tf.keras.layers.Concatenate()([context, x3])
    x_out = Dense(64, activation='relu')(combined); x_out = Dropout(0.3)(x_out)
    x_out = Dense(16, activation='relu')(x_out);   x_out = Dropout(0.2)(x_out)
    outputs = Dense(1, activation='sigmoid')(x_out)

    return Model(inputs=inputs, outputs=outputs, name='CATALYST')


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8: Train
# ─────────────────────────────────────────────────────────────────────────────
def train_model(model, X_train, y_train, X_val, y_val):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    ]
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    return history


# ─────────────────────────────────────────────────────────────────────────────
# STEP 9: Threshold Optimization
# ─────────────────────────────────────────────────────────────────────────────
def find_optimal_threshold(model, X_val, y_val):
    y_prob      = model.predict(X_val)
    thresholds  = np.arange(0.1, 0.9, 0.01)
    results     = []

    for t in thresholds:
        y_pred = (y_prob > t).astype(int)
        results.append({
            'threshold': t,
            'f1':        f1_score(y_val, y_pred),
            'recall':    recall_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred, zero_division=0)
        })

    df        = pd.DataFrame(results)
    best_idx  = df['f1'].idxmax()
    best_t    = df.loc[best_idx, 'threshold']

    print(f"Optimal Threshold : {best_t:.2f}")
    print(f"Best F1           : {df.loc[best_idx,'f1']:.4f}")
    print(f"Recall at best    : {df.loc[best_idx,'recall']:.4f}")

    plt.figure(figsize=(10, 5))
    plt.plot(df['threshold'], df['f1'],        label='F1')
    plt.plot(df['threshold'], df['recall'],    label='Recall')
    plt.plot(df['threshold'], df['precision'], label='Precision')
    plt.axvline(x=best_t, color='red', linestyle='--', label=f'Best={best_t:.2f}')
    plt.xlabel('Threshold'); plt.ylabel('Score')
    plt.title('Threshold Optimization'); plt.legend()
    plt.savefig('threshold_optimization.png', dpi=300)
    plt.show()
    return best_t


# ─────────────────────────────────────────────────────────────────────────────
# STEP 10: Evaluate
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_models(baseline_model, catalyst_model,
                    X_test, y_test, optimal_threshold=0.5):
    base_prob = baseline_model.predict(X_test)
    enh_prob  = catalyst_model.predict(X_test)
    base_pred = (base_prob > 0.5).astype(int).flatten()
    enh_pred  = (enh_prob > optimal_threshold).astype(int).flatten()

    print(f"\n{'Metric':<12} {'Baseline':>10} {'CATALYST':>10} {'Diff':>8}")
    print("-" * 44)
    for name, fn, use_prob in [
        ('Accuracy',  accuracy_score,  False),
        ('AUC-ROC',   roc_auc_score,   True),
        ('F1-Score',  f1_score,        False),
        ('Recall',    recall_score,    False),
        ('Precision', precision_score, False),
    ]:
        b = fn(y_test, base_prob if use_prob else base_pred)
        e = fn(y_test, enh_prob  if use_prob else enh_pred)
        sym = "+" if (e-b) >= 0 else ""
        print(f"{name:<12} {b:>10.4f} {e:>10.4f} {sym+f'{e-b:.4f}':>8}")

    print("\nClassification Report (CATALYST):")
    print(classification_report(y_test, enh_pred,
          target_names=['Normal', 'Anomaly']))
    return base_pred, enh_pred, base_prob, enh_prob


# ─────────────────────────────────────────────────────────────────────────────
# STEP 11: Visualization
# ─────────────────────────────────────────────────────────────────────────────
def visualize(history_baseline, history_catalyst,
              y_test, base_pred, enh_pred,
              base_prob, enh_prob):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    axes[0,0].plot(history_baseline.history['accuracy'], label='Baseline Train')
    axes[0,0].plot(history_catalyst.history['accuracy'],  label='CATALYST Train')
    axes[0,0].set_title('Training Accuracy'); axes[0,0].legend()

    axes[0,1].plot(history_baseline.history['val_accuracy'], label='Baseline Val')
    axes[0,1].plot(history_catalyst.history['val_accuracy'],  label='CATALYST Val')
    axes[0,1].set_title('Validation Accuracy'); axes[0,1].legend()

    sns.heatmap(confusion_matrix(y_test, base_pred), annot=True, fmt='d',
                ax=axes[0,2], cmap='Blues')
    axes[0,2].set_title('Baseline Confusion Matrix')

    sns.heatmap(confusion_matrix(y_test, enh_pred), annot=True, fmt='d',
                ax=axes[1,0], cmap='Greens')
    axes[1,0].set_title('CATALYST Confusion Matrix')

    fpr_b, tpr_b, _ = roc_curve(y_test, base_prob)
    fpr_e, tpr_e, _ = roc_curve(y_test, enh_prob)
    axes[1,1].plot(fpr_b, tpr_b, label=f'Baseline (AUC={roc_auc_score(y_test,base_prob):.3f})')
    axes[1,1].plot(fpr_e, tpr_e, label=f'CATALYST (AUC={roc_auc_score(y_test,enh_prob):.3f})')
    axes[1,1].plot([0,1],[0,1],'k--')
    axes[1,1].set_title('ROC Curve'); axes[1,1].legend()

    axes[1,2].plot(history_baseline.history['loss'],     label='Baseline Train')
    axes[1,2].plot(history_catalyst.history['loss'],      label='CATALYST Train')
    axes[1,2].plot(history_baseline.history['val_loss'], label='Baseline Val', linestyle='--')
    axes[1,2].plot(history_catalyst.history['val_loss'],  label='CATALYST Val', linestyle='--')
    axes[1,2].set_title('Loss Curves'); axes[1,2].legend()

    plt.tight_layout()
    plt.savefig('catalyst_results.png', dpi=300)
    plt.show()
    print("Plots saved: catalyst_results.png")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # 1. Load
    df = load_data(DATA_PATH)

    # 2. Preprocess
    df, scaler = preprocess_data(df)

    # 3. Feature Engineering
    df = feature_engineering(df)

    # 4. Balance
    X = df[FEATURE_COLS].values
    y = df['human_label'].values
    X_bal, y_bal = balance_data(X, y)

    # 5. Split
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_splits(X_bal, y_bal)
    input_shape = (1, len(FEATURE_COLS))

    # 6. Build models
    baseline_model = build_baseline_model(input_shape)
    catalyst_model  = build_catalyst_model(input_shape)

    # 7. Train
    print("\nTraining Baseline...")
    history_baseline = train_model(baseline_model, X_train, y_train, X_val, y_val)

    print("\nTraining CATALYST...")
    history_catalyst = train_model(catalyst_model, X_train, y_train, X_val, y_val)

    # 8. Threshold
    print("\nOptimizing threshold...")
    optimal_threshold = find_optimal_threshold(catalyst_model, X_val, y_val)

    # 9. Evaluate
    base_pred, enh_pred, base_prob, enh_prob = evaluate_models(
        baseline_model, catalyst_model, X_test, y_test, optimal_threshold)

    # 10. Visualize
    visualize(history_baseline, history_catalyst,
              y_test, base_pred, enh_pred, base_prob, enh_prob)
