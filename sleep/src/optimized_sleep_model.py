import json

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import joblib
import warnings

warnings.filterwarnings('ignore')

RANDOM_STATE = 42
QUALITY_MAPPING = {
    'Very Bad': 1,
    'Pretty Bad': 2,
    'Quite Good': 3,
    'Very Good': 4
}
QUALITY_LABELS = {value: key for key, value in QUALITY_MAPPING.items()}


class OptimizedSleepModel:
    """Train and evaluate the sleep quality model.

    The test set is split off before any fitted preprocessing, and SMOTE is applied
    only to training data (inside each cross-validation fold), so the reported
    scores are measured on real, unseen survey responses.
    """

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.data_path = self.project_root / "data" / "P2.4_Military_Responses.csv"
        self.models_dir = self.project_root / "models"
        self.results_dir = self.project_root / "results"
        self.models_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        self.feature_names = []

    def load_data(self):
        df = pd.read_csv(self.data_path)
        print(f"Date încărcate: {df.shape}")
        return df

    @staticmethod
    def _quality_column(df):
        for col in df.columns:
            if 'sleep quality' in col.lower():
                return col
        raise ValueError("Nu am găsit coloana cu calitatea somnului.")

    def prepare_target(self, df):
        quality_col = self._quality_column(df)
        df['sleep_quality_numeric'] = df[quality_col].map(QUALITY_MAPPING)

        missing = df['sleep_quality_numeric'].isna().sum()
        if missing:
            print(f"Eliminăm {missing} rânduri fără o valoare validă pentru calitatea somnului.")
            df = df.dropna(subset=['sleep_quality_numeric']).reset_index(drop=True)
        df['sleep_quality_numeric'] = df['sleep_quality_numeric'].astype(int)

        print("Distribuția calității somnului:")
        print(df['sleep_quality_numeric'].value_counts().sort_index())

        return df, 'sleep_quality_numeric'

    def preprocess_features(self, df, target_col):
        """Turn survey answers into numeric features.

        Only fixed, row-by-row transformations happen here. Anything learned from the
        data (missing-value means, scaling) is fitted later on the training set only.
        """
        quality_col = self._quality_column(df)

        exclude_cols = [target_col, quality_col]
        for col in df.columns:
            if 'timestamp' in col.lower() or 'id' in col.lower():
                exclude_cols.append(col)

        features_df = df.drop(exclude_cols, axis=1, errors='ignore').copy()

        # Time answers (e.g. bedtime "23:30") are converted to minutes before any
        # encoding, so they keep their numeric meaning.
        time_cols = [col for col in features_df.columns if any(kw in col.lower() for kw in ['time', 'hour'])]
        for col in time_cols:
            features_df[f"{col}_minutes"] = features_df[col].map(self._to_minutes)

        frequency_mapping = {
            'Never': 0,
            'Less than 1': 0.5,
            '1-2 times': 1.5,
            'More than 3': 4,
            'Less than once a week': 0.5,
            'Once or twice a week': 1.5,
            'Three or more times a week': 4
        }

        for col in features_df.columns:
            if not pd.api.types.is_numeric_dtype(features_df[col]):
                if any(kw in col.lower() for kw in ['frequency', 'often', 'times']):
                    features_df[col] = features_df[col].map(frequency_mapping)
                else:
                    # Label codes depend only on the set of answer options, not on the target.
                    le = LabelEncoder()
                    features_df[col] = le.fit_transform(features_df[col].fillna('Missing').astype(str))

        for col in features_df.columns:
            if not pd.api.types.is_numeric_dtype(features_df[col]):
                features_df[col] = pd.to_numeric(features_df[col], errors='coerce')

        self.feature_names = features_df.columns.tolist()
        y = df[target_col]

        return features_df, y

    @staticmethod
    def _to_minutes(val):
        if pd.isna(val):
            return np.nan
        time_str = str(val).strip()
        try:
            if ':' in time_str:
                parts = time_str.split(':')
                hours = int(parts[0])
                minutes = int(parts[1]) if len(parts) > 1 else 0
            else:
                hours = int(float(time_str))
                minutes = 0
        except ValueError:
            return np.nan

        total_minutes = hours * 60 + minutes
        if total_minutes < 720:
            total_minutes += 1440
        return total_minutes

    def split_and_scale(self, X, y):
        """Hold out a test set first, then fit imputation and scaling on training data only."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
        )

        self.train_means = X_train.mean()
        X_train = X_train.fillna(self.train_means).fillna(0)
        X_test = X_test.fillna(self.train_means).fillna(0)

        self.scaler = StandardScaler()
        X_train = pd.DataFrame(self.scaler.fit_transform(X_train), columns=X.columns, index=X_train.index)
        X_test = pd.DataFrame(self.scaler.transform(X_test), columns=X.columns, index=X_test.index)

        print(f"\nTrain: {len(X_train)} rânduri, test: {len(X_test)} rânduri (date reale, fără SMOTE)")
        print("Distribuția claselor în setul de train înainte de SMOTE:")
        print(y_train.value_counts().sort_index())

        return X_train, X_test, y_train, y_test

    def optimize_random_forest(self, X_train, y_train):
        """Tune the model with SMOTE inside each CV fold, so validation folds stay untouched."""
        print("\nOptimizăm hiperparametrii pentru Random Forest...")

        pipeline = Pipeline([
            ('smote', SMOTE(random_state=RANDOM_STATE)),
            ('rf', RandomForestClassifier(random_state=RANDOM_STATE)),
        ])

        param_grid = {
            'rf__n_estimators': [100, 200, 300],
            'rf__max_depth': [10, 15, 20, None],
            'rf__min_samples_split': [2, 5, 10],
            'rf__min_samples_leaf': [1, 2, 4],
            'rf__class_weight': ['balanced', 'balanced_subsample', None]
        }

        random_search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_grid,
            n_iter=10,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
            scoring='f1_macro',
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbose=1
        )

        random_search.fit(X_train, y_train)

        print(f"\nCei mai buni parametri: {random_search.best_params_}")
        print(f"Cel mai bun scor în cross-validation (F1 macro): {random_search.best_score_:.3f}")

        self.cv_f1 = random_search.best_score_
        self.best_params = random_search.best_params_

        # best_estimator_ is refitted on the whole training set (SMOTE + RF).
        return random_search.best_estimator_

    def evaluate_model(self, pipeline, X_train, X_test, y_train, y_test):
        print("\nEvaluare finală pe setul de test (răspunsuri reale, nevăzute la antrenare):")

        y_pred = pipeline.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')

        # Baseline: always predict the most common class. The model has to beat this.
        baseline = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)
        baseline_pred = baseline.predict(X_test)
        baseline_accuracy = accuracy_score(y_test, baseline_pred)
        baseline_f1 = f1_score(y_test, baseline_pred, average='macro')

        print(f"Acuratețe: {accuracy:.3f}  (baseline clasa majoritară: {baseline_accuracy:.3f})")
        print(f"F1 Score (macro): {f1:.3f}  (baseline: {baseline_f1:.3f})")

        labels = sorted(y_test.unique())
        label_names = [QUALITY_LABELS[label] for label in labels]

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, labels=labels, target_names=label_names, zero_division=0))

        cm = confusion_matrix(y_test, y_pred, labels=labels)
        print("\nConfusion Matrix:")
        print(cm)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=label_names, yticklabels=label_names)
        plt.title('Confusion Matrix - Optimized Model (test set)')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig(self.results_dir / 'optimized_confusion_matrix.png')
        plt.close()

        self.analyze_feature_importance(pipeline.named_steps['rf'], X_train)

        self.metrics = {
            'test_accuracy': round(float(accuracy), 4),
            'test_f1_macro': round(float(f1), 4),
            'baseline_accuracy': round(float(baseline_accuracy), 4),
            'baseline_f1_macro': round(float(baseline_f1), 4),
            'cv_f1_macro': round(float(self.cv_f1), 4),
            'n_train': int(len(X_train)),
            'n_test': int(len(X_test)),
            'best_params': {key: (None if value is None else value) for key, value in self.best_params.items()},
        }

        return accuracy, f1

    def analyze_feature_importance(self, model, X):
        """Analizăm importanța tuturor features"""
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        print(f"\nTop 10 cele mai importante features:")
        print(importance_df.head(10))

        plt.figure(figsize=(12, 10))
        sns.barplot(data=importance_df.head(30), x='importance', y='feature')
        plt.title(f'Top 30 Cele Mai Importante Features')
        plt.xlabel('Importanță')
        plt.tight_layout()
        plt.savefig(self.results_dir / 'optimized_feature_importance.png')
        plt.close()

        return importance_df

    def save_model(self, pipeline):
        model_path = self.models_dir / 'optimized_sleep_model.pkl'

        # SMOTE is only used during training, so the dashboard gets the fitted forest.
        metadata = {
            'model': pipeline.named_steps['rf'],
            'feature_names': self.feature_names,
            'scaler': self.scaler,
            'train_means': self.train_means.to_dict(),
            'metrics': self.metrics,
        }

        joblib.dump(metadata, model_path)
        with open(self.models_dir / 'metrics.json', 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, indent=2)

        print(f"\nModel salvat la: {model_path}")
        print(f"Metrici salvate la: {self.models_dir / 'metrics.json'}")

    def run_pipeline(self):
        df = self.load_data()

        df, target_col = self.prepare_target(df)

        X, y = self.preprocess_features(df, target_col)

        X_train, X_test, y_train, y_test = self.split_and_scale(X, y)

        optimized_pipeline = self.optimize_random_forest(X_train, y_train)

        accuracy, f1 = self.evaluate_model(optimized_pipeline, X_train, X_test, y_train, y_test)

        self.save_model(optimized_pipeline)

        print("\n=== PERFORMANȚĂ MODEL (set de test) ===")
        print(f"Acuratețe: {accuracy * 100:.1f}%  (baseline: {self.metrics['baseline_accuracy'] * 100:.1f}%)")
        print(f"F1 Score (macro): {f1:.3f}  (baseline: {self.metrics['baseline_f1_macro']:.3f})")

        return optimized_pipeline, accuracy, f1


if __name__ == "__main__":
    np.random.seed(RANDOM_STATE)

    optimized_model = OptimizedSleepModel()
    model, accuracy, f1 = optimized_model.run_pipeline()
