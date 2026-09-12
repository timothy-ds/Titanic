"""Titanic: 검증, 특징 생성, 누수 방지 파이프라인, CV 선택, 최종 학습."""
from pathlib import Path
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE = Path(__file__).resolve().parent
REQUIRED = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
NUMERIC = ['Age', 'SibSp', 'Parch', 'Fare', 'FamilySize', 'IsAlone', 'FarePerPerson']
CATEGORICAL = ['Pclass', 'Sex', 'Embarked', 'Title', 'Deck']


def validate_data(df, training=False, require_id=False):
    """입력 오류는 조용히 보정하지 않고 설명 가능한 오류로 반환한다."""
    df = df.copy()
    required = REQUIRED + (['Survived'] if training else []) + (['PassengerId'] if require_id else [])
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError('필수 컬럼이 없습니다: ' + ', '.join(missing))
    if df.empty or len(df) > 50000:
        raise ValueError('데이터는 1~50,000행이어야 합니다.')
    for col in ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare'] + (['Survived'] if training else []):
        try:
            df[col] = pd.to_numeric(df[col], errors='raise')
        except (ValueError, TypeError) as exc:
            raise ValueError(f'{col}에는 숫자만 입력하세요.') from exc
        if np.isinf(df[col].to_numpy(dtype=float)).any():
            raise ValueError(f'{col}에 무한대 값이 있습니다.')
    if not df.Pclass.isin([1, 2, 3]).all():
        raise ValueError('Pclass는 1, 2, 3 중 하나여야 합니다.')
    for col in ['Age', 'SibSp', 'Parch', 'Fare']:
        if (df[col].dropna() < 0).any():
            raise ValueError(f'{col}에 음수를 사용할 수 없습니다.')
    for col in ['SibSp', 'Parch']:
        if (df[col].dropna() % 1 != 0).any():
            raise ValueError(f'{col}는 정수여야 합니다.')
    for col, allowed in [('Sex', ['male', 'female']), ('Embarked', ['S', 'C', 'Q'])]:
        values = df[col].astype('string').str.strip()
        values = values.str.lower() if col == 'Sex' else values.str.upper()
        df[col] = values.replace('', pd.NA).astype(object).where(values.notna(), np.nan)
        df[col] = df[col].where(pd.notna(df[col]), np.nan)
        if not df[col].dropna().isin(allowed).all():
            raise ValueError(f'{col} 허용값: {", ".join(allowed)}')
    if training:
        if not df.Survived.isin([0, 1]).all() or df.Survived.nunique() != 2:
            raise ValueError('Survived는 결측치 없는 0/1이며 두 클래스가 모두 필요합니다.')
        if df.Survived.value_counts().min() < 10:
            raise ValueError('5-fold 검증을 위해 각 클래스가 최소 10행 필요합니다.')
    if 'PassengerId' in df:
        ids = pd.to_numeric(df.PassengerId, errors='coerce')
        if ids.isna().any() or not np.isfinite(ids).all() or (ids % 1 != 0).any() or (ids <= 0).any() or ids.duplicated().any():
            raise ValueError('PassengerId는 중복·결측 없는 양의 정수여야 합니다.')
        df['PassengerId'] = ids.astype('int64')
    return df


class TitanicFeatures(TransformerMixin, BaseEstimator):
    """행 단위 특징만 생성한다. 다른 행이나 정답 정보를 참조하지 않는다."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        x = X[REQUIRED].copy()
        x['Pclass'] = x['Pclass'].astype(str)
        x['FamilySize'] = x.SibSp + x.Parch + 1
        x['IsAlone'] = np.where(x.FamilySize.isna(), np.nan, (x.FamilySize == 1).astype(float))
        x['FarePerPerson'] = x.Fare / x.FamilySize
        names = X.get('Name', pd.Series('', index=X.index)).fillna('').astype(str)
        title = names.str.extract(r',\s*([^.]*)\.', expand=False).str.strip()
        title = title.replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})
        x['Title'] = title.where(title.isin(['Mr', 'Mrs', 'Miss', 'Master']), 'Rare').where(title.notna(), 'Unknown')
        cabins = X.get('Cabin', pd.Series('', index=X.index)).fillna('').astype(str)
        x['Deck'] = cabins.str.strip().str.slice(0, 1).str.upper().replace('', 'Unknown').fillna('Unknown')
        return x


def make_pipeline(model):
    numeric = Pipeline([('impute', SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True)), ('scale', StandardScaler())])
    categorical = Pipeline([('impute', SimpleImputer(strategy='constant', fill_value='Unknown')), ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    return Pipeline([('features', TitanicFeatures()), ('preprocess', ColumnTransformer([('numeric', numeric, NUMERIC), ('category', categorical, CATEGORICAL)])), ('model', model)])


def train_models(data, seed=42):
    data = validate_data(data, training=True)
    X, y = data.drop(columns='Survived'), data.Survived.astype(int)
    X_dev, X_hold, y_dev, y_hold = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    candidates = {
        'Dummy baseline': DummyClassifier(strategy='most_frequent'),
        'Logistic regression': LogisticRegression(C=1.0, max_iter=2000, random_state=seed),
        'Random forest': RandomForestClassifier(n_estimators=240, max_depth=7, min_samples_leaf=2, max_features=0.8, random_state=seed, n_jobs=1),
        'Gradient boosting': GradientBoostingClassifier(n_estimators=150, learning_rate=0.04, max_depth=2, min_samples_leaf=5, random_state=seed),
    }
    rows = []
    for name, estimator in candidates.items():
        scores = cross_validate(make_pipeline(estimator), X_dev, y_dev, cv=cv, scoring={'accuracy': 'accuracy', 'roc_auc': 'roc_auc'}, n_jobs=1, error_score='raise')
        rows.append({'model': name, 'cv_accuracy': float(scores['test_accuracy'].mean()), 'cv_std': float(scores['test_accuracy'].std()), 'cv_roc_auc': float(scores['test_roc_auc'].mean())})
    comparison = pd.DataFrame(rows).sort_values(['cv_accuracy', 'cv_roc_auc'], ascending=False).reset_index(drop=True)
    winner = comparison.iloc[0]['model']
    evaluation_model = make_pipeline(clone(candidates[winner])).fit(X_dev, y_dev)
    probability = evaluation_model.predict_proba(X_hold)[:, 1]
    prediction = (probability >= 0.5).astype(int)
    fpr, tpr, _ = roc_curve(y_hold, probability)
    report = {
        'selected_model': winner, 'seed': seed, 'development_rows': len(X_dev), 'holdout_rows': len(X_hold), 'final_training_rows': len(X),
        'accuracy': float(accuracy_score(y_hold, prediction)), 'precision': float(precision_score(y_hold, prediction, zero_division=0)),
        'recall': float(recall_score(y_hold, prediction, zero_division=0)), 'f1': float(f1_score(y_hold, prediction, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_hold, probability)), 'confusion_matrix': confusion_matrix(y_hold, prediction, labels=[0, 1]).tolist(),
        'comparison': comparison.to_dict('records'), 'roc': {'false_positive_rate': fpr.tolist(), 'true_positive_rate': tpr.tolist()},
    }
    # 검증 완료 후 전체 train으로 새 모델을 학습한다. test는 fit에 사용하지 않는다.
    final_model = make_pipeline(clone(candidates[winner])).fit(X, y)
    return final_model, report


def predict_batch(model, data, threshold=0.5):
    if not 0 < threshold < 1:
        raise ValueError('임계값은 0과 1 사이여야 합니다.')
    data = validate_data(data, require_id=True)
    probability = model.predict_proba(data)[:, 1]
    submission = pd.DataFrame({'PassengerId': data.PassengerId.to_numpy(), 'Survived': (probability >= threshold).astype(int)})
    details = submission.assign(SurvivalProbability=probability)
    return submission, details


def main():
    parser = argparse.ArgumentParser(description='Titanic 모델 학습 및 Kaggle 제출 파일 생성')
    parser.add_argument('--train', type=Path, default=BASE / 'data/train.csv')
    parser.add_argument('--test', type=Path, default=BASE / 'data/test.csv')
    parser.add_argument('--output', type=Path, default=BASE / 'artifacts')
    args = parser.parse_args()
    # 모듈 이름을 고정하여 joblib 파일을 다른 스크립트에서도 불러올 수 있게 한다.
    from ml import train_models, predict_batch
    model, report = train_models(pd.read_csv(args.train))
    submission, details = predict_batch(model, pd.read_csv(args.test))
    args.output.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output / 'model.joblib')
    (args.output / 'metrics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    submission.to_csv(args.output / 'submission.csv', index=False)
    details.to_csv(args.output / 'predictions.csv', index=False)
    print(json.dumps({k: v for k, v in report.items() if k not in ['roc', 'comparison']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
