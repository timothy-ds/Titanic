# 타이타닉 생존 예측 · Streamlit

제공된 train.csv와 test.csv로 학습, 평가, 개인 예측, 일괄 예측을 실행하는 한국어 앱입니다.
Python 3.12를 기준으로 작성했습니다. 외부 API 키나 유료 서비스는 필요하지 않습니다.

제공 데이터로 실제 실행한 결과: train 891행, test 418행. 선택 모델은 Random Forest이며,
개발 데이터 5-fold 평균 정확도 **83.15%**, 별도 검증 179행 정확도 **79.33%**, ROC AUC **0.8536**입니다.
`artifacts/submission.csv`에는 418명의 예측이 들어 있습니다. 자동 테스트 5개가 통과했습니다.
인터넷 공개 배포는 아직 수행하지 않았습니다.

## 1. Windows에서 실행하기

1. ZIP을 압축 해제합니다. `app.py`와 `requirements.txt`가 있는 `titanic-streamlit` 폴더를 엽니다.
2. Python 3.12가 없다면 https://www.python.org/downloads/ 에서 설치합니다. 설치 시 Python을 PATH에 추가하는 옵션을 선택합니다.
3. 해당 폴더에서 터미널(PowerShell)을 열고 아래를 차례로 실행합니다.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

가상환경을 활성화하지 않아도 되어 PowerShell 실행 정책을 바꿀 필요가 없습니다.
`py` 명령이 없다면 Python 설치를 확인하고, Python 3.12를 가리키는 `python`으로 대체하세요.
브라우저가 자동으로 열리지 않으면 http://localhost:8501 로 접속합니다.
종료할 때는 터미널에서 Ctrl+C를 누릅니다. 다음 실행부터는 마지막 명령만 실행하면 됩니다.

macOS / Linux:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m streamlit run app.py
```

## 2. 앱 사용 순서

1. 제공된 데이터를 그대로 사용하거나 사이드바에서 새로운 CSV를 업로드합니다.
2. **모델 학습·평가** 탭에서 **모델 학습 시작**을 누릅니다.
3. 교차검증 비교표, 정확도, 정밀도, 재현율, F1, ROC AUC, 혼동행렬을 확인합니다.
4. **개인 생존 예측**에서 정보를 입력하고 예측 버튼을 누릅니다.
5. **일괄 예측·다운로드**에서 `submission.csv` 또는 확률 포함 파일을 받습니다.

임계값 기본값은 0.5입니다. 사이드바에서 변경하면 개인·일괄 분류 결과에 적용됩니다.
검증 성능은 계속 0.5 기준으로 표시합니다. 모델 학습 결과는 캐시되며 학습 데이터가 달라지면
기존 화면 결과를 무효화합니다. 재시작한 서버에서는 다시 학습 버튼을 누르세요.

## 3. 코드 구성

```text
titanic-streamlit/
├── app.py                 # 한국어 Streamlit 앱
├── ml.py                  # 재사용 가능한 학습·예측 파이프라인 및 CLI
├── requirements.txt       # 직접 사용하는 라이브러리 버전 고정
├── test_project.py        # 입력·추론·직렬화·앱 동작 테스트
├── README.md
├── .gitignore
├── .streamlit/config.toml # 화면 테마와 업로드 용량
├── data/train.csv         # 사용자가 제공한 원본 복사본
├── data/test.csv
└── artifacts/             # CLI 학습 시 생성되는 결과
```

## 4. 머신러닝 설계와 성능 해석

- `Survived`가 타깃이며, 0은 사망, 1은 생존입니다.
- train을 계층화하여 개발 80%, 별도 검증 20%로 나눕니다. 난수 시드는 42입니다.
- 개발 구간에서만 5-fold 교차검증으로 Dummy, Logistic Regression, Random Forest,
  Gradient Boosting을 비교합니다. 평균 정확도가 가장 높은 모델을 선택하며 동률이면 ROC AUC를 사용합니다.
- 결측치 대체와 인코딩은 Pipeline 내부에 있어 각 fold의 학습 구간에만 fit됩니다.
- 숫자는 중앙값 대체 및 표준화, 범주는 결측 대체 및 원핫 인코딩을 사용합니다.
- 가족 크기, 혼자 탑승 여부, 가족 1인당 운임, 이름에서 추출한 호칭, 객실 데크를 생성합니다.
- PassengerId, 이름 원문, Ticket, Survived를 예측 특징으로 직접 사용하지 않습니다.
- 선택된 모델을 별도 검증 구간에서 평가한 후, 전체 train으로 다시 학습하여 test를 예측합니다.
- 검증 성능은 최종 전체 학습 모델 자체를 평가한 값이 아닙니다. test.csv에는 정답이 없으므로
  test 정확도나 Kaggle 점수를 로컬에서 계산할 수 없습니다.
- CV 표준편차는 fold 점수의 변동이며 신뢰구간이 아닙니다. 표시 확률은 별도 보정되지 않았습니다.
- 승객 단위 무작위 분할에서는 가족·동행인이 양쪽에 포함될 수 있습니다. 새로운 가족에 대한 성능을
  확인하려면 그룹 분할 평가가 추가로 필요합니다. 높은 점수를 얻기 위해 검증 결과를 반복적으로 보며
  설정을 바꾸면 평가가 낙관적으로 변할 수 있습니다. 100% 정확도를 보장하는 코드는 아닙니다.

누수 방지 설계 참고: https://scikit-learn.org/stable/common_pitfalls.html

## 5. 앱 없이 학습·예측하기

```powershell
.\.venv\Scripts\python.exe ml.py
```

기본 데이터로 학습하고 `artifacts/model.joblib`, `metrics.json`, `submission.csv`, `predictions.csv`를 생성합니다.
사용자 경로를 지정하려면:

```powershell
.\.venv\Scripts\python.exe ml.py --train data/train.csv --test data/test.csv --output artifacts
```

저장된 모델 사용 예시(프로젝트 폴더에서 실행):

```python
import joblib
import pandas as pd
from ml import predict_batch

model = joblib.load('artifacts/model.joblib')
submission, details = predict_batch(model, pd.read_csv('data/test.csv'))
submission.to_csv('submission.csv', index=False)
```

joblib 파일은 직접 생성한 신뢰할 수 있는 파일만 로드하세요. 모델을 다른 환경에서 사용할 때는
동일한 라이브러리 버전을 설치하세요. 웹앱은 모델 파일 업로드를 받지 않고 CSV로 학습합니다.

## 6. Streamlit Community Cloud 배포

실제 인터넷 공개 배포는 사용자의 GitHub/Streamlit 계정에서 아래 단계로 진행합니다.
현재 전달된 결과는 로컬 실행 및 배포용 프로젝트입니다.

1. GitHub에 로그인하고 새 저장소(예: `titanic-streamlit`)를 만듭니다.
2. 프로젝트 **폴더 안의 파일들**을 저장소 루트에 업로드하고 커밋합니다. 루트에서 `app.py`, `ml.py`,
   `requirements.txt`, `data/train.csv`, `data/test.csv`가 보여야 합니다. `.streamlit/config.toml`도 포함합니다.
   ZIP 자체, `.venv`, `__pycache__`는 업로드하지 마세요. `artifacts`는 앱 실행에 필요하지 않습니다.
3. https://share.streamlit.io 에 로그인하고 GitHub 계정을 연결합니다.
4. **Create app → Yup, I have an app**을 선택합니다.
5. Repository는 만든 저장소, Branch는 실제 브랜치(보통 `main`), Main file path는 `app.py`로 지정합니다.
6. **Advanced settings → Python version**에서 **3.12**를 선택하고 저장합니다. Secrets는 필요 없습니다.
7. **Deploy**를 누릅니다. 의존성 설치가 끝나면 `https://…streamlit.app` 형태의 접속 주소가 생성됩니다.
8. 배포된 앱에서 학습을 실행하고 개인 예측·CSV 다운로드까지 확인합니다.

코드를 수정하고 GitHub에 커밋하면 배포된 앱에 반영됩니다. 배포 과정에서 실패하면 앱 로그를 확인하세요.
제공 데이터가 저장소에 포함되므로 저장소와 앱의 공개 범위는 데이터 공유 의도에 맞게 설정하세요.

공식 배포 안내(2026-09-12 확인):

- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/file-organization

## 7. 입력 형식과 문제 해결

필수 특징 컬럼: `Pclass, Sex, Age, SibSp, Parch, Fare, Embarked`.
학습에는 `Survived`, 일괄 예측에는 중복 없는 양의 정수 `PassengerId`가 추가로 필요합니다.
`Name`, `Cabin`이 없으면 해당 파생 특징을 Unknown으로 처리합니다. `Ticket`은 사용하지 않습니다.
추가 컬럼은 무시하며 파일 내용은 데이터로만 처리합니다.

Sex는 male/female, Embarked는 S/C/Q, Pclass는 1/2/3입니다. 나이·운임·가족 수 결측은 허용합니다.
학습 데이터에는 두 클래스가 각각 10행 이상 있어야 합니다. UTF-8 CSV를 사용하세요.

| 증상 | 해결 |
| --- | --- |
| `ModuleNotFoundError` | 같은 가상환경의 python으로 `-m pip install -r requirements.txt` 실행 |
| `train.csv`가 없다는 오류 | `data/train.csv`와 폴더 대소문자 확인 |
| 배포 후 `app.py`를 찾지 못함 | GitHub에서 app.py의 실제 경로를 Main file path에 입력 |
| 포트 사용 중 | 실행 명령 끝에 `--server.port 8502` 추가 |
| CSV 문자 깨짐 | UTF-8 CSV로 다시 저장 |
| 새 학습 데이터 업로드 후 예측 비활성화 | 모델 학습 버튼을 다시 실행 |

테스트 실행:

```powershell
.\.venv\Scripts\python.exe -m unittest -v test_project.py
```

데이터 스키마 오류, 결측치 처리, 제출 형식·순서, 정답 컬럼 격리, 모델 저장/복원,
Streamlit의 초기 화면·학습·개인 예측·임계값 변경을 검사합니다.
