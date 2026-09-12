"""실행: python -m streamlit run app.py"""
import io
import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from ml import train_models, validate_data, predict_batch

BASE = Path(__file__).resolve().parent
st.set_page_config(page_title='Titanic · 생존 예측 실험실', page_icon='🚢', layout='wide')
st.title('Titanic · 생존 예측 실험실')
st.caption('데이터를 살펴보고, 모델을 비교하고, 승객의 생존 가능성을 예측하세요.')


def read_data(upload, filename):
    raw = upload.getvalue() if upload is not None else (BASE / 'data' / filename).read_bytes()
    return pd.read_csv(io.BytesIO(raw), encoding='utf-8-sig')


@st.cache_data(show_spinner=False, max_entries=4)
def cached_train(data):
    # 데이터 내용이 바뀌면 캐시 키가 달라져 이전 모델을 재사용하지 않는다.
    return train_models(data)


def csv_bytes(frame):
    return frame.to_csv(index=False).encode('utf-8-sig')


with st.sidebar:
    st.header('데이터 설정')
    train_upload = st.file_uploader('학습 데이터 · train.csv', type=['csv'], key='train_upload')
    test_upload = st.file_uploader('일괄 예측 데이터 · test.csv', type=['csv'], key='test_upload')
    st.caption('업로드하지 않으면 함께 제공된 CSV를 사용합니다. UTF-8 CSV, 최대 50,000행.')
    st.divider()
    threshold = st.slider('예측 임계값', 0.1, 0.9, 0.5, 0.05)
    st.caption('개인·일괄 예측에 적용됩니다. 성능 평가는 고정 임계값 0.5 기준입니다.')

try:
    train = validate_data(read_data(train_upload, 'train.csv'), training=True)
except (ValueError, OSError, UnicodeError, pd.errors.ParserError) as exc:
    st.error(f'학습 데이터를 읽을 수 없습니다: {exc}')
    st.stop()

test = None
try:
    test = validate_data(read_data(test_upload, 'test.csv'), require_id=True)
except (ValueError, OSError, UnicodeError, pd.errors.ParserError) as exc:
    st.warning(f'일괄 예측 데이터 오류: {exc} 다른 탭은 계속 사용할 수 있습니다.')

a, b, c, d = st.columns(4)
a.metric('학습 승객', f'{len(train):,}명')
b.metric('학습 데이터 생존율', f'{train.Survived.mean():.1%}')
c.metric('예측 대상', f'{len(test):,}명' if test is not None else '없음')
d.metric('나이 결측', f'{train.Age.isna().sum():,}명')
explore, evaluation, single, batch, guide = st.tabs(['데이터 탐색', '모델 학습·평가', '개인 생존 예측', '일괄 예측·다운로드', '사용 안내'])

with explore:
    st.subheader('승객 특성과 생존율')
    left, right = st.columns(2)
    with left:
        group = st.selectbox('비교할 항목', ['Sex', 'Pclass', 'Embarked'])
        summary = train.groupby(group, dropna=False).Survived.agg(['mean', 'count']).rename(columns={'mean': '생존율', 'count': '승객 수'})
        st.bar_chart(summary[['생존율']])
        st.dataframe(summary, width='stretch')
    with right:
        st.write('컬럼별 결측치')
        missing = train.isna().sum().sort_values(ascending=False)
        st.bar_chart(missing[missing > 0])
    st.caption('탐색 그래프는 전체 train 기준의 기술통계입니다. 모델 선택 기준은 아래 교차검증 점수입니다.')
    st.dataframe(train.head(100), width='stretch', hide_index=True)

# 동일 데이터에 대한 결과만 화면에 표시한다. 업로드 변경 시 기존 결과는 무효화한다.
signature = pd.util.hash_pandas_object(train, index=True).values.tobytes() + str(train.columns.tolist()).encode()
if st.session_state.get('training_signature') != signature:
    st.session_state.pop('training_result', None)

with evaluation:
    st.subheader('교차검증으로 선택하고, 별도 데이터로 평가합니다')
    st.write('80% 개발 데이터에서 5-fold 교차검증 → 평균 정확도로 모델 선택 → 남겨둔 20%로 평가 → 전체 데이터로 최종 재학습')
    if st.button('모델 학습 시작', type='primary'):
        with st.spinner('4개 모델을 교차검증하고 있습니다. 첫 실행에는 시간이 걸립니다.'):
            try:
                st.session_state.training_result = cached_train(train)
                st.session_state.training_signature = signature
            except (ValueError, RuntimeError) as exc:
                st.error(f'학습 실패: {exc}')
    result = st.session_state.get('training_result')
    if result is None:
        st.info('모델 학습 시작 버튼을 누르면 성능과 예측 기능이 활성화됩니다.')
    else:
        model, report = result
        st.success(f"선택된 모델: {report['selected_model']}")
        st.dataframe(pd.DataFrame(report['comparison']), width='stretch', hide_index=True)
        columns = st.columns(5)
        for column, key in zip(columns, ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']):
            column.metric(key.upper(), f'{report[key]:.3f}')
        st.caption(f"별도 검증 {report['holdout_rows']}행 · 임계값 0.5 · seed 42. 이 점수는 전체 데이터로 재학습하기 전 모델의 성능입니다.")
        left, right = st.columns(2)
        with left:
            st.write('혼동행렬 · 행=실제 / 열=예측')
            st.dataframe(pd.DataFrame(report['confusion_matrix'], index=['실제 사망', '실제 생존'], columns=['예측 사망', '예측 생존']), width='stretch')
        with right:
            st.write('ROC 곡선')
            st.line_chart(pd.DataFrame(report['roc']), x='false_positive_rate', y='true_positive_rate')
        st.download_button('평가 결과 JSON 다운로드', json.dumps(report, ensure_ascii=False, indent=2), 'metrics.json', 'application/json')

with single:
    st.subheader('승객 정보를 입력하세요')
    with st.form('passenger'):
        left, right = st.columns(2)
        with left:
            sex = st.selectbox('성별', ['female', 'male'], format_func=lambda x: {'female': '여성', 'male': '남성'}[x])
            pclass = st.selectbox('객실 등급', [1, 2, 3], index=2)
            age = st.number_input('나이', min_value=0.0, max_value=120.0, value=28.0, step=1.0)
            age_unknown = st.checkbox('나이 모름')
            title = st.selectbox('호칭', ['Miss', 'Mrs', 'Mr', 'Master', 'Dr', 'Unknown'])
        with right:
            fare = st.number_input('운임 (당시 파운드)', min_value=0.0, value=15.0, step=1.0)
            sibsp = st.number_input('함께 탑승한 형제자매·배우자 수', min_value=0, max_value=20, value=0)
            parch = st.number_input('함께 탑승한 부모·자녀 수', min_value=0, max_value=20, value=0)
            embarked = st.selectbox('탑승 항구', ['S', 'C', 'Q'], format_func=lambda x: {'S': 'S · Southampton', 'C': 'C · Cherbourg', 'Q': 'Q · Queenstown'}[x])
            cabin = st.text_input('객실 번호 (모르면 빈칸)', placeholder='예: C85')
        predict_clicked = st.form_submit_button('생존 가능성 예측', type='primary')
    if predict_clicked:
        if result is None:
            st.warning('모델 학습·평가 탭에서 먼저 학습하세요.')
        else:
            row = pd.DataFrame([dict(PassengerId=1, Pclass=pclass, Sex=sex, Age=np.nan if age_unknown else age, SibSp=sibsp, Parch=parch, Fare=fare, Embarked=embarked, Name='' if title == 'Unknown' else f'Example, {title}. Passenger', Cabin=cabin)])
            _, predicted = predict_batch(result[0], row, threshold)
            probability = float(predicted.SurvivalProbability.iloc[0])
            st.metric('모델이 추정한 생존 확률', f'{probability:.1%}')
            st.progress(probability)
            st.write(f"예측 결과: {'생존' if predicted.Survived.iloc[0] else '사망'} · 임계값 {threshold:.2f}")
            st.caption('확률은 별도 보정되지 않은 모델 출력입니다. 역사 데이터 학습용 결과로 해석하세요.')

with batch:
    st.subheader('일괄 예측 및 Kaggle 제출 파일')
    if result is None:
        st.info('먼저 모델 학습을 실행하세요.')
    elif test is None:
        st.info('올바른 test.csv를 업로드하세요.')
    else:
        if 'Survived' in test:
            st.info('업로드된 Survived 컬럼은 예측에 사용하지 않습니다.')
        submission, details = predict_batch(result[0], test, threshold)
        st.dataframe(details, width='stretch', hide_index=True)
        left, right = st.columns(2)
        left.download_button('Kaggle 제출용 submission.csv', csv_bytes(submission), 'submission.csv', 'text/csv')
        right.download_button('확률 포함 predictions.csv', csv_bytes(details), 'predictions.csv', 'text/csv')
        st.caption('제출 파일은 PassengerId, Survived 두 컬럼으로 구성되며 원본 승객 순서를 유지합니다. test 정답이 없어 여기서 test 정확도는 계산할 수 없습니다.')

with guide:
    st.markdown('''
1. 기본 데이터를 사용하거나 사이드바에서 CSV를 업로드합니다.
2. **모델 학습·평가 → 모델 학습 시작**을 누릅니다.
3. 교차검증 비교표와 별도 검증 성능을 확인합니다.
4. 개인 정보를 입력하거나 일괄 예측 CSV를 다운로드합니다.

**전처리:** 숫자 결측치는 학습 구간의 중앙값으로 대체하고, 범주형은 원핫 인코딩합니다.
가족 크기, 혼자 탑승 여부, 1인당 운임, 호칭, 객실 데크를 추가합니다.
이름 원문·티켓 번호·승객 ID는 직접 학습하지 않습니다.

**성능 해석:** 무작위 승객 분할이므로 가족·동행인이 양쪽에 섞일 수 있습니다.
새로운 가족 집단에 대한 일반화 성능은 별도 그룹 검증이 필요합니다.
100% 정확도나 Kaggle 순위는 보장하지 않습니다.

로컬 실행 및 Streamlit Community Cloud 배포 방법은 프로젝트의 **README.md**에 있습니다.
''')
