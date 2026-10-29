import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn import metrics

APP_DIR = Path(__file__).parent

FEATURES = [
    'fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
    'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density',
    'pH', 'sulphates', 'alcohol', 'is_red',
]


@st.cache_resource
def load_model():
    return joblib.load(APP_DIR / 'model.joblib')


@st.cache_data
def load_config():
    return json.loads((APP_DIR / 'config.json').read_text())


@st.cache_data
def load_test():
    df = pd.read_csv(APP_DIR / 'test.csv')
    return df[FEATURES], df['target']


@st.cache_data
def test_proba(_model, X):
    return _model.predict_proba(X)[:, 1]


@st.cache_data
def curves(y, p):
    fpr, tpr, _ = metrics.roc_curve(y, p)
    prec, rec, _ = metrics.precision_recall_curve(y, p)
    return fpr, tpr, prec, rec


model = load_model()
config = load_config()
X_test, y_test = load_test()
p_test = test_proba(model, X_test)

if 'threshold' not in st.session_state:
    st.session_state.threshold = float(config['threshold'])

with st.sidebar:
    st.header('Настройки')
    threshold = st.slider('Порог для «хорошего» вина', 0.0, 1.0, step=0.001, format='%.3f', key='threshold')
    st.caption(f'Порог по умолчанию (подобран на валидации): {config["threshold"]:.3f}')

st.title('Модель подбора хороших вин')

tab_one, tab_metrics, tab_batch, tab_coef = st.tabs(
    ['Оценка одного вина', 'Модельные метрики', 'Пакетное предсказание', 'Что выучила модель'])

with tab_one:
    st.subheader('Результаты лабораторного анализа')
    stats = X_test.describe().T
    values = {}
    cols = st.columns(3)
    for i, feature in enumerate(FEATURES[:-1]):
        lo, hi, med = stats.loc[feature, ['min', 'max', '50%']]
        values[feature] = cols[i % 3].number_input(
            feature, min_value=float(lo), max_value=float(hi), value=float(med),
            step=float((hi - lo) / 100), format='%.4f')
    wine_type = st.radio('Тип вина', ['белое', 'красное'], horizontal=True)
    values['is_red'] = int(wine_type == 'красное')

    proba = model.predict_proba(pd.DataFrame([values])[FEATURES])[0, 1]
    c1, c2 = st.columns(2)
    c1.metric('Вероятность, что вино хорошее', f'{proba:.1%}')
    if proba >= threshold:
        c2.success(f'Берём в подборку (вероятность ≥ порога {threshold:.3f})')
    else:
        c2.warning(f'Не берём в подборку (вероятность < порога {threshold:.3f})')

with tab_metrics:
    pred = (p_test >= threshold).astype(int)
    cm = metrics.confusion_matrix(y_test, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    c1, c2, c3 = st.columns(3)
    c1.metric('Precision', f'{metrics.precision_score(y_test, pred, zero_division=0):.3f}')
    c2.metric('Recall', f'{metrics.recall_score(y_test, pred):.3f}')
    c3.metric('F1', f'{metrics.f1_score(y_test, pred):.3f}')

    st.info(
        'Как работает порог. Если порог увеличить, в подборку попадут только вина, '
        'в которых модель уверена: посредственных станет меньше (precision растёт), но подборка '
        'станет короче, и больше хороших вин окажется за бортом (recall падает). Если порог '
        'уменьшить, подборка станет шире и пропустит меньше хороших вин, но в неё чаще будут '
        'попадать посредственные.')

    fpr, tpr, prec, rec = curves(y_test, p_test)
    point_fpr = fp / (fp + tn) if fp + tn else 0.0
    point_tpr = tp / (tp + fn) if tp + fn else 0.0
    point_prec = tp / (tp + fp) if tp + fp else 1.0

    fig, ax = plt.subplots(figsize=(4, 3.5))
    metrics.ConfusionMatrixDisplay(cm, display_labels=['не хорошее', 'хорошее']).plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set_title('Матрица ошибок')
    st.columns(2)[0].pyplot(fig)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(fpr, tpr, label=f'ROC-AUC = {metrics.roc_auc_score(y_test, p_test):.3f}')
    axes[0].plot([0, 1], [0, 1], 'k--', lw=0.8)
    axes[0].scatter(point_fpr, point_tpr, color='red', s=70, zorder=3, label=f'порог {threshold:.3f}')
    axes[0].set(xlabel='FPR', ylabel='TPR (recall)', title='ROC-кривая')
    axes[0].legend(loc='lower right')

    axes[1].plot(rec, prec, label=f'PR-AUC = {metrics.average_precision_score(y_test, p_test):.3f}')
    axes[1].axhline(y_test.mean(), color='k', ls='--', lw=0.8)
    axes[1].scatter(point_tpr, point_prec, color='red', s=70, zorder=3, label=f'порог {threshold:.3f}')
    axes[1].set(xlabel='Recall', ylabel='Precision', title='PR-кривая')
    axes[1].legend(loc='upper right')
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with tab_batch:
    st.subheader('Пакетное предсказание')
    st.write(f'Загрузите CSV с признаками вин. Используется порог из боковой панели: {threshold:.3f}.')
    st.caption('Нужные столбцы: ' + ', '.join(FEATURES))
    uploaded = st.file_uploader('CSV-файл', type='csv')

    if uploaded is not None:
        batch = pd.read_csv(uploaded)
        missing = [f for f in FEATURES if f not in batch.columns]
        if missing:
            st.error('В файле не хватает столбцов: ' + ', '.join(missing))
        else:
            result = batch.copy()
            result['proba_good'] = model.predict_proba(batch[FEATURES])[:, 1]
            result['decision'] = np.where(result['proba_good'] >= threshold, 'Берём в подборку', 'Не берём в подборку')
            st.dataframe(result)
            st.download_button('Скачать результат (CSV)', result.to_csv(index=False).encode('utf-8'), file_name='predictions.csv', mime='text/csv')

with tab_coef:
    st.subheader('Коэффициенты логистической регрессии')
    st.write('Признаки стандартизированы, поэтому коэффициенты можно сравнивать между собой: чем больше модуль, тем сильнее признак влияет на решение.')
    clf = model[-1]
    coef = pd.Series(clf.coef_.ravel(), index=FEATURES).sort_values()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(coef.index, coef.values, color=np.where(coef > 0, 'tab:green', 'tab:purple'))
    ax.axvline(0, color='k', lw=0.8)
    ax.set_xlabel('коэффициент (после StandardScaler)')
    ax.set_title('Зелёные - за хорошее вино, фиолетовые - против')
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    c1, c2 = st.columns(2)
    c1.markdown('**Сильнее всего «за» хорошее вино:**\n\n' +
                '\n'.join(f'- `{f}` ({v:+.3f})' for f, v in coef[::-1].head(3).items()))
    c2.markdown('**Сильнее всего «против»:**\n\n' +
                '\n'.join(f'- `{f}` ({v:+.3f})' for f, v in coef.head(3).items()))
