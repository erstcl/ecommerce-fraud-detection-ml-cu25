[🇷🇺 Русская версия](README.ru.md) | [🇬🇧 English version](README.md)

# E-Commerce Fraud Detection with SHAP Interpretability

[![Notebook quality](https://github.com/erstcl/ecommerce-fraud-detection-ml-cu25/actions/workflows/notebook-quality.yml/badge.svg)](https://github.com/erstcl/ecommerce-fraud-detection-ml-cu25/actions/workflows/notebook-quality.yml)

**Прикладной ML-проект**: классификация несбалансированных данных и интерпретация модели
**Timeline**: Ноябрь — Декабрь 2024

---

## Задача

Построить систему машинного обучения для выявления мошеннических транзакций в e-commerce на основе поведенческих паттернов пользователей, данных транзакций и security-параметров. Fraud detection — критически важная задача для онлайн-бизнеса: пропущенная мошенническая транзакция = прямые убытки, а ложное срабатывание = потерянный клиент.

---

## Датасет

**Источник**: [E-Commerce Fraud Detection Dataset](https://www.kaggle.com/datasets/umuttuygurr/e-commerce-fraud-detection-dataset) (Kaggle)

### Характеристики
- **Размер**: 299,695 транзакций
- **Признаки**: 17 исходных фичей
- **Целевая переменная**: `is_fraud` (бинарная классификация)
- **Class imbalance**: 2.206% fraud (6,612 из 299,695)
- **Временной период**: 2024 год

### Группы признаков

**User profile**:
- `account_age_days` — возраст аккаунта
- `total_transactions_user` — количество транзакций пользователя
- `avg_amount_user` — средняя сумма транзакций пользователя

**Transaction**:
- `amount` — сумма транзакции
- `shipping_distance_km` — расстояние доставки
- `promo_used` — использование промокода
- `merchant_category` — категория продавца
- `channel` — канал транзакции (web/app)

**Security**:
- `avs_match` — Address Verification System match
- `cvv_result` — результат проверки CVV
- `three_ds_flag` — флаг 3D Secure

**Geo**:
- `country` — страна пользователя
- `bin_country` — страна эмитента карты

**Time**:
- `transaction_time` — timestamp транзакции

---

## Решение

Работа организована в три этапа:

### Этап 1: EDA и Baseline

**Исследование данных**:
- Fraud rate значительно выше у cross-border транзакций (11.28% vs 1.43%)
- Топ-2 страны по fraud: TR (2.80%), RO (2.40%)
- Низкий fraud rate у транзакций со всеми security checks (0.58% для AVS+CVV+3DS)
- Outliers: 11 экстремальных значений по `amount` и `shipping_distance_km`

**Baseline**:
- Модель: CatBoostClassifier
- Validation strategy: Stratified 80/20 split
- Метрика: ROC-AUC = 0.97784, PR-AUC = 0.85269

### Этап 2: Anomaly Detection & Feature Engineering

**Работа с аномалиями**:

Применены 3 подхода к поиску выбросов:

1. **Статистические методы** (Z-score, IQR):
   - `amount`: Z-score outliers (1.79% данных) → fraud rate 30.16% (в 14× выше базового)
   - `shipping_distance_km`: Z-score outliers (3.61%) → fraud rate 17.27% (в 8× выше)
   - IQR-выбросы по `shipping_distance_km` → fraud rate 12.81%

2. **Удаление экстремальных outliers**:
   - Удалены точки с `amount > 10000` и `shipping_distance_km > 10000`
   - Результат: улучшение стабильности модели

3. **ML-методы для поиска сложных аномалий**:
   - Применены: Isolation Forest, LOF, One-Class SVM, Elliptic Envelope
   - Создан `anomaly_count` (консенсусный счётчик аномалий)
   - Создан `consensus_strong_anomaly` (точки, помеченные ≥2 методами)
   - 11,808 точек выявлены как strong anomalies → fraud rate 23.76% (в 11× выше!)
   - **Insight**: Isolation Forest и LOF показали лучшую precision/recall trade-off

**Feature Engineering**:

Созданы 4 группы новых признаков:

1. **Target Encoding** (без data leakage):
   - `merchant_category_te`, `country_te`, `bin_country_te`
   - `channel` закодирован через One-Hot Encoding

2. **User behavior features**:
   - `amount_zscore_user` — Z-score суммы относительно среднего пользователя
   - `dist_zscore_user` — Z-score расстояния доставки
   - `merchant_category` × `amount` — кросс-фичи
   - kNN-based: плотность транзакций в feature space

3. **Temporal features** (sin/cos encoding):
   - `hour_sin/cos`, `dow_sin/cos`, `month_sin/cos`
   - `is_night`, `is_business_hours`, `is_evening`, `is_weekend`

4. **Domain-specific features**:
   - `is_cross_border` — country ≠ bin_country (огромный fraud indicator!)
   - `security_score` — комбинация AVS, CVV, 3DS (взвешенный скор)
   - `all_security_passed` / `no_security` — флаги
   - `amount_to_avg_ratio`, `amount_diff_from_avg` — отклонения от user baseline
   - `is_long_distance` — флаг доставки на большое расстояние (90th percentile)
   - **`risk_score`** — комплексный скор:  
     `risk_score = 3×is_cross_border + 2×no_security + 1×three_ds_flag + is_long_distance + is_night`

**Feature selection**:
- Применён CatBoost feature importances
- Отобраны top-25 признаков для финальной модели
- Удалены нестабильные и дублирующие фичи

**Результат**: После feature engineering baseline улучшен (детали в notebook)

### Этап 3: Interpretability & Shapley Flow

**Интерпретация моделей**:

1. **SHAP global interpretation**:
   - Построены SHAP summary plots для CatBoost
   - Топ влияющие признаки: `security_score`, `risk_score`, `amount`, `cross_border`, `shipping_distance`

2. **LIME local interpretation**:
   - Локальная интерпретация fraud-транзакций
   - Сравнение LIME vs SHAP: LIME показывает более простые линейные аппроксимации, SHAP — полную картину взаимодействий

3. **Model comparison**:
   - Сравнены Logistic Regression (с StandardScaler) и CatBoost
   - SHAP summary plot показывает, что CatBoost лучше улавливает нелинейные паттерны (например, U-shaped зависимости)

**SHAP-эмбеддинги и аномалии**:

1. **Создание SHAP-эмбеддингов**:
   - Функция `get_shap_embeddings(model, X_data, shap_feature)` для извлечения SHAP values
   - SHAP-эмбеддинги для train и test

2. **Anomaly detection на SHAP-эмбеддингах**:
   - Isolation Forest с `contamination=0.01` на SHAP space
   - Выявлено 2,398 SHAP-аномалий
   - **Результат**: ROC-AUC = 0.97340 (незначительное снижение), но модель стала более стабильной

3. **Кластеризация SHAP-эмбеддингов**:
   - **PCA** для снижения размерности до 2 компонент
   - **k-Means** (k=5) → добавлен `cluster` feature
   - Переобучение CatBoost с `cat_features=['cluster']`
   - **Результат**: ROC-AUC = 0.97566 (небольшой прирост)
   - **DBSCAN**: не дал значимого улучшения (много outliers в cluster=-1)

**Shapley Flow анализ**:

1. **Граф взаимосвязей признаков**:
   - Построен граф на основе корреляций SHAP values (|corr| > 0.5)
   - NetworkX для визуализации
   - Community detection (greedy modularity) → выявлено **18 групп признаков**

2. **Ключевые группы (communities)**:
   - Security cluster: `shap_security_score`, `shap_avs_match`, `shap_risk_score`, `shap_all_security_passed`
   - Geography cluster: `shap_shipping_distance_km`, `shap_is_cross_border`, `shap_is_long_distance`
   - User behavior: `shap_user_amount_std`, `shap_avg_amount_user`, `shap_anomaly_consensus`

3. **Train vs Test сравнение**:
   - Test граф более разреженный (5 сообществ vs 4 в train)
   - 4 стабильные группы сохраняются между train/test
   - **Insight**: `shap_is_cross_border` и `shap_shipping_distance_km` всегда в одном кластере → сильная связь

**Финальная валидация**:

Сравнение 3 подходов:
1. **SHAP-эмбеддинги + Isolation Forest**: ROC-AUC = 0.97340
2. **SHAP-эмбеддинги + кластеризация**: ROC-AUC = 0.97566
3. **Исходные признаки (hold-out validation)**: ROC-AUC = **0.97640**

SHAP-эмбеддинги сами по себе (без исходных фичей): ROC-AUC = 0.96381 (на 1.26pp ниже)

---

## Результаты

### Метрики (финальная модель, hold-out validation)

| Metric | Score |
|--------|-------|
| **ROC-AUC** | **0.97640** |
| **PR-AUC (Average Precision)** | **0.85556** |
| **Fraud Precision** | 0.34 |
| **Fraud Recall** | 0.90 |
| **Fraud F1-Score** | 0.49 |

**Прирост относительно baseline**:
- ROC-AUC: +0.00144 (+0.15%)
- PR-AUC: +0.00287 (+0.34%)
- Fraud Precision: +0.04 (+13.3%)
- Fraud Recall: +0.02 (+2.3%)
- Fraud F1: +0.04 (+8.9%)

### Ключевые инсайты

**Бизнес-выводы**:
1. **Cross-border транзакции** — главный fraud indicator (fraud rate 11.28% vs 1.43%)
2. **Security checks критичны**: комбинация AVS+CVV+3DS снижает fraud в 32× (0.58% vs 18.08%)
3. **Аномалии по сумме и расстоянию** — сильные сигналы (fraud rate до 30% у outliers)
4. **География важна**: TR и RO — топ страны по fraud

**Технические выводы**:
1. Feature engineering даёт больший эффект, чем тюнинг гиперпараметров
2. SHAP-эмбеддинги полезны для интерпретации, но не заменяют исходные признаки
3. Anomaly detection методы помогают выявить сложные паттерны (консенсус-подход эффективен)
4. Shapley Flow раскрывает структуру взаимодействий признаков

---

## Воспроизведение анализа

```bash
git clone https://github.com/erstcl/ecommerce-fraud-detection-ml-cu25.git
cd ecommerce-fraud-detection-ml-cu25
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab ml_project.ipynb
```

Notebook скачивает используемое зеркало датасета. Сохранённая версия остаётся
проверяемым аналитическим отчётом; CI валидирует JSON-структуру и не допускает
закоммиченные error outputs.

## Технологии

### ML Stack

**Data processing**:
- pandas, numpy
- scikit-learn (preprocessing, imputation, feature selection)

**Visualization**:
- matplotlib, seaborn
- plotly (интерактивные графы)

**Anomaly detection**:
- Isolation Forest, LOF, One-Class SVM, Elliptic Envelope

**Modeling**:
- CatBoost (основная модель)
- Logistic Regression (для сравнения)
- RandomForest (feature importance)

**Interpretability**:
- SHAP (TreeExplainer, summary plots, dependence plots)
- LIME (Tabular explainer, SP-LIME)

**Graph analysis**:
- NetworkX (Shapley Flow, community detection)

**Clustering**:
- k-Means, DBSCAN
- PCA, UMAP (dimensionality reduction)

### Environment

- **Platform**: Google Colab
- **Language**: Python 3.12
- **GPU**: NVIDIA Tesla T4 (для ускорения CatBoost)

---

## Границы оценки

Репозиторий содержит трёхэтапное прикладное ML-исследование:

1. **Checkpoint 1** (17-26 ноября): EDA + Baseline
2. **Checkpoint 2** (24 ноября — 3 декабря): Anomaly Detection + Feature Engineering
3. **Checkpoint 3** (1-10 декабря): SHAP Interpretability + Shapley Flow

Все headline-метрики получены на сохранённом stratified hold-out из notebook. Это
экспериментальные результаты на указанном публичном датасете, а не production
performance. Перед внедрением потребовалась бы дополнительная out-of-time
валидация.
