import streamlit as st
import pandas as pd
import sklearn


import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

@st.cache_data
def load_data():
    df = pd.read_csv("C:/Users/estevao.maia/Documents/Modelo preditivo/Bronze/UCI_Credit_Card.csv")
    df.columns = df.columns.str.strip()
    return df

df = load_data()
st.title("Análise de Inadimplência de Crédito")

st.header("Análise Exploratória")
if st.checkbox("Mostrar estatísticas descritivas"):
    st.write(df.describe())

if st.checkbox("Mostrar histogramas"):
    col = st.selectbox("Escolha uma coluna numérica:", df.select_dtypes(include='number').columns)
    fig, ax = plt.subplots()
    sns.histplot(df[col], kde=True, ax=ax)
    st.pyplot(fig)

if st.checkbox("Mostrar boxplot"):
    fig, ax = plt.subplots()
    sns.boxplot(data=df.select_dtypes(include='number'), ax=ax)
    plt.xticks(rotation=90)
    st.pyplot(fig)

if st.checkbox("Mostrar correlação"):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(), cmap="coolwarm", annot=False, ax=ax)
    st.pyplot(fig)

st.header("Pré-processamento")

X = df.drop(columns=['ID', 'default.payment.next.month'])
y = df['default.payment.next.month']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

st.header("Modelagem e Avaliação")

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

model_choice = st.selectbox("Escolha o modelo", ["Random Forest", "Logistic Regression", "XGBoost"])

if model_choice == "Random Forest":
    model = RandomForestClassifier()
    params = {'n_estimators': [100, 200], 'max_depth': [4, 8, None]}
elif model_choice == "Logistic Regression":
    model = LogisticRegression(max_iter=1000)
    params = {'C': [0.01, 0.1, 1, 10]}
else:
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    params = {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1]}

if st.button("Treinar modelo"):
    grid = GridSearchCV(model, params, cv=3, scoring='f1')
    grid.fit(X_train, y_train)
    y_pred = grid.predict(X_test)

    st.subheader("Melhores Hiperparâmetros")
    st.write(grid.best_params_)

    st.subheader("Métricas")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.write(pd.DataFrame(report).transpose())

    st.subheader("Matriz de Confusão")
    st.write(confusion_matrix(y_test, y_pred))

    st.subheader("AUC-ROC")
    y_proba = grid.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    st.write(f"AUC: {auc:.2f}")
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label='ROC Curve')
    ax.plot([0, 1], [0, 1], '--')
    st.pyplot(fig)
