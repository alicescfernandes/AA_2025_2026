# Análise Detalhada da Abordagem TP1_V3 e Guia de Replicação

## Sumário Executivo

Este documento analisa em detalhe a implementação do trabalho prático anterior (`TP1_V3.ipynb`) e fornece um guia completo para replicar a metodologia no novo trabalho sobre classificação de diabetes no dataset Pima Indians.

---

## 1. Análise da Estrutura do Notebook

### 1.1 Organização Geral

O notebook segue uma estrutura lógica e bem documentada:

1. **Importações e Setup** (Cell 0)
2. **Introdução e Contexto** (Cell 1)
3. **Carregamento e Visualização Inicial** (Cell 2)
4. **Divisão dos Dados** (Cells 3-4)
5. **Calibração de Hiperparâmetros** (Cells 5-15)
6. **Avaliação de Desempenho** (Cells 16-21)
7. **Pré-processamento** (Cells 22-31)
8. **Reavaliação com Pré-processamento** (Cells 32-40)

### 1.2 Bibliotecas Utilizadas

```python
# Principais bibliotecas
- pickle: Carregamento de dados
- numpy: Operações numéricas
- matplotlib.pyplot: Visualizações
- sklearn.model_selection: Divisão de dados e validação cruzada
- sklearn.metrics: Métricas de avaliação
- sklearn.ensemble: RandomForestClassifier
- sklearn.linear_model: LogisticRegression
- sklearn.svm: SVC
- sklearn.preprocessing: StandardScaler
- sklearn.decomposition: PCA
- sklearn.discriminant_analysis: LDA
- pandas.DataFrame: Manipulação de resultados do GridSearch
```

---

## 2. Metodologia Detalhada

### 2.1 Carregamento e Análise Inicial dos Dados

**Implementação:**
```python
data = pickle.load(open("pimaDiabetes.p","rb"))
X = data['data']
y = data['target']
```

**Características do Dataset:**
- **Total de amostras:** 768
- **Classe 0 (sem diabetes):** 500 amostras (65.1%)
- **Classe 1 (com diabetes):** 268 amostras (34.9%)
- **Número de features:** 8 atributos
- **Desequilíbrio de classes:** Sim (razão ~1.87:1)

**Análise Crítica:**
- O desequilíbrio de classes é moderado mas significativo
- Pode afetar o desempenho, especialmente para a classe minoritária (diabetes)
- Justifica o uso de estratificação em todas as divisões

### 2.2 Divisão dos Dados

**Metodologia:**
```python
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y
)
```

**Características:**
- **Split 70/30:** 70% treino+validação, 30% teste
- **Estratificação:** `stratify=y` mantém proporção de classes
- **Resultado:**
  - Treino+Validação: 537 amostras (350 classe 0, 187 classe 1)
  - Teste: 231 amostras (150 classe 0, 81 classe 1)

**Justificativa:**
- 30% para teste é adequado para datasets deste tamanho
- Estratificação é essencial devido ao desequilíbrio
- O conjunto de teste é mantido separado até a avaliação final

### 2.3 Calibração de Hiperparâmetros

#### 2.3.1 Estratégia Geral

**Técnica:** GridSearchCV com validação cruzada estratificada

**Configuração Base:**
```python
GridSearchCV(
    estimator=Modelo(),
    param_grid=parametros,
    cv=StratifiedKFold(n_splits=3),
    verbose=1,
    n_jobs=-1
)
```

**Características:**
- **Validação Cruzada:** 3 folds estratificados
- **Paralelização:** `n_jobs=-1` (usa todos os cores)
- **Feedback:** `verbose=1` mostra progresso

#### 2.3.2 Random Forest Classifier

**Grid de Parâmetros:**
```python
parameters_random_forest = {
    "n_estimators": np.arange(100, 700, 50),      # 12 valores: 100-650
    "min_samples_split": np.arange(2, 13, 1),      # 11 valores: 2-12
    "min_samples_leaf": np.arange(2, 13, 1),       # 11 valores: 2-12
    "max_features": ["sqrt", "log2", None],        # 3 valores
}
```

**Total de Combinações:** 12 × 11 × 11 × 3 = 4,356 combinações
**Total de Fits:** 4,356 × 3 folds = 13,068 modelos treinados

**Análise dos Intervalos:**
- `n_estimators`: Intervalo amplo (100-650) permite encontrar ponto ótimo
- `min_samples_split` e `min_samples_leaf`: Valores baixos (2-12) previnem overfitting
- `max_features`: Inclui opções padrão (sqrt, log2) e completa (None)

#### 2.3.3 Logistic Regression

**Grid de Parâmetros:**
```python
parameters_logistic_regression = [
    {
        "solver": ["newton-cg", "lbfgs", "sag", "saga"],
        "penalty": [None],
        "C": [0.01, 0.02, 0.05, 0.1, 1, 10, 100, 200, 300],
    },
    {
        "solver": ["newton-cg", "lbfgs", "sag"],
        "penalty": ["l2"],
        "C": [0.01, 0.02, 0.05, 0.1, 1, 10, 100, 200, 300],
    },
    {
        "solver": ["liblinear"],
        "penalty": ["l1", "l2"],
        "C": [0.01, 0.02, 0.05, 0.1, 1, 10, 100, 200, 300],
    },
    {
        "solver": ["saga"],
        "penalty": ["l1", "l2"],
        "C": [0.01, 0.02, 0.05, 0.1, 1, 10, 100, 200, 300],
    },
    {
        "solver": ["saga"],
        "penalty": ["elasticnet"],
        "C": [0.01, 0.02, 0.05, 0.1, 1, 10, 100, 200, 300],
        'l1_ratio': [0, 0.25, 0.5, 0.75, 1]
    }
]
```

**Características:**
- **Grids Separados:** Necessário devido a incompatibilidades solver/penalty
- **Solver Coverage:** Cobre todos os solvers principais do sklearn
- **Regularização:** Testa None, L1, L2 e ElasticNet
- **C Range:** Amplo (0.01 a 300) cobre desde alta até baixa regularização
- **ElasticNet:** Apenas com solver 'saga', com 5 valores de l1_ratio

**Total de Combinações:** ~144 combinações
**Total de Fits:** 144 × 3 folds = 432 modelos

**Observação Importante:**
- `max_iter=10000` no estimador base para evitar convergência prematura
- `shuffle=True` no StratifiedKFold para Logistic Regression

#### 2.3.4 Support Vector Classifier (SVC)

**Grid de Parâmetros:**
```python
parameters_svc = {
    'C': [0.5, 1, 5, 10, 20, 100, 200, 400],           # 8 valores
    'kernel': ['linear', 'rbf', 'sigmoid'],           # 3 valores
    'gamma': ['scale', 'auto', 0.1, 1, 10, 100, 200], # 7 valores
    'degree': [2, 3, 4, 5]                            # 4 valores
}
```

**Total de Combinações:** 8 × 3 × 7 × 4 = 672 combinações
**Total de Fits:** 672 × 3 folds = 2,016 modelos

**Análise:**
- **Kernels:** Linear, RBF (mais comum) e Sigmoid
- **C Range:** Amplo (0.5 a 400) para diferentes níveis de regularização
- **Gamma:** Inclui 'scale' e 'auto' (padrões) além de valores fixos
- **Degree:** Apenas para kernels polinomiais (não usado por linear/rbf/sigmoid neste caso)

**Observação:**
- `probability=True` necessário para obter `predict_proba()` e calcular curvas ROC/PR

### 2.4 Visualização dos Resultados da Calibração

**Função `plot_heatmap()`:**

```python
def plot_heatmap(x_param, y_param, ax, xlabel, ylabel, results, fig):
    # Extrai valores únicos dos parâmetros
    x_values = results[f"param_{x_param}"].unique()
    y_values = results[f"param_{y_param}"].unique()
    
    # Cria matriz de scores médios
    mean_scores = np.zeros((len(y_values), len(x_values)))
    
    # Calcula score médio para cada combinação
    for i, y_val in enumerate(y_values):
        for j, x_val in enumerate(x_values):
            score = results[
                (results[f"param_{x_param}"] == x_val) &
                (results[f"param_{y_param}"] == y_val)
            ]["mean_test_score"].mean()
            mean_scores[i, j] = score
    
    # Visualização
    cax = ax.imshow(mean_scores, cmap='viridis', aspect='auto', origin='lower')
    # ... configuração de eixos e labels
    # ... adiciona valores numéricos sobrepostos
    fig.colorbar(cax, ax=ax, orientation='vertical')
```

**Características:**
- **6 Heatmaps por Classificador:** Todas as combinações de pares de parâmetros
- **Valores Numéricos:** Scores sobrepostos para leitura precisa
- **Colormap:** 'viridis' para boa visualização
- **Informação:** Mostra `mean_test_score` médio entre folds

**Uso:**
- Random Forest: 6 heatmaps (3×2 grid)
- Logistic Regression: 6 heatmaps (3×2 grid) ou 3 heatmaps (1×3 grid)
- SVC: 6 heatmaps (3×2 grid)

### 2.5 Avaliação de Desempenho

#### 2.5.1 Funções de Visualização

**1. Matriz de Confusão (`plot_confusion_matrix`)**
```python
def plot_confusion_matrix(y_true, y_pred, ax, title):
    cm = confusion_matrix(y_true, y_pred)
    # Visualização com colormap 'Blues'
    # Valores numéricos sobrepostos
    # Labels: "classe 0", "classe 1"
```

**2. Métricas de Classificação (`plot_classification_metrics`)**
```python
def plot_classification_metrics(y_true, y_pred, ax, title):
    # Calcula: Precision, Recall, F1-Score
    # Gráfico de pontos com valores anotados
    # Y-axis: 0 a 1
```

**3. Curva ROC (`plot_roc_curve`)**
```python
def plot_roc_curve(y_true, y_prob, classifier, color, ax):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    # Plota curva ROC com AUC no label
    # Linha diagonal de referência (classificador aleatório)
```

**4. Curva Precision-Recall (`plot_precision_recall_curve`)**
```python
def plot_precision_recall_curve(y_true, y_prob, classifier, color, ax):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    # Plota curva Precision vs Recall
```

#### 2.5.2 Métricas Utilizadas

**Métricas Principais:**
1. **Precision:** TP / (TP + FP) - Precisão das predições positivas
2. **Recall:** TP / (TP + FN) - Taxa de verdadeiros positivos detectados
3. **F1-Score:** 2 × (Precision × Recall) / (Precision + Recall) - Média harmônica
4. **AUC-ROC:** Área sob a curva ROC - Desempenho geral
5. **Matriz de Confusão:** Visualização completa dos erros

**Avaliação em Dois Conjuntos:**
- **Conjunto de Treino:** Detecta overfitting
- **Conjunto de Teste:** Avaliação final imparcial

#### 2.5.3 Análise Crítica dos Resultados

**Problemas Identificados:**
1. **Baixa Precisão:** Modelos têm dificuldade em classificar corretamente
2. **Falsos Negativos:** Crítico no contexto clínico (diabetes não diagnosticado)
3. **Desequilíbrio de Classes:** Afeta especialmente a classe minoritária
4. **Possível Falta de Normalização:** Alguns modelos são sensíveis à escala

**Considerações Contextuais:**
- **Falsos Negativos > Falsos Positivos:** Em diagnóstico médico, é preferível ter falsos positivos
- **Ajuste de Threshold:** Pode ser necessário ajustar threshold de probabilidade
- **Class Weight:** Considerar balanceamento de classes

### 2.6 Pré-processamento dos Dados

#### 2.6.1 Normalização

**Técnica:** StandardScaler
```python
X_norm = StandardScaler().fit_transform(X)
```

**Efeito:**
- Média de cada feature: 0
- Desvio padrão de cada feature: 1
- Fórmula: `(X - mean) / std`

**Justificativa:**
- Alguns classificadores são sensíveis à escala (SVC, Logistic Regression)
- Garante que todas as features tenham impacto similar
- PCA requer normalização para funcionar corretamente

**Visualização:**
- Gráficos de média e variância antes/depois da normalização
- Confirma que normalização foi aplicada corretamente

#### 2.6.2 Análise de Componentes Principais (PCA)

**Implementação:**
```python
# Análise de variância acumulada
cumulative_variance = np.cumsum(PCA().fit(X_norm).explained_variance_ratio_) * 100

# Teste com diferentes números de componentes
for n in range(1, 9):
    pca = PCA(n_components=n, random_state=42)
    X_train_pca = pca.fit_transform(X_train_norm)
    X_test_pca = pca.transform(X_test_norm)
    # Treina modelo e avalia
```

**Observações:**
- **Sem Normalização:** Primeiro componente captura ~88.85% da variância
- **Com Normalização:** Variância distribuída mais uniformemente
- **Critério:** 90-95% de variância acumulada para escolher n_components
- **Uso Final:** `PCA(n_components=0.9)` para 90% de variância

**Justificativa:**
- Reduz dimensionalidade mantendo informação relevante
- Pode melhorar generalização
- Facilita visualização

#### 2.6.3 Linear Discriminant Analysis (LDA)

**Implementação:**
```python
X_lda = LDA().fit_transform(X_norm, y)
```

**Características:**
- **Supervisionado:** Usa informação das classes
- **Objetivo:** Maximizar separação entre classes
- **Máximo de Componentes:** min(n_features, n_classes - 1) = 1 para binário
- **Vantagem:** Focado em classificação, não apenas variância

**Comparação com PCA:**
- PCA: Não supervisionado, maximiza variância geral
- LDA: Supervisionado, maximiza separação entre classes
- LDA pode ser melhor para classificação

#### 2.6.4 Visualização Comparativa

**Gráficos:**
- Dados originais (primeira dimensão)
- Dados normalizados (primeira dimensão)
- Dados com PCA (primeira componente)
- Dados com LDA (única componente)

**Observação:**
- Dados originais muito sobrepostos
- PCA e LDA criam separação mais nítida entre classes

### 2.7 Reavaliação com Pré-processamento

**Estratégia:**
1. Aplicar normalização/PCA/LDA aos dados
2. Dividir novamente com estratificação
3. Treinar modelos com melhores parâmetros encontrados anteriormente
4. Avaliar desempenho
5. Comparar com resultados sem pré-processamento

**Observação:**
- Não recalibra hiperparâmetros após pré-processamento
- Usa mesmos parâmetros ótimos encontrados nos dados originais
- Pode ser melhorado recalibrando após pré-processamento

---

## 3. Guia de Replicação Passo a Passo

### 3.1 Estrutura do Notebook Recomendada

```
1. INTRODUÇÃO
   - Título e descrição do trabalho
   - Contexto do dataset
   - Objetivos

2. IMPORTAÇÕES
   - Todas as bibliotecas necessárias
   - Organizadas por categoria

3. CARREGAMENTO E ANÁLISE INICIAL
   - Carregar dados com pickle
   - Explorar estrutura dos dados
   - Visualizar distribuição de classes
   - Estatísticas descritivas

4. DIVISÃO DOS DADOS
   - train_test_split com estratificação
   - Documentar tamanhos e proporções
   - Justificar escolha do split

5. ESCOLHA DOS CLASSIFICADORES
   - Listar os 3+ classificadores escolhidos
   - Justificar cada escolha
   - Mencionar Random Forest (obrigatório)

6. CALIBRAÇÃO DE HIPERPARÂMETROS
   6.1. Random Forest
        - Definir grid de parâmetros
        - GridSearchCV
        - Visualizar resultados (heatmaps)
        - Documentar melhores parâmetros
   
   6.2. Logistic Regression
        - Definir grids (separados por solver)
        - GridSearchCV
        - Visualizar resultados
        - Documentar melhores parâmetros
   
   6.3. SVC (ou outro)
        - Definir grid de parâmetros
        - GridSearchCV
        - Visualizar resultados
        - Documentar melhores parâmetros

7. AVALIAÇÃO INICIAL (SEM PRÉ-PROCESSAMENTO)
   - Treinar modelos com melhores parâmetros
   - Matrizes de confusão (treino e teste)
   - Métricas: Precision, Recall, F1 (treino e teste)
   - Curvas ROC (treino e teste)
   - Curvas Precision-Recall (treino e teste)
   - Análise crítica dos resultados

8. PRÉ-PROCESSAMENTO
   8.1. Análise dos Dados Originais
        - Média e variância por feature
        - Visualizações
    
   8.2. Normalização
        - Aplicar StandardScaler
        - Verificar média e variância após normalização
        - Justificar necessidade
    
   8.3. Análise PCA
        - Variância explicada por componente
        - Variância acumulada
        - Escolher número de componentes
        - Visualizar dados transformados
    
   8.4. Análise LDA
        - Aplicar LDA
        - Visualizar dados transformados
        - Comparar com PCA
    
   8.5. Visualização Comparativa
        - Dados originais vs normalizados vs PCA vs LDA

9. REAVALIAÇÃO COM PRÉ-PROCESSAMENTO
   9.1. Com Dados Normalizados
        - Dividir dados normalizados
        - Treinar modelos (recalibrar opcional)
        - Avaliar desempenho
        - Comparar com resultados anteriores
    
   9.2. Com PCA
        - Aplicar PCA aos dados normalizados
        - Dividir dados transformados
        - Treinar modelos
        - Avaliar desempenho
    
   9.3. Com LDA
        - Aplicar LDA aos dados normalizados
        - Dividir dados transformados
        - Treinar modelos
        - Avaliar desempenho

10. COMPARAÇÃO FINAL
    - Tabela comparativa de métricas
    - Discussão do impacto do pré-processamento
    - Identificar melhor abordagem

11. CONCLUSÕES
    - Resumo dos resultados
    - Justificativas metodológicas
    - Limitações e melhorias futuras
```

### 3.2 Código Base para Replicação

#### 3.2.1 Setup Inicial

```python
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    confusion_matrix,
    roc_curve, auc,
    precision_recall_curve,
    precision_score, recall_score, f1_score, accuracy_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from pandas import DataFrame
```

#### 3.2.2 Carregamento de Dados

```python
# Carregar dados
data = pickle.load(open("pimaDiabetes.p", "rb"))
X = data['data']
y = data['target']

# Informações básicas
print(f"Shape de X: {X.shape}")
print(f"Shape de y: {y.shape}")
print(f"Distribuição de classes: {np.bincount(y)}")
print(f"Proporção: Classe 0: {np.sum(y==0)/len(y):.2%}, Classe 1: {np.sum(y==1)/len(y):.2%}")
```

#### 3.2.3 Divisão dos Dados

```python
# Divisão estratificada 70/30
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

print(f"Treino+Validação: {len(X_train_val)} amostras")
print(f"  - Classe 0: {np.sum(y_train_val==0)}")
print(f"  - Classe 1: {np.sum(y_train_val==1)}")
print(f"Teste: {len(X_test)} amostras")
print(f"  - Classe 0: {np.sum(y_test==0)}")
print(f"  - Classe 1: {np.sum(y_test==1)}")
```

#### 3.2.4 Função de Heatmap

```python
def plot_heatmap(x_param, y_param, ax, xlabel, ylabel, results, fig):
    """
    Cria heatmap 2D das combinações de parâmetros do GridSearch.
    
    Parameters:
    -----------
    x_param, y_param : str
        Nomes dos parâmetros (sem prefixo 'param_')
    ax : matplotlib.axes.Axes
        Eixo para plotar
    xlabel, ylabel : str
        Labels dos eixos
    results : DataFrame
        DataFrame com resultados do GridSearch (cv_results_)
    fig : matplotlib.figure.Figure
        Figura para adicionar colorbar
    """
    x_values = results[f"param_{x_param}"].unique()
    y_values = results[f"param_{y_param}"].unique()
    mean_scores = np.zeros((len(y_values), len(x_values)))
    
    for i, y_val in enumerate(y_values):
        for j, x_val in enumerate(x_values):
            mask = (
                (results[f"param_{x_param}"] == x_val) &
                (results[f"param_{y_param}"] == y_val)
            )
            score = results[mask]["mean_test_score"].mean()
            mean_scores[i, j] = score

    cax = ax.imshow(mean_scores, cmap='viridis', aspect='auto', origin='lower')
    ax.set_xticks(np.arange(len(x_values)))
    ax.set_yticks(np.arange(len(y_values)))
    ax.set_xticklabels(x_values)
    ax.set_yticklabels(y_values)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    for i in range(len(y_values)):
        for j in range(len(x_values)):
            score = mean_scores[i, j]
            ax.text(j, i, f'{score:.3f}', ha='center', va='center', 
                   color='white', fontsize='medium', fontweight='bold')
            
    fig.colorbar(cax, ax=ax, orientation='vertical')
```

#### 3.2.5 Funções de Avaliação

```python
def plot_confusion_matrix(y_true, y_pred, ax, title):
    """Plota matriz de confusão."""
    cm = confusion_matrix(y_true, y_pred)
    cax = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(cax, ax=ax)

    ax.set_title(title)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["classe 0", "classe 1"])
    ax.set_xticklabels(["classe 0 estimada", "classe 1 estimada"])
    ax.tick_params(axis='y', labelrotation=90)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f'{cm[i, j]}', ha='center', va='center',
                   color='white' if cm[i, j] > cm.max() / 2 else 'black')


def plot_classification_metrics(y_true, y_pred, ax, title):
    """Plota gráfico de barras com métricas de classificação."""
    metrics = ['Precision', 'Recall', 'F1-Score']
    scores = [
        precision_score(y_true, y_pred),
        recall_score(y_true, y_pred),
        f1_score(y_true, y_pred)
    ]

    ax.plot(metrics, scores, 'o', color='blue', markersize=10)

    for i, score in enumerate(scores):
        ax.text(i, score - 0.1, f'{score:.3f}', ha='center',
               va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylim(0, 1)
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.grid()


def plot_roc_curve(y_true, y_prob, classifier, color, ax):
    """Plota curva ROC."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    ax.plot(fpr, tpr, label=f"AUC {classifier} = {roc_auc:.2f}", color=color)
    ax.plot([0, 1], [0, 1], linestyle="--", color='gray')
    ax.set_xlabel("Taxa de Falsos Positivos")
    ax.set_ylabel("Taxa de Verdadeiros Positivos")
    ax.set_title("Curva ROC")
    ax.legend(loc="lower right")
    ax.grid()


def plot_precision_recall_curve(y_true, y_prob, classifier, color, ax):
    """Plota curva Precision-Recall."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ax.plot(recall, precision, label=classifier, color=color)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Curva Precision-Recall")
    ax.legend(loc="lower left")
    ax.grid()
```

#### 3.2.6 GridSearch para Random Forest

```python
parameters_random_forest = {
    "n_estimators": np.arange(100, 700, 50),
    "min_samples_split": np.arange(2, 13, 1),
    "min_samples_leaf": np.arange(2, 13, 1),
    "max_features": ["sqrt", "log2", None],
}

grid_search_rf = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=parameters_random_forest,
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
    verbose=1,
    n_jobs=-1,
    scoring='accuracy'  # ou outra métrica
)

grid_search_rf.fit(X_train_val, y_train_val)

print("Melhores parâmetros Random Forest:")
print(grid_search_rf.best_params_)
print(f"Melhor score: {grid_search_rf.best_score_:.4f}")

# Visualização
results_rf = DataFrame(grid_search_rf.cv_results_).replace({None: 'None'})

fig, axs = plt.subplots(3, 2, figsize=(20, 15))
fig.suptitle('Combinações de Parâmetros vs Score Médio - Random Forest')

plot_heatmap("min_samples_split", "n_estimators", axs[0, 0], 
             "min_samples_split", "n_estimators", results_rf, fig)
plot_heatmap("min_samples_leaf", "n_estimators", axs[0, 1], 
             "min_samples_leaf", "n_estimators", results_rf, fig)
plot_heatmap("max_features", "n_estimators", axs[1, 0], 
             "max_features", "n_estimators", results_rf, fig)
plot_heatmap("min_samples_split", "min_samples_leaf", axs[1, 1], 
             "min_samples_split", "min_samples_leaf", results_rf, fig)
plot_heatmap("max_features", "min_samples_split", axs[2, 0], 
             "max_features", "min_samples_split", results_rf, fig)
plot_heatmap("max_features", "min_samples_leaf", axs[2, 1], 
             "max_features", "min_samples_leaf", results_rf, fig)

plt.tight_layout()
plt.show()
```

#### 3.2.7 GridSearch para Logistic Regression

```python
parameters_logistic_regression = [
    {
        "solver": ["newton-cg", "lbfgs", "sag", "saga"],
        "penalty": [None],
        "C": [0.01, 0.02, 0.05, 0.1, 1, 10, 100, 200, 300],
    },
    {
        "solver": ["newton-cg", "lbfgs", "sag"],
        "penalty": ["l2"],
        "C": [0.01, 0.02, 0.05, 0.1, 1, 10, 100, 200, 300],
    },
    {
        "solver": ["liblinear"],
        "penalty": ["l1", "l2"],
        "C": [0.01, 0.02, 0.05, 0.1, 1, 10, 100, 200, 300],
    },
    {
        "solver": ["saga"],
        "penalty": ["l1", "l2"],
        "C": [0.01, 0.02, 0.05, 0.1, 1, 10, 100, 200, 300],
    },
    {
        "solver": ["saga"],
        "penalty": ["elasticnet"],
        "C": [0.01, 0.02, 0.05, 0.1, 1, 10, 100, 200, 300],
        'l1_ratio': [0, 0.25, 0.5, 0.75, 1]
    }
]

grid_search_lr = GridSearchCV(
    estimator=LogisticRegression(random_state=42, max_iter=10000),
    param_grid=parameters_logistic_regression,
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
    verbose=1,
    n_jobs=-1,
    scoring='accuracy'
)

grid_search_lr.fit(X_train_val, y_train_val)

print("Melhores parâmetros Logistic Regression:")
print(grid_search_lr.best_params_)
print(f"Melhor score: {grid_search_lr.best_score_:.4f}")

# Visualização (similar ao Random Forest)
results_lr = DataFrame(grid_search_lr.cv_results_).replace({None: 'None'})
# ... criar heatmaps
```

#### 3.2.8 GridSearch para SVC

```python
parameters_svc = {
    'C': [0.5, 1, 5, 10, 20, 100, 200, 400],
    'kernel': ['linear', 'rbf', 'sigmoid'],
    'gamma': ['scale', 'auto', 0.1, 1, 10, 100, 200],
    'degree': [2, 3, 4, 5]
}

grid_search_svc = GridSearchCV(
    estimator=SVC(random_state=42, probability=True),
    param_grid=parameters_svc,
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
    verbose=1,
    n_jobs=-1,
    scoring='accuracy'
)

grid_search_svc.fit(X_train_val, y_train_val)

print("Melhores parâmetros SVC:")
print(grid_search_svc.best_params_)
print(f"Melhor score: {grid_search_svc.best_score_:.4f}")

# Visualização (similar aos anteriores)
```

#### 3.2.9 Avaliação Completa

```python
# Treinar modelos finais com melhores parâmetros
rf = RandomForestClassifier(**grid_search_rf.best_params_, random_state=42)
lr = LogisticRegression(**grid_search_lr.best_params_, random_state=42, max_iter=10000)
svc = SVC(**grid_search_svc.best_params_, random_state=42, probability=True)

# Treinar
rf.fit(X_train_val, y_train_val)
lr.fit(X_train_val, y_train_val)
svc.fit(X_train_val, y_train_val)

# Predições no conjunto de teste
y_pred_rf = rf.predict(X_test)
y_pred_lr = lr.predict(X_test)
y_pred_svc = svc.predict(X_test)

y_prob_rf = rf.predict_proba(X_test)[:, 1]
y_prob_lr = lr.predict_proba(X_test)[:, 1]
y_prob_svc = svc.predict_proba(X_test)[:, 1]

# Visualizações
fig, ax = plt.subplots(1, 3, figsize=(20, 5))
fig.suptitle('Matrizes de Confusão - Conjunto de Teste', fontsize=16)
plot_confusion_matrix(y_test, y_pred_rf, ax[0], "Random Forest")
plot_confusion_matrix(y_test, y_pred_lr, ax[1], "Logistic Regression")
plot_confusion_matrix(y_test, y_pred_svc, ax[2], "SVC")
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(1, 3, figsize=(20, 5))
fig.suptitle('Métricas de Classificação - Conjunto de Teste', fontsize=16)
plot_classification_metrics(y_test, y_pred_rf, ax[0], "Random Forest")
plot_classification_metrics(y_test, y_pred_lr, ax[1], "Logistic Regression")
plot_classification_metrics(y_test, y_pred_svc, ax[2], "SVC")
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(1, 2, figsize=(20, 5))
ax[0].set_title("Curva ROC - Conjunto de Teste")
plot_roc_curve(y_test, y_prob_rf, 'Random Forest', 'red', ax[0])
plot_roc_curve(y_test, y_prob_lr, 'Logistic Regression', 'blue', ax[0])
plot_roc_curve(y_test, y_prob_svc, 'SVC', 'green', ax[0])

ax[1].set_title("Curva Precision-Recall - Conjunto de Teste")
plot_precision_recall_curve(y_test, y_prob_rf, 'Random Forest', 'red', ax[1])
plot_precision_recall_curve(y_test, y_prob_lr, 'Logistic Regression', 'blue', ax[1])
plot_precision_recall_curve(y_test, y_prob_svc, 'SVC', 'green', ax[1])
plt.tight_layout()
plt.show()
```

#### 3.2.10 Pré-processamento

```python
# Normalização
scaler = StandardScaler()
X_norm = scaler.fit_transform(X)

# Verificar normalização
print("Média após normalização:", np.mean(X_norm, axis=0))
print("Desvio padrão após normalização:", np.std(X_norm, axis=0))

# Análise PCA
pca_full = PCA()
pca_full.fit(X_norm)
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_) * 100

plt.figure(figsize=(10, 5))
plt.bar(range(1, 9), cumulative_variance)
plt.xlabel('Número de Componentes')
plt.ylabel('Variância Acumulada (%)')
plt.title('Variância Acumulada por Componente Principal')
plt.grid(True)
plt.show()

# Escolher número de componentes (ex: 90% variância)
n_components_pca = np.argmax(cumulative_variance >= 90) + 1
print(f"Número de componentes para 90% variância: {n_components_pca}")

# Aplicar PCA
pca = PCA(n_components=0.9, random_state=42)  # ou n_components=n_components_pca
X_pca = pca.fit_transform(X_norm)

# Aplicar LDA
lda = LDA()
X_lda = lda.fit_transform(X_norm, y)

# Visualização comparativa
fig, axs = plt.subplots(2, 2, figsize=(15, 10))
# ... plotar dados originais, normalizados, PCA, LDA
```

---

## 4. Pontos Críticos e Boas Práticas

### 4.1 Estratificação

**SEMPRE usar `stratify=y` em:**
- `train_test_split()`
- `StratifiedKFold()`

**Razão:** Mantém proporção de classes em todos os splits, essencial para dados desbalanceados.

### 4.2 Validação Cruzada

**Usar StratifiedKFold:**
- Mantém proporção de classes em cada fold
- Mais adequado que KFold para dados desbalanceados
- 3 folds é um bom compromisso entre robustez e tempo

### 4.3 Métricas Múltiplas

**Não confiar apenas em Accuracy:**
- Em dados desbalanceados, accuracy pode ser enganosa
- Usar Precision, Recall, F1-Score, AUC-ROC
- Considerar contexto (falsos negativos vs falsos positivos)

### 4.4 Análise de Overfitting

**Sempre avaliar em dois conjuntos:**
- Treino: Detecta overfitting
- Teste: Avaliação final imparcial
- Se gap grande entre treino e teste → overfitting

### 4.5 Pré-processamento

**Ordem correta:**
1. Dividir dados (treino/teste)
2. Ajustar scaler/PCA/LDA apenas no conjunto de treino
3. Transformar conjunto de teste com o scaler/PCA/LDA ajustado

**Erro comum:** Ajustar no conjunto completo antes de dividir (data leakage)

### 4.6 Documentação

**Documentar:**
- Justificativa de cada escolha metodológica
- Intervalos de parâmetros testados
- Resultados e interpretação
- Limitações e melhorias futuras

---

## 5. Melhorias e Extensões Possíveis

### 5.1 Balanceamento de Classes

```python
# Opção 1: class_weight
rf = RandomForestClassifier(class_weight='balanced', ...)

# Opção 2: SMOTE (requer imbalanced-learn)
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train_val, y_train_val)
```

### 5.2 Ajuste de Threshold

```python
# Ajustar threshold para priorizar recall
from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
# Escolher threshold que maximize recall mantendo precision aceitável
```

### 5.3 Ensemble Methods

```python
from sklearn.ensemble import VotingClassifier
ensemble = VotingClassifier(
    estimators=[('rf', rf), ('lr', lr), ('svc', svc)],
    voting='soft'
)
```

### 5.4 Feature Engineering

- Seleção de features
- Criação de features interativas
- Tratamento de outliers

### 5.5 Recalibração Após Pré-processamento

- Recalibrar hiperparâmetros após aplicar normalização/PCA/LDA
- Pode melhorar resultados

---

## 6. Checklist de Replicação

- [ ] Carregar dados corretamente
- [ ] Analisar distribuição de classes
- [ ] Dividir dados com estratificação (70/30)
- [ ] Escolher 3+ classificadores (incluindo Random Forest)
- [ ] Definir grids de parâmetros adequados
- [ ] Executar GridSearchCV com StratifiedKFold (3 folds)
- [ ] Visualizar resultados com heatmaps
- [ ] Treinar modelos com melhores parâmetros
- [ ] Avaliar em treino e teste
- [ ] Calcular métricas: Precision, Recall, F1, AUC
- [ ] Plotar matrizes de confusão
- [ ] Plotar curvas ROC e Precision-Recall
- [ ] Aplicar normalização (StandardScaler)
- [ ] Analisar PCA (variância acumulada)
- [ ] Aplicar LDA
- [ ] Reavaliar com dados pré-processados
- [ ] Comparar resultados antes/depois pré-processamento
- [ ] Documentar todas as decisões metodológicas
- [ ] Justificar escolhas e interpretar resultados

---

## 7. Conclusão

Este guia fornece uma análise completa da abordagem utilizada no trabalho anterior e um roteiro detalhado para replicação. A metodologia é sólida e segue boas práticas de machine learning:

- Divisão adequada de dados com estratificação
- Calibração sistemática de hiperparâmetros
- Avaliação com múltiplas métricas
- Análise de pré-processamento
- Documentação completa

Ao seguir este guia, você terá uma base sólida para implementar o novo trabalho, podendo adaptar e melhorar conforme necessário.

