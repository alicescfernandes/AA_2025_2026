# 3 - Desenvolvimento

## 3.1 - Escolha dos Classificadores

Para este trabalho, foram selecionados **três classificadores binários** para a tarefa de classificação de diabetes no dataset Pima Indians:

### 3.1.1 - Random Forest Classifier

O **Random Forest** é um classificador obrigatório neste trabalho, conforme especificado no enunciado. Este algoritmo de ensemble combina múltiplas árvores de decisão, cada uma treinada em subconjuntos aleatórios dos dados (bootstrap) e utilizando subconjuntos aleatórios de features em cada split. A predição final é obtida através de votação majoritária entre todas as árvores.

**Vantagens:**
- Reduz overfitting através da combinação de múltiplas árvores
- Não requer normalização dos dados (baseado em árvores de decisão)
- Fornece informação sobre importância das features
- Lida bem com dados não lineares e interações complexas entre features

**Hiperparâmetros a calibrar:**
- `n_estimators`: Número de árvores no ensemble
- `min_samples_split`: Número mínimo de amostras necessárias para dividir um nó
- `min_samples_leaf`: Número mínimo de amostras em cada folha
- `max_features`: Número máximo de features consideradas em cada split

### 3.1.2 - Support Vector Classifier (SVC)

O **Support Vector Classifier** é um algoritmo baseado em máquinas de vetores de suporte que procura encontrar o hiperplano ótimo que separa as classes com a maior margem possível. Pode utilizar diferentes kernels (linear, RBF, sigmoid) para lidar com problemas não lineares.

**Vantagens:**
- Eficaz em espaços de alta dimensionalidade
- Versátil através da escolha de diferentes kernels
- Boa generalização quando bem calibrado
- Útil para problemas de classificação binária

**Hiperparâmetros a calibrar:**
- `C`: Parâmetro de regularização (controla o trade-off entre margem e erro de classificação)
- `kernel`: Tipo de kernel ('linear', 'rbf', 'sigmoid')
- `gamma`: Coeficiente para kernels RBF e sigmoid
- `degree`: Grau do kernel polinomial (quando aplicável)

**Nota:** O SVC é sensível à escala dos dados, pelo que a normalização pode ser particularmente importante para este classificador.

### 3.1.3 - Logistic Regression

A **Regressão Logística** é um modelo linear que utiliza a função logística (sigmoid) para modelar a probabilidade de uma observação pertencer a uma classe. Apesar do nome "regressão", é um algoritmo de classificação amplamente utilizado.

**Vantagens:**
- Modelo interpretável e simples
- Computacionalmente eficiente
- Fornece probabilidades de classe
- Suporta diferentes tipos de regularização (L1, L2, ElasticNet)
- Boa baseline para problemas de classificação binária

**Hiperparâmetros a calibrar:**
- `solver`: Algoritmo de otimização ('newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga')
- `penalty`: Tipo de regularização (None, 'l1', 'l2', 'elasticnet')
- `C`: Parâmetro de regularização (inverso da força de regularização)
- `l1_ratio`: Proporção de L1 na regularização ElasticNet

**Nota:** A Regressão Logística também beneficia da normalização dos dados, especialmente quando utiliza regularização.

---

## 3.2 - Metodologia de Treino e Avaliação

### 3.2.1 - Divisão dos Dados

Os dados serão divididos em dois conjuntos principais:
- **Conjunto de Treino+Validação (70%)**: Utilizado para calibração de hiperparâmetros e treino dos modelos
- **Conjunto de Teste (30%)**: Mantido separado e utilizado apenas para avaliação final dos modelos

A divisão será realizada com **estratificação** (`stratify=y`) para garantir que a proporção de classes (diabetes vs não-diabetes) seja mantida em ambos os conjuntos, essencial dado o desequilíbrio moderado presente no dataset (500 vs 268 amostras).

### 3.2.2 - Calibração de Hiperparâmetros

A calibração dos hiperparâmetros será realizada através de **GridSearchCV** com **validação cruzada estratificada** (`StratifiedKFold`). Esta abordagem permite:

- **Exploração sistemática** do espaço de hiperparâmetros
- **Avaliação robusta** através de múltiplos folds (3 folds)
- **Manutenção da proporção de classes** em cada fold através da estratificação
- **Seleção objetiva** dos melhores parâmetros baseada em métricas de desempenho

Para cada classificador, será definido um grid de parâmetros adequado, testando múltiplas combinações e selecionando aquela que apresenta melhor desempenho na validação cruzada.

### 3.2.3 - Métricas de Avaliação

Para avaliar e comparar o desempenho dos classificadores, serão utilizadas as seguintes métricas:

1. **Matriz de Confusão**: Visualização completa dos verdadeiros/falsos positivos e negativos
2. **Precision**: Proporção de predições positivas que são corretas (TP / (TP + FP))
3. **Recall**: Proporção de casos positivos corretamente identificados (TP / (TP + FN))
4. **F1-Score**: Média harmónica entre Precision e Recall
5. **Curva ROC (Receiver Operating Characteristic)**: Avalia o desempenho do classificador em diferentes thresholds, fornecendo a área sob a curva (AUC-ROC)
6. **Curva Precision-Recall**: Útil especialmente para dados desbalanceados, mostra o trade-off entre precision e recall

A avaliação será realizada tanto no **conjunto de treino** (para detetar possível overfitting) quanto no **conjunto de teste** (para avaliação final imparcial).

---

## 3.3 - Pré-processamento dos Dados

Será investigado o impacto da **normalização dos dados** (transformação para média nula e variância unitária através de `StandardScaler`) no desempenho dos classificadores. A normalização pode ser particularmente importante para:

- **SVC**: Algoritmos baseados em distâncias são sensíveis à escala
- **Logistic Regression**: A regularização funciona melhor com features normalizadas
- **Random Forest**: Embora não seja estritamente necessário, pode ainda assim beneficiar

Além da normalização, será também explorada a aplicação de técnicas de redução de dimensionalidade:
- **PCA (Principal Component Analysis)**: Para reduzir dimensionalidade mantendo variância
- **LDA (Linear Discriminant Analysis)**: Para reduzir dimensionalidade maximizando separação entre classes

---

## 3.4 - Estrutura do Desenvolvimento

O desenvolvimento seguirá a seguinte estrutura:

1. **Carregamento e Análise Inicial dos Dados**: Exploração do dataset, distribuição de classes, estatísticas descritivas
2. **Divisão dos Dados**: Separação estratificada em treino+validação e teste
3. **Calibração de Hiperparâmetros**: GridSearchCV para cada um dos três classificadores
4. **Avaliação Inicial**: Avaliação dos modelos sem pré-processamento
5. **Pré-processamento**: Aplicação de normalização, PCA e LDA
6. **Reavaliação**: Comparação do desempenho com e sem pré-processamento
7. **Análise Comparativa Final**: Comparação rigorosa dos três classificadores e discussão dos resultados

