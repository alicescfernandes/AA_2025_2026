# Trabalho Laboratorial — Pima Indians Diabetes Dataset






































OLÁ

## 1. Dados

Os dados disponibilizados encontram-se no ficheiro `pimaDiabetes.p`.

O *Pima Indians Diabetes Dataset*, criado pelo **National Institute of Diabetes and Digestive and Kidney Diseases (EUA)**, contém informação sobre **768 mulheres** da população indígena Pima, que apresenta elevada incidência de diabetes.

- **500** mulheres *não têm diabetes*
- **268** mulheres *têm diabetes*

Cada observação inclui **8 características**:

| Característica | Intervalo |
|----------------|-----------|
| Número de gravidezes | [0, 17] |
| Concentração de glicose plasmática (2h após teste de tolerância) | [0, 199] |
| Pressão arterial diastólica (mm Hg) | [0, 122] |
| Espessura da dobra cutânea do tríceps (mm) | [0, 99] |
| Nível sérico de insulina em 2 horas (µh/ml) | [0, 846] |
| Índice de massa corporal (peso/altura²) | [0, 67.1] |
| Diabetes Pedigree Function | [0.078, 2.42] |
| Idade (anos) | [21, 81] |

---

## 2. Objetivos

Pretende-se **determinar automaticamente se um paciente apresenta diabetes**, com base nas características fornecidas.

Para isto, é necessário:

- Treinar e avaliar **três (ou mais) classificadores binários**  
  - Pelo menos um deles **tem obrigatoriamente** de ser `RandomForestClassifier`
- Fazer um **estudo comparativo** dos modelos

---

## 3. Desenvolvimento

### 1. Modelos de Classificação

1. Escolher **3 classificadores binários**, incluindo obrigatoriamente o Random Forest.  
2. Treinar os modelos definindo os **hiperparâmetros** adequados.  
3. Escolher a **metodologia de treino/teste** apropriada, garantindo uma estimativa fidedigna da performance.  
4. Utilizar **métricas adequadas** e **calibrar** os modelos.  
5. Comparar rigorosamente o desempenho dos classificadores.

---

### 2. Pré-processamento dos Dados

Investigar se a **normalização dos dados** (média nula e variância unitária)  melhora o desempenho dos classificadores.

---

### 3. Observações Gerais

O relatório deve justificar:

- escolhas metodológicas (classificadores, treino/teste, métricas)
- análise rigorosa dos resultados
- discussão fundamentada das decisões tomadas

---

## 4. Ficheiro a Entregar

Submeter via Moodle **um único Jupyter Notebook**:

```
AxxxxxAxxxxxAxxxxxTP1.ipynb
```

Onde `Axxxxx` são os números dos alunos (ordenados crescentemente).

O notebook deve ser **bem comentado**, explicando:

- cada passo do processo
- razões para cada escolha
- análise detalhada dos resultados
