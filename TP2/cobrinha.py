import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, classification_matrix

X, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

model = LogisticRegression(
    solver="lbfgs",
    max_iter=1000,
    random_state=42
)

model.fit(X_train, y_train)

y_proba = model.predict_proba(X_test)

print("Probabilidades (primeiras 5 amostras):")
print(y_proba[:5])

classes = model.classes_
y_pred_argmax = classes[np.argmax(y_proba, axis=1)]

confusion_matrix(y_test, y_pred_argmax)
print(y_pred_argmax)

print(classification_report(y_test, y_pred_argmax))
