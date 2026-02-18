import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv("dataset/heart.csv")

X = data.drop("target", axis=1)
y = data["target"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

svm = SVC(probability=True, random_state=42)
rf = RandomForestClassifier(n_estimators=200, random_state=42)
nn = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1500, random_state=42)

svm.fit(X_train, y_train)
rf.fit(X_train, y_train)
nn.fit(X_train, y_train)

svm_acc = accuracy_score(y_test, svm.predict(X_test))
rf_acc = accuracy_score(y_test, rf.predict(X_test))
nn_acc = accuracy_score(y_test, nn.predict(X_test))

hybrid = VotingClassifier(
    estimators=[
        ('svm', svm),
        ('rf', rf),
        ('nn', nn)
    ],
    voting='soft'
)

hybrid.fit(X_train, y_train)

hybrid_acc = accuracy_score(y_test, hybrid.predict(X_test))

model_scores = {
    "SVM": svm_acc,
    "Random Forest": rf_acc,
    "Neural Network": nn_acc,
    "Hybrid": hybrid_acc
}

pickle.dump(hybrid, open("model/hybrid_model.pkl", "wb"))
pickle.dump(scaler, open("model/scaler.pkl", "wb"))
pickle.dump(hybrid_acc, open("model/accuracy.pkl", "wb"))
pickle.dump(model_scores, open("model/model_scores.pkl", "wb"))
pickle.dump(list(X.columns), open("model/feature_names.pkl", "wb"))
