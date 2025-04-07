import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score

# read data
df = pd.read_csv("/Users/asuka/Downloads/LLM_Stance.csv")

# true label
y_true = df['Sentiment Full'].astype(str).str.strip()

# clean prediction format
svm_pred = df['SVM_Pred'].astype(str).str.strip()
tree_pred = df['Tree_Pred'].astype(str).str.strip()
llm_pred = df['stance_predictions'].astype(str).str.strip()

# evalution
def evaluate_model(name, y_pred):
    print(f"Performance of {name}:")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred, average='macro'))
    print("Recall:", recall_score(y_true, y_pred, average='macro'))

evaluate_model("SVM Model", svm_pred)
evaluate_model("Decision Tree Model", tree_pred)
evaluate_model("LLM Model", llm_pred)
