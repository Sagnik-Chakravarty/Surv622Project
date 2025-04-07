import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score

# read data
df = pd.read_csv("/Users/asuka/Downloads/LLM_Stance.csv")

# set ramdom seed
np.random.seed(123)

# get train set
train_idx = np.random.choice(df.index, size=int(0.8 * len(df)), replace=False)
test_idx = df.index.difference(train_idx)
df_test = df.loc[test_idx]


# clean prediction format
y_true = df_test['Sentiment Full'].astype(str).str.strip()
svm_pred = df_test['SVM_Pred'].astype(str).str.strip()
tree_pred = df_test['Tree_Pred'].astype(str).str.strip()
llm_pred = df_test['stance_predictions'].astype(str).str.strip()


# evalution
def evaluate_model(name, y_pred):
    print(f"Performance of {name}:")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred, average='macro'))
    print("Recall:", recall_score(y_true, y_pred, average='macro'))


evaluate_model("SVM Model", svm_pred)
evaluate_model("Decision Tree Model", tree_pred)
evaluate_model("LLM Model", llm_pred)
