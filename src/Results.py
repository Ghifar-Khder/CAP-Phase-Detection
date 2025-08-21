import tensorflow as tf
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Define paths
# ensure to change paths for each health condition
MODEL_PATH = r"models\healthy_model.keras"
TRAIN_SET_PATH = r"data\splitted-data\healthy\healthy_train_set.csv"
VAL_SET_PATH = r"data\splitted-data\healthy\healthy_validation_set.csv"
TEST_SET_PATH = r"data\splitted-data\healthy\healthy_test_set.csv"
PLOTS_DIR = r"results"
os.makedirs(PLOTS_DIR, exist_ok=True)

# Load the trained model
model = tf.keras.models.load_model(MODEL_PATH)

def load_dataset(filepath):
    data = pd.read_csv(filepath).values
    x_data = data[:, :-1]
    y_data = data[:, -1]
    x_data = x_data.reshape(x_data.shape[0], x_data.shape[1], 1)
    return x_data, y_data

# Load datasets
x_train, y_train = load_dataset(TRAIN_SET_PATH)
x_val, y_val = load_dataset(VAL_SET_PATH)
x_test, y_test = load_dataset(TEST_SET_PATH)

# Function to compute precision, recall, and f1-score from confusion matrix
def calculate_metrics(conf_matrix):
    tn, fp, fn, tp = conf_matrix.ravel()
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0.0
    return precision * 100, recall * 100, f1_score * 100  # Convert to percentage

# Function to evaluate the model
def evaluate_model(x, y, set_name):
    evaluation_results = model.evaluate(x, y, verbose=2)
    loss, accuracy = evaluation_results[:2]  # Extract loss and accuracy
    predictions = (model.predict(x) > 0.5).astype(int)
    conf_matrix = confusion_matrix(y, predictions)
    precision, recall, f1_score = calculate_metrics(conf_matrix)
    y_probs = model.predict(x)
    fpr, tpr, _ = roc_curve(y, y_probs)
    roc_auc = auc(fpr, tpr)
    
    # Save confusion matrix
    """plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {set_name}')
    plt.savefig(os.path.join(PLOTS_DIR, f"confusion_matrix_{set_name}.png"))
    plt.close()"""
    
    # Save AUC-ROC curve
    """plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc * 100:.1f}%')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {set_name}')
    plt.legend()
    plt.savefig(os.path.join(PLOTS_DIR, f"roc_curve_{set_name}.png"))
    plt.close()"""
    
    return {
        "Dataset": set_name,
        "Accuracy": f"{accuracy * 100:.1f}%",
        "Precision": f"{precision:.1f}%",
        "Recall": f"{recall:.1f}%",
        "F1-score": f"{f1_score:.1f}%"
    }

# Collect evaluation results
evaluation_results = [
    evaluate_model(x_train, y_train, "Train Set"),
    evaluate_model(x_val, y_val, "Validation Set"),
    evaluate_model(x_test, y_test, "Test Set")
]

df_results = pd.DataFrame(evaluation_results)

df_results.to_csv(os.path.join(PLOTS_DIR, "healthy-results.csv"), index=False) # change name according to the health condition

# Save metrics table as an image
"""fig, ax = plt.subplots(figsize=(6, 3))
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=df_results.values, colLabels=df_results.columns, cellLoc='center', loc='center', bbox=[0, 0, 1, 1])

# Apply background colors relevant to values
for i in range(1, len(df_results.columns)):
    for j in range(len(df_results)):
        value = float(df_results.iloc[j, i].replace('%', '')) / 100  # Convert back to float for colormap
        color = plt.cm.Blues(value)  # Use blue gradient color mapping
        table[(j + 1, i)].set_facecolor(color)

plt.title("Overall Metrics Table")
plt.savefig(os.path.join(PLOTS_DIR, "overall_metrics_table.png")) # change name according to the health condition
plt.close()

print(f"Evaluation results saved in: {PLOTS_DIR}")
"""