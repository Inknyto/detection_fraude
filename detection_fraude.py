#  ~/Documents/esmt/test_logiciel/evaluation_formative_1/integration_gitlab_ci/detection_de_fraude.py :08 Jan at 05:41:28 PM
import sys
sys.path.append('/home/nyto/python-venv/lib/python3.13/site-packages')

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    scaler = RobustScaler()
    df['Time_Scaled'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))
    df['Amount_Scaled'] = scaler.fit_transform(
        df['Amount'].values.reshape(-1, 1))
    df.drop(columns=['Time', 'Amount'], inplace=True, axis=1)
    return df


def split_features_target(df):
    X = df.drop(columns=['Class'])
    y = df['Class']
    return X, y


def prepare_train_test_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


def apply_smote(X_train, y_train, random_state=42):
    sm = SMOTE(random_state=random_state)
    return sm.fit_resample(X_train, y_train)


def train_model(X_train, y_train):
    model = RandomForestClassifier(criterion='entropy', random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc_roc': roc_auc_score(y_test, y_pred)
    }
    return metrics


def plot_roc_curve(model, X_test, y_test):
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc_roc = roc_auc_score(y_test, model.predict(X_test))

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (area = {auc_roc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()


def main():
    # Pipeline execution
    df = load_and_preprocess_data('../creditcard.csv')
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = prepare_train_test_data(X, y)
    X_train_resampled, y_train_resampled = apply_smote(X_train, y_train)
    model = train_model(X_train_resampled, y_train_resampled)
    metrics = evaluate_model(model, X_test, y_test)

    print("\nTest Set Performance:")
    print("Précision:", metrics['accuracy'])
    print("Rappel:", metrics['recall'])
    print("F1-Score:", metrics['f1'])
    print("AUC-ROC:", metrics['auc_roc'])

    plot_roc_curve(model, X_test, y_test)


if __name__ == "__main__":
    main()
