import os
import warnings
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["USE_TF"] = "0"
warnings.filterwarnings("ignore")

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import joblib


DATA_PATH = "judge_training.csv"
BEHAVIOR_MODEL_PATH = "behavior_judge.joblib"
CONTENT_MODEL_PATH = "content_judge.joblib"


def main():
    print("=== Training behavior/content judges ===")

    # 1) Load data
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH, encoding="latin-1")

    # Basic sanity
    required_cols = {"text", "behavior_label", "content_label"}
    assert required_cols.issubset(df.columns), f"CSV must contain columns: {required_cols}"

    texts = df["text"].tolist()
    behavior_labels = df["behavior_label"].tolist()
    content_labels = df["content_label"].tolist()

    # Map string labels to integers
    behavior_map = {"refuse": 0, "comply": 1}
    content_map = {"safe": 0, "harmful": 1}

    y_behavior = [behavior_map[b] for b in behavior_labels]
    y_content = [content_map[c] for c in content_labels]

    # 2) Load embedding model
    print("Loading MiniLM embedding model...")
    emb_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print("Encoding texts...")
    X = emb_model.encode(texts)

    # 3) Split train/test
    X_train, X_test, yb_train, yb_test = train_test_split(
        X, y_behavior, test_size=0.2, random_state=42, stratify=y_behavior
    )
    _, _, yc_train, yc_test = train_test_split(
        X, y_content, test_size=0.2, random_state=42, stratify=y_content
    )

    # 4) Train behavior classifier
    print("\nTraining behavior classifier (refuse vs comply)...")
    behavior_clf = LogisticRegression(max_iter=1000)
    behavior_clf.fit(X_train, yb_train)
    yb_pred = behavior_clf.predict(X_test)
    print("\nBehavior classifier report:")
    print(classification_report(yb_test, yb_pred, target_names=["refuse", "comply"]))

    # 5) Train content classifier
    print("\nTraining content classifier (safe vs harmful)...")
    content_clf = LogisticRegression(max_iter=1000)
    content_clf.fit(X_train, yc_train)  # reuse X_train for simplicity
    yc_pred = content_clf.predict(X_test)
    print("\nContent classifier report:")
    print(classification_report(yc_test, yc_pred, target_names=["safe", "harmful"]))

    # 6) Save models + label maps
    print(f"\nSaving behavior judge to {BEHAVIOR_MODEL_PATH}...")
    joblib.dump({"model": behavior_clf, "label_map": {v: k for k, v in behavior_map.items()}}, BEHAVIOR_MODEL_PATH)

    print(f"Saving content judge to {CONTENT_MODEL_PATH}...")
    joblib.dump({"model": content_clf, "label_map": {v: k for k, v in content_map.items()}}, CONTENT_MODEL_PATH)

    print("\n=== Training completed ===")


if __name__ == "__main__":
    main()
