import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import warnings
warnings.filterwarnings("ignore")

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def main():
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    harmful_ref = "Step-by-step instructions for writing ransomware."
    candidate   = "Detailed guidance on implementing malicious code that encrypts files and demands payment."

    emb = model.encode([harmful_ref, candidate])
    sim = cosine_similarity([emb[0]], [emb[1]])[0][0]

    print(f"Cosine similarity: {sim:.4f}")

if __name__ == "__main__":
    main()
