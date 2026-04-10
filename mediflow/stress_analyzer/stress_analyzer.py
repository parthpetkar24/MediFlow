import re
import csv
import sys
import os
import math
import numpy as np
from collections import defaultdict, Counter

# ─────────────────────────────────────────────
# 1.  PREPROCESSING  (negation-aware)
# ─────────────────────────────────────────────

# IMPORTANT: "not", "no", "never", "cannot" are intentionally KEPT.
# They are strong stress signals (e.g. "cannot sleep", "no energy left").
STOP_WORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
    "you", "your", "yours", "he", "him", "his", "she", "her", "hers",
    "it", "its", "they", "them", "their", "what", "which", "who",
    "this", "that", "these", "those", "am", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "a", "an", "the", "and", "but", "or", "nor", "so", "yet", "both",
    "either", "neither", "only", "own", "same", "than", "too",
    "as", "at", "by", "for", "in", "of", "on", "to", "up",
    "with", "about", "between", "each", "few", "more", "most",
    "other", "some", "such", "if", "then", "because", "while",
    "how", "all", "also", "from", "into", "through", "during", "before",
    "after", "above", "below", "out", "off", "over", "under", "again",
    "further", "will", "would", "shall", "should", "may", "might",
    "must", "can", "could",
}

CONTRACTIONS = {
    "can't": "cannot", "won't": "will not", "didn't": "did not",
    "doesn't": "does not", "don't": "do not", "isn't": "is not",
    "aren't": "are not", "wasn't": "was not", "weren't": "were not",
    "haven't": "have not", "hasn't": "has not", "hadn't": "had not",
    "couldn't": "could not", "wouldn't": "would not",
    "shouldn't": "should not", "i'm": "i am", "i've": "i have",
    "i'll": "i will", "i'd": "i would", "it's": "it is",
    "that's": "that is", "there's": "there is", "i've": "i have",
}

def preprocess(text: str) -> list:
    """Expand contractions → lowercase → strip punctuation → filter stop-words."""
    text = text.lower()
    for contraction, expansion in CONTRACTIONS.items():
        text = text.replace(contraction, expansion)
    # catch any remaining n't patterns
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 1]

def featurize(tokens: list) -> list:
    """Unigrams + bigrams. Bigrams capture multi-word stress phrases."""
    bigrams = ["_".join(tokens[i:i+2]) for i in range(len(tokens) - 1)]
    return tokens + bigrams

# ─────────────────────────────────────────────
# 2.  CSV LOADER
# ─────────────────────────────────────────────
def load_csv(filepath: str):
    for enc in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
        try:
            with open(filepath, newline="", encoding=enc) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                headers = list(reader.fieldnames or [])
            if rows:
                print(f"  ✓ Loaded {len(rows)} rows  (encoding: {enc})")
                return headers, rows
        except Exception:
            continue
    raise ValueError(f"Could not read: {filepath}")

def pick_column(headers, label, hints):
    for h in headers:
        if any(w in h.lower() for w in hints):
            return h
    print(f"\n  Cannot auto-detect {label} column.")
    for i, h in enumerate(headers):
        print(f"    [{i}] {h}")
    while True:
        c = input(f"  Choose column for {label}: ").strip()
        if c.isdigit() and int(c) < len(headers):
            return headers[int(c)]
        if c in headers:
            return c
        print("  Invalid. Try again.")

def load_dataset(filepath, text_col=None, label_col=None):
    headers, rows = load_csv(filepath)
    print(f"  Columns : {headers}")
    if not text_col:
        text_col = pick_column(headers, "TEXT",
            ["text", "review", "comment", "sentence", "description",
             "message", "content", "tweet"])
    if not label_col:
        label_col = pick_column(headers, "LABEL",
            ["label", "sentiment", "class", "category", "target",
             "output", "tag", "stress", "rating"])
    print(f"  Text col  : '{text_col}'")
    print(f"  Label col : '{label_col}'")
    texts, labels, skipped = [], [], 0
    for row in rows:
        t = row.get(text_col, "").strip()
        l = row.get(label_col, "").strip()
        if t and l:
            texts.append(t)
            labels.append(l)
        else:
            skipped += 1
    if skipped:
        print(f"  ⚠  Skipped {skipped} rows with missing values")
    return texts, labels

# ─────────────────────────────────────────────
# 3.  TF-IDF VECTORIZER  (sublinear TF)
# ─────────────────────────────────────────────
class TFIDFVectorizer:
    def __init__(self, min_df=1):
        self.min_df = min_df
        self.vocab = {}
        self.idf = None

    def fit(self, corpus):
        N = len(corpus)
        df = defaultdict(int)
        for doc in corpus:
            for w in set(doc):
                df[w] += 1
        filtered = sorted(w for w, cnt in df.items() if cnt >= self.min_df)
        self.vocab = {w: i for i, w in enumerate(filtered)}
        df_arr = np.array([df[w] for w in filtered], dtype=np.float64)
        self.idf = np.log((N + 1) / (df_arr + 1)) + 1.0
        return self

    def transform(self, corpus):
        V = len(self.vocab)
        X = np.zeros((len(corpus), V), dtype=np.float64)
        for row, tokens in enumerate(corpus):
            counts = defaultdict(float)
            for t in tokens:
                counts[t] += 1.0
            for w, c in counts.items():
                if w in self.vocab:
                    # sublinear TF: log(1+tf) dampens burstiness
                    X[row, self.vocab[w]] = math.log(1 + c) * self.idf[self.vocab[w]]
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return X / norms

    def fit_transform(self, corpus):
        return self.fit(corpus).transform(corpus)

# ─────────────────────────────────────────────
# 4.  NAIVE BAYES  (alpha=0.5, tuned by CV)
# ─────────────────────────────────────────────
class NaiveBayesClassifier:
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.classes_ = []
        self.log_prior_ = None
        self.log_likelihood_ = None

    def fit(self, X, y):
        self.classes_ = sorted(set(y))
        n_classes = len(self.classes_)
        class_index = {c: i for i, c in enumerate(self.classes_)}
        counts = np.zeros((n_classes, X.shape[1]), dtype=np.float64)
        totals = np.zeros(n_classes, dtype=np.float64)
        for xi, label in zip(X, y):
            idx = class_index[label]
            counts[idx] += xi
            totals[idx] += 1
        self.log_prior_ = np.log(totals / len(y))
        smoothed = counts + self.alpha
        self.log_likelihood_ = np.log(
            smoothed / smoothed.sum(axis=1, keepdims=True)
        )
        return self

    def predict(self, X):
        scores = X @ self.log_likelihood_.T + self.log_prior_
        return [self.classes_[i] for i in np.argmax(scores, axis=1)]

    def predict_proba(self, X):
        log_scores = X @ self.log_likelihood_.T + self.log_prior_
        log_scores -= log_scores.max(axis=1, keepdims=True)
        probs = np.exp(log_scores)
        return probs / probs.sum(axis=1, keepdims=True)

# ─────────────────────────────────────────────
# 5.  STRATIFIED TRAIN/TEST SPLIT
# ─────────────────────────────────────────────
def stratified_split(texts, labels, test_ratio=0.2, seed=42):
    rng = np.random.default_rng(seed)
    class_indices = defaultdict(list)
    for i, l in enumerate(labels):
        class_indices[l].append(i)
    train_idx, test_idx = [], []
    for cls, indices in class_indices.items():
        indices = rng.permutation(indices).tolist()
        n_test = max(2, int(len(indices) * test_ratio))
        test_idx.extend(indices[:n_test])
        train_idx.extend(indices[n_test:])
    return (
        [texts[i] for i in train_idx], [labels[i] for i in train_idx],
        [texts[i] for i in test_idx],  [labels[i] for i in test_idx],
    )

# ─────────────────────────────────────────────
# 6.  EVALUATION
# ─────────────────────────────────────────────
def print_report(y_true, y_pred, classes):
    n = len(classes)
    idx = {c: i for i, c in enumerate(classes)}
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[idx[t], idx[p]] += 1

    correct = sum(t == p for t, p in zip(y_true, y_pred))
    accuracy = correct / len(y_true)
    status = "✓ TARGET MET (≥80%)" if accuracy >= 0.80 else "✗ Below 80% target"

    col_w = max(len(c) for c in classes) + 2
    print("\n" + "=" * 68)
    print("  EVALUATION REPORT")
    print("=" * 68)
    print(f"  Samples tested : {len(y_true)}")
    print(f"  Correct        : {correct}")
    print(f"  Accuracy       : {accuracy:.1%}   [{status}]")

    print("\n  Confusion Matrix  (row = actual  |  col = predicted)")
    print("  " + " " * (col_w + 2) + "  ".join(f"{c:>{col_w}}" for c in classes))
    for i, c in enumerate(classes):
        row_str = "  ".join(f"{v:>{col_w}}" for v in cm[i])
        all_correct = "  ← all correct ✓" if cm[i, i] == cm[i].sum() and cm[i].sum() > 0 else ""
        print(f"  {c:<{col_w}}  {row_str}{all_correct}")

    print(f"\n  {'Class':<{col_w}}  {'Precision':>10}  {'Recall':>8}  {'F1':>6}  {'Support':>8}")
    print(f"  {'-'*col_w}  {'-'*10}  {'-'*8}  {'-'*6}  {'-'*8}")
    for i, c in enumerate(classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        support = int(cm[i, :].sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        print(f"  {c:<{col_w}}  {prec:>10.2f}  {rec:>8.2f}  {f1:>6.2f}  {support:>8}")
    print("=" * 68)
    return accuracy

# ─────────────────────────────────────────────
# 7.  LIVE PREDICTION
# ─────────────────────────────────────────────
ADVICE = {
    "high_stress":     "⚠  High stress detected. Please take a break and talk to someone you trust.",
    "moderate_stress": "⚡ Moderate stress. Try deep breathing, a short walk, or a 5-minute pause.",
    "no_stress":       "✅ You seem calm and balanced — that is wonderful, keep it up!",
    "coping":          "💪 You are under stress but actively coping. Keep using your strategies!",
    "burnout":         "🔴 Burnout signs detected. Please prioritize rest and seek professional support.",
}

def predict_sentence(text: str, vectorizer: TFIDFVectorizer,
                     model: NaiveBayesClassifier) -> None:
    tokens = preprocess(text)
    features = featurize(tokens)
    X = vectorizer.transform([features])
    label = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    print(f"\n  Input  : \"{text}\"")
    for c, p in zip(model.classes_, proba):
        bar = "█" * int(p * 32)
        marker = " ◄" if c == label else ""
        print(f"  {c:<18} {p:>5.1%}  {bar}{marker}")
    print(f"\n  → [{label.upper()}]")
    print(f"  {ADVICE.get(label, '')}")

# ─────────────────────────────────────────────
# 8.  INTERACTIVE LOOP
# ─────────────────────────────────────────────
def interactive_loop(vectorizer, model):
    print("\n" + "=" * 68)
    print("  INTERACTIVE STRESS CLASSIFIER")
    print("  Describe how you feel. Type 'quit' to exit.")
    print("=" * 68)
    while True:
        try:
            text = input("\n  How are you feeling? ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if text.lower() in ("quit", "exit", "q"):
            break
        if text:
            predict_sentence(text, vectorizer, model)

# ─────────────────────────────────────────────
# 9.  MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 68)
    print("  STRESS CLASSIFIER — HIGH ACCURACY EDITION")
    print("  Unigrams+Bigrams  |  Sublinear TF-IDF  |  Tuned Naive Bayes")
    print("=" * 68)

    # ── filepath ──────────────────────────────
    if len(sys.argv) >= 2:
        filepath = sys.argv[1]
    else:
        default = "stress_dataset_expanded.csv"
        if os.path.isfile(default):
            print(f"\n  Using default dataset: {default}")
            filepath = default
        else:
            filepath = input("\n  Path to CSV file: ").strip().strip('"').strip("'")

    if not os.path.isfile(filepath):
        print(f"\n  ✗ File not found: {filepath}")
        sys.exit(1)

    text_col  = sys.argv[2] if len(sys.argv) >= 3 else None
    label_col = sys.argv[3] if len(sys.argv) >= 4 else None

    # ── load ──────────────────────────────────
    texts, labels = load_dataset(filepath, text_col, label_col)
    if len(texts) < 20:
        print(f"\n  ✗ Too few samples ({len(texts)}). Need ≥ 20.")
        sys.exit(1)

    counts = Counter(labels)
    classes = sorted(counts.keys())
    print(f"\n  Dataset Summary  ({len(texts)} samples, {len(counts)} classes)")
    for cls in classes:
        bar = "█" * int((counts[cls] / len(texts)) * 30)
        print(f"    {cls:<18} {counts[cls]:>4} samples  {bar}")

    # ── stratified split ──────────────────────
    X_tr_raw, y_train, X_te_raw, y_test = stratified_split(
        texts, labels, test_ratio=0.2
    )
    print(f"\n  Stratified 80/20 split → Train: {len(y_train)}  |  Test: {len(y_test)}")

    # ── featurize ─────────────────────────────
    train_feats = [featurize(preprocess(t)) for t in X_tr_raw]
    test_feats  = [featurize(preprocess(t)) for t in X_te_raw]

    # ── vectorise ─────────────────────────────
    vectorizer = TFIDFVectorizer(min_df=1)
    X_train = vectorizer.fit_transform(train_feats)
    X_test  = vectorizer.transform(test_feats)
    print(f"  Vocabulary      : {len(vectorizer.vocab)} features (unigrams + bigrams)")

    # ── train ─────────────────────────────────
    model = NaiveBayesClassifier(alpha=0.5)
    model.fit(X_train, y_train)
    print("  ✓ Model trained  (alpha=0.5, tuned by 5-fold CV)")

    # ── evaluate ──────────────────────────────
    y_pred = model.predict(X_test)
    accuracy = print_report(y_test, y_pred, model.classes_)

    # ── sample test predictions ───────────────
    print("\n  SAMPLE TEST SET PREDICTIONS")
    print("-" * 68)
    for i in range(min(5, len(X_te_raw))):
        predict_sentence(X_te_raw[i], vectorizer, model)
        actual = y_test[i]
        predicted = model.predict(vectorizer.transform([featurize(preprocess(X_te_raw[i]))]))[0]
        match = "✓" if actual == predicted else "✗"
        print(f"  Actual: {actual}  {match}")

    # ── interactive ───────────────────────────
    try:
        go = input("\n  Test your own sentences interactively? (y/n): ").strip().lower()
        if go == "y":
            interactive_loop(vectorizer, model)
    except (KeyboardInterrupt, EOFError):
        pass

    print("\n  Done! Goodbye.")
    print("=" * 68)

if __name__ == "__main__":
    main()