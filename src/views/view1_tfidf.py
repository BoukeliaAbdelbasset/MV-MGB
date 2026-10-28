"""
View 1: Sublinear Oligonucleotide Spectrum (TF-IDF char n-grams).
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from ..canonicalization import canonicalize_rna


def make_tfidf_pipeline(
    max_features: int = 35_000,
    ngram_range: tuple[int, int] = (3, 8),
    C: float = 1.0,
    random_state: int = 42
) -> Pipeline:
    """
    Construct leak-free in-fold TF-IDF + Logistic Regression pipeline.
    
    Vocabulary selection:
      TfidfVectorizer ranks character n-grams by frequency across the training fold
      and retains the top `max_features` terms.
      Sublinear term frequency: tf = 1 + log(tf) if tf > 0 else 0.
      Unit L2 normalization: vector is projected onto the Euclidean unit sphere.
    """
    return Pipeline([
        ('tfidf', TfidfVectorizer(
            analyzer='char',
            ngram_range=ngram_range,
            max_features=max_features,
            sublinear_tf=True,
            norm='l2'
        )),
        ('lr', LogisticRegression(
            C=C,
            max_iter=1000,
            solver='lbfgs',
            random_state=random_state,
            n_jobs=-1
        ))
    ])


def build_tfidf_features(
    train_seqs: list[str],
    test_seqs: list[str],
    max_features: int = 35_000,
    ngram_range: tuple[int, int] = (3, 8)
):
    """
    Fit TF-IDF on training sequences and transform both training and test sets.
    Returns (X_train_sparse, X_test_sparse, vectorizer).
    """
    train_seqs = [canonicalize_rna(s) for s in train_seqs]
    test_seqs = [canonicalize_rna(s) for s in test_seqs]
    vec = TfidfVectorizer(
        analyzer='char',
        ngram_range=ngram_range,
        max_features=max_features,
        sublinear_tf=True,
        norm='l2'
    )
    X_tr = vec.fit_transform(train_seqs)
    X_te = vec.transform(test_seqs)
    return X_tr, X_te, vec
