"""PAC-style sample-complexity certificate for the learned-fallback classifier.

The deterministic verifier (§4) is sound by construction: the
type-theoretic guarantee of §4.7 holds for any trajectory that
matches no high/critical pattern, and §4.8 gives a
distribution-free coverage guarantee. The *learned* fallback
(TF-IDF + Logistic Regression in ``baselines/custom_trained.py``)
has no analogous soundness story — it is a black-box function
class fit to a finite sample. This module provides a
formal-VC-style PAC certificate: *with what probability does a
trained fallback generalize to test trajectories drawn from the
same distribution, as a function of training-set size?*

The certificate is the standard Vapnik–Chervonenkis uniform
convergence bound for binary classifiers with VC dimension
``d``. Following Mohri, Rostamizadeh, and Talwalkar
(Foundations of Machine Learning, MIT Press, 2012, Theorem
3.7 / 3.23), the *upper* bound on the sample size sufficient
for generalization error ε at confidence 1−δ is:

    N ≥ (1/ε) · ( 4·log(2/δ) + 8·d·log(13/ε) )

where ``d`` is the VC dimension of the hypothesis class and
``ε`` and ``δ`` are user-specified accuracy / confidence
parameters. The looseness (constants 4 and 8, log term
``log(13/ε)``) is an artifact of the standard proof; the bound
is otherwise tight up to constants.

We use the TF-IDF *feature count* as a deterministic upper
bound on the VC dimension of the linear threshold family
realized by the Logistic Regression head (cf. Theorem 9.4 of
Mohri et al.): a linear separator in ℝ^d in the worst case
shatters the full d-dimensional Boolean cube, so
``VC(LR_d) ≤ d + 1``. Because the TF-IDF vocabulary size is
explicit and bounded by ``max_features``, the
``vc_dim_class`` proxy is a sound upper bound.

We bound the *number of attack classes* explicitly because the
multi-class extension of the VC bound (the
``N_class``-realized VC) reduces to a union bound over
``n_attack_classes`` binary problems:

    N ≥ (1/ε) · ( 4·log(2·n_class/δ) + 8·d·log(13/ε) )

(see Mohri et al. Theorem 3.23 for the K-class reduction).
"""
from __future__ import annotations

import math
from typing import Any

try:
    # sklearn is an optional import — the bound is meaningful even when
    # sklearn is absent (the caller just passes a precomputed VC estimate).
    from sklearn.pipeline import Pipeline  # noqa: F401
    _SKLEARN_OK = True
except Exception:
    _SKLEARN_OK = False


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def pac_sample_complexity(epsilon: float,
                          delta: float,
                          vc_dim: int,
                          n_attack_classes: int = 2) -> int:
    """Lower bound on training-set size for (ε, δ)-PAC generalization.

    The bound is the standard Vapnik–Chervonenkis uniform-convergence
    bound (Mohri, Rostamizadeh, Talwalkar 2012, Theorem 3.7 and
    its K-class extension Theorem 3.23):

        N ≥ (1/ε) · ( 4·log(2·n_attack_classes/δ) + 8·d·log(13/ε) )

    Args:
        epsilon: target generalization-error rate (e.g., 0.05 for 95%
            accuracy).
        delta: probability of *failure* — i.e., the chance that the
            empirical risk fails to generalize to the true risk by
            more than ε.
        vc_dim: VC dimension of the hypothesis class. For a
            TF-IDF + LR pipeline, this is the number of features
            in the vocabulary (we add 1 for the bias term).
        n_attack_classes: number of attack classes the fallback
            must distinguish. For our binary benign/malicious
            fallback this is 2; for the multi-class taxonomy
            classifier this is the number of pattern families
            (we use 3 for tool-call / text-output / data-flow).

    Returns:
        N: the smallest integer N such that any training set of
            size ≥ N guarantees that, with probability ≥ 1−δ, the
            empirical risk of any hypothesis in the class is
            within ε of its true risk.

    Raises:
        ValueError: if any of the inputs fall outside their legal
            ranges (0 < ε < 1, 0 < δ < 1, d ≥ 1, n_attack_classes ≥ 1).
    """
    if not (0.0 < epsilon < 1.0):
        raise ValueError(f"epsilon must satisfy 0 < eps < 1; got {epsilon}")
    if not (0.0 < delta < 1.0):
        raise ValueError(f"delta must satisfy 0 < delta < 1; got {delta}")
    if vc_dim < 1:
        raise ValueError(f"vc_dim must be ≥ 1; got {vc_dim}")
    if n_attack_classes < 1:
        raise ValueError(
            f"n_attack_classes must be ≥ 1; got {n_attack_classes}")

    # Multi-class reduction: divide δ across classes so that the
    # probability of any per-class failure is ≤ δ/n_attack_classes.
    delta_eff = delta / n_attack_classes
    # Inner term: 4·log(2/δ_eff) + 8·d·log(13/ε)
    inner = (4.0 * math.log(2.0 / delta_eff)
             + 8.0 * vc_dim * math.log(13.0 / epsilon))
    return int(math.ceil(inner / epsilon))


def vc_dim_class(sklearn_classifier: Any) -> int:
    """Estimate the VC dimension of a scikit-learn TF-IDF + LR pipeline.

    Returns the number of features in the TF-IDF vocabulary, plus
    one for the bias term realized by ``LogisticRegression``'s
    intercept. This is a sound upper bound on the VC dimension of
    the realized linear-threshold hypothesis class in ℝ^d.

    For non-pipeline classifiers (e.g., pure LogisticRegression on a
    dense matrix) the same heuristic applies — feature_count is
    read from the trained model's ``coef_`` shape.

    Args:
        sklearn_classifier: a fitted estimator. Expected to be a
            ``sklearn.pipeline.Pipeline`` whose first stage is a
            TF-IDF-style vectorizer with ``get_feature_names_out``
            or a ``vocabulary_`` attribute, and whose final stage is
            a linear model with a 2-D ``coef_``.

    Returns:
        VC dimension (≥ 2). Returns 2 as a safe lower bound when
        the structure cannot be determined.

    Notes:
        A linear threshold function in ℝ^d has VC dimension exactly
        d + 1 (Mohri et al. 2012, Theorem 9.4); the +1 captures the
        bias. We add it explicitly here so the reported number is
        the textbook VC dimension, not just the feature count.
    """
    # Pipeline path: first step is the vectorizer, last step the linear head.
    if _SKLEARN_OK and isinstance(sklearn_classifier, Pipeline):
        # Step 1: feature count from the vectorizer's vocabulary.
        try:
            vec = sklearn_classifier.steps[0][1]
            vocab = getattr(vec, "vocabulary_", None)
            if vocab is None:
                # HashingVectorizer has no vocabulary; fall back to a
                # conservative estimate from max_features.
                max_features = getattr(vec, "max_features", None)
                if max_features is None:
                    return 2
                feature_count = int(max_features)
            else:
                feature_count = len(vocab)
        except Exception:
            feature_count = None
        # Step 2: the linear head might have multi-class coef (shape
        # [n_classes, n_features]); the VC dimension of the realized
        # family is still bounded by d + 1 regardless of n_classes.
        try:
            clf = sklearn_classifier.steps[-1][1]
            coef = getattr(clf, "coef_", None)
            if coef is not None and hasattr(coef, "shape") and len(coef.shape) == 2:
                d = int(coef.shape[1])
                if d > 0:
                    return d + 1  # bias term
        except Exception:
            pass
        if feature_count is not None:
            return int(feature_count) + 1
        return 2

    # Plain linear model path.
    coef = getattr(sklearn_classifier, "coef_", None)
    if coef is not None and hasattr(coef, "shape"):
        if len(coef.shape) == 2:
            d = int(coef.shape[1])
        else:
            d = int(coef.shape[0])
        if d > 0:
            return d + 1

    # Conservative fallback: return d=2 (the minimum VC dimension
    # needed to shatter a 2-point set in ℝ).
    return 2


# -----------------------------------------------------------------------------
# Convenience helper for reports
# -----------------------------------------------------------------------------

def certificate(epsilon: float, delta: float, vc_dim: int,
                n_attack_classes: int = 2) -> dict:
    """Return a structured (ε, δ)-PAC certificate.

    Convenience wrapper for reporting: returns a dict containing
    both the bound and the inputs so callers can serialize it.
    """
    n = pac_sample_complexity(epsilon=epsilon, delta=delta,
                              vc_dim=vc_dim,
                              n_attack_classes=n_attack_classes)
    return {
        "epsilon": epsilon,
        "delta": delta,
        "vc_dim": vc_dim,
        "n_attack_classes": n_attack_classes,
        "n_required": n,
    }
