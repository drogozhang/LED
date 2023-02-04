import datasets
import numpy as np
import torch

"""Metrics for assessing baseline models."""

from sklearn import metrics


def xentropy(
        y_true: np.ndarray,
        y_pred: np.ndarray
) -> np.float64:
    """Return the xentropy of ``y_pred`` with respect to ``y_true``.

    Parameters
    ----------
    y_true : np.ndarray, required
        An ``n_samples`` by ``n_classes`` array for the class
        probabilities given to each sample.
    y_pred : np.ndarray, required
        An ``n_samples`` by ``n_classes`` array for the predicted class
        probabilities given to each sample.

    Returns
    -------
    np.float64
        The xentropy of ``y_pred` with respect to ``y_true``.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(- np.sum(np.log(y_pred ** y_true), axis=1))

METRICS = {
    'accuracy': (
        'accuracy',
        metrics.accuracy_score,
        {
            'greater_is_better': True,
            'needs_proba': False
        }
    ),
    'balanced_accuracy': (
        'balanced accuracy',
        metrics.balanced_accuracy_score,
        {
            'greater_is_better': True,
            'needs_proba': False
        }
    ),
    'precision_micro': (
        'precision (micro)',
        lambda y_true, y_pred: metrics.precision_score(
            y_true=y_true,
            y_pred=y_pred,
            average='micro'),
        {
            'greater_is_better': True,
            'needs_proba': False
        }
    ),
    'recall_micro': (
        'recall (micro)',
        lambda y_true, y_pred: metrics.recall_score(
            y_true=y_true,
            y_pred=y_pred,
            average='micro'),
        {
            'greater_is_better': True,
            'needs_proba': False
        }
    ),
    'f1_micro': (
        'f1 (micro)',
        lambda y_true, y_pred: metrics.f1_score(
            y_true=y_true,
            y_pred=y_pred,
            average='micro'),
        {
            'greater_is_better': True,
            'needs_proba': False
        }
    ),
    'precision_macro': (
        'precision (macro)',
        lambda y_true, y_pred: metrics.precision_score(
            y_true=y_true,
            y_pred=y_pred,
            average='macro'),
        {
            'greater_is_better': True,
            'needs_proba': False
        }
    ),
    'recall_macro': (
        'recall (macro)',
        lambda y_true, y_pred: metrics.recall_score(
            y_true=y_true,
            y_pred=y_pred,
            average='macro'),
        {
            'greater_is_better': True,
            'needs_proba': False
        }
    ),
    'f1_macro': (
        'f1 (macro)',
        lambda y_true, y_pred: metrics.f1_score(
            y_true=y_true,
            y_pred=y_pred,
            average='macro'),
        {
            'greater_is_better': True,
            'needs_proba': False
        }
    ),
    'f1_weighted': (
        'f1 (weighted)',
        lambda y_true, y_pred: metrics.f1_score(
            y_true=y_true,
            y_pred=y_pred,
            average='weighted'),
        {
            'greater_is_better': True,
            'needs_proba': False
        }
    ),
    'log_loss': (
        'log loss',
        lambda y_true, y_pred: metrics.log_loss(
            y_true=y_true,
            y_pred=y_pred,
            eps=1e-9),
        {
            'greater_is_better': False,
            'needs_proba': True
        }
    ),
    'matthews_corrcoef': (
        'matthews correlation coefficient',
        metrics.matthews_corrcoef,
        {
            'greater_is_better': True,
            'needs_proba': False
        }
    ),
    # N.B., do not include a key for "xentropy" or "calibrated_xentropy"
    # in this dictionary. Those keys are reserved for the cross-entropy
    # between the predicted probabilities and the dataset's label
    # scores, which is computed in the scruples.scripts.analyze.*.predictions
    # scripts.
    'xentropy': (
        'cross entropy',
        xentropy,
        {
            'greater_is_better': False,
            'needs_proba': True,
            "both_proba": True
        }
    ),
    'conf_mat': (
        'confusion matrix',
        metrics.confusion_matrix,
        {
            'greater_is_better': None,
            'needs_proba': False,
        }
    ),
}
"""A dictionary defining the important metrics to assess baselines.

The dictionary maps metric names to ``(name, metric, scorer_kwargs)``
tuples. ``name`` is the name of the metric, while ``metric`` is a
function for computing it, and ``scorer_kwargs`` is a dictionary
containing two keys: ``"greater_is_better"``, a boolean defining whether
or not higher values are better, and ``"needs_proba"``, a boolean
defining whether to pass the predicted labels or the predicted
probabilities.
"""


class MetricSoftClassification(datasets.Metric):
    def _info(self) -> datasets.MetricInfo:
        return datasets.MetricInfo(
            description="",
            citation="",
            features=datasets.Features(
                {
                    "predictions": datasets.Sequence(datasets.Value("float")),  # logits [n_example, n_class]
                    "references": datasets.Sequence(datasets.Value("float")),  # probs [n_example, n_class]
                }
            ),
        )

    def calculate_metrics(self, predictions=None, references=None, prefix=""):
        pred_probs = np.exp(predictions)
        pred_probs = pred_probs / np.sum(pred_probs, axis=-1, keepdims=True)
        pred_label = np.argmax(predictions, axis=-1)

        gold_probs = np.array(references, dtype="float")
        gold_label = np.argmax(references, axis=-1)

        out_dict = dict()
        for metric_name, metric_meta in METRICS.items():
            if metric_meta[2]["needs_proba"]:
                if "both_proba" in metric_meta[2] and metric_meta[2]["both_proba"]:
                    out_dict[prefix+metric_name] = metric_meta[1](gold_probs, pred_probs)
                else:
                    out_dict[prefix+metric_name] = metric_meta[1](gold_label, pred_probs)
            else:
                out_dict[prefix+metric_name] = metric_meta[1](gold_label, pred_label)
        return out_dict


    def _compute(self, *, predictions=None, references=None, **kwargs):
        predictions = np.array(predictions, dtype="float")
        references = np.array(references, dtype="float")

        out_dict = self.calculate_metrics(predictions, references)
        return out_dict

