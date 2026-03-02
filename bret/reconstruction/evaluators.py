"""
Evaluation metrics for reconstruction quality.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)
import logging

logger = logging.getLogger(__name__)

_2CLASS_LABELS = ["house", "face"]
_3CLASS_LABELS = ["house", "face", "mixed"]


class ReconstructionEvaluator:
    """
    Compute performance metrics for percept reconstruction.

    Ground truth is ``ReportedPercept`` (from perceptData CSV joined by
    :func:`~bret.io.loaders.join_reported_percepts`).  The primary prediction
    column is ``InferredPerceptAggregated`` (section-mode, ~1.8 s epochs).

    Two metric sets are always reported:
    - **2-class**: rows where ``ReportedPercept == 'mixed'`` are excluded so
      only house/face confusion is measured.
    - **3-class**: all rows kept, mixed treated as a third class.
    """

    def __init__(self):
        logger.info("Initialized ReconstructionEvaluator")

    # ── low-level metric helper ───────────────────────────────────────────────

    def evaluate(
        self,
        y_true: pd.Series,
        y_pred: pd.Series,
        exclude_mixed: bool = False,
    ) -> dict:
        """
        Compute accuracy, F1 (macro + weighted), MCC, and per-class metrics.

        Args:
            y_true: Ground-truth labels (``ReportedPercept``).
            y_pred: Predicted labels (e.g. ``InferredPerceptAggregated``).
            exclude_mixed: When True, drop rows where ``y_true == 'mixed'``
                before scoring (use for 2-class evaluation).

        Returns:
            Dict with keys: ``n_samples``, ``accuracy``, ``f1_macro``,
            ``f1_weighted``, ``mcc``, ``per_class`` (dict of
            label → {precision, recall, f1}).
            Returns a dict of NaNs when fewer than 2 samples remain.
        """
        mask = y_true.notna() & y_pred.notna()
        if exclude_mixed:
            mask &= y_true != "mixed"

        yt = y_true[mask].astype(str)
        yp = y_pred[mask].astype(str)
        n = len(yt)

        nan_result = {
            "n_samples": n,
            "accuracy": np.nan,
            "f1_macro": np.nan,
            "f1_weighted": np.nan,
            "mcc": np.nan,
            "per_class": {},
        }
        if n < 2:
            return nan_result

        labels = sorted(yt.unique().tolist())
        if len(labels) < 2:
            return nan_result

        prec, rec, f1, _ = precision_recall_fscore_support(
            yt, yp, labels=labels, average=None, zero_division=0
        )
        per_class = {
            lbl: {"precision": float(p), "recall": float(r), "f1": float(f)}
            for lbl, p, r, f in zip(labels, prec, rec, f1)
        }

        return {
            "n_samples": n,
            "accuracy": float(accuracy_score(yt, yp)),
            "f1_macro": float(f1_score(yt, yp, average="macro", zero_division=0)),
            "f1_weighted": float(f1_score(yt, yp, average="weighted", zero_division=0)),
            "mcc": float(matthews_corrcoef(yt, yp)),
            "per_class": per_class,
        }

    # ── data prep ────────────────────────────────────────────────────────────

    @staticmethod
    def _rivalry_rows(df: pd.DataFrame) -> pd.DataFrame:
        """Keep only rivalry trials that have a ReportedPercept label."""
        if "Type" not in df.columns or "ReportedPercept" not in df.columns:
            raise ValueError(
                "DataFrame must contain 'Type' and 'ReportedPercept' columns. "
                "Run reconstruct with a report run that has a perceptData file."
            )
        return df[(df["Type"] == "rivalry") & df["ReportedPercept"].notna()].copy()

    # ── section-level evaluation (primary / cleanest) ─────────────────────────

    def evaluate_by_section(self, df: pd.DataFrame) -> dict:
        """
        Evaluate at fixation-section granularity — the cleanest signal.

        Within each ``FixpointSection`` the modal ``ReportedPercept`` is the
        ground truth and the pre-computed ``InferredPerceptAggregated`` is the
        prediction.  Sections whose reported mode is NaN are dropped.

        Returns:
            Dict with ``by_section`` DataFrame (one row per section) and
            ``metrics_2class`` / ``metrics_3class`` summary dicts.
        """
        data = self._rivalry_rows(df)
        if "FixpointSection" not in data.columns or "InferredPerceptAggregated" not in data.columns:
            raise ValueError("DataFrame must contain 'FixpointSection' and 'InferredPerceptAggregated'.")

        sec = (
            data.groupby("FixpointSection", sort=False)
            .agg(
                trial=("Trial", "first"),
                reported_mode=("ReportedPercept", lambda x: x.mode().iloc[0] if x.notna().any() else np.nan),
                inferred_agg=("InferredPerceptAggregated", "first"),
                n_samples=("Timestamp", "count"),
            )
            .reset_index()
        )
        sec = sec[sec["reported_mode"].notna()].reset_index(drop=True)

        if sec.empty:
            logger.warning("evaluate_by_section: no usable sections after filtering")
            empty = {"n_samples": 0, "accuracy": np.nan, "f1_macro": np.nan,
                     "f1_weighted": np.nan, "mcc": np.nan, "per_class": {}}
            return {"by_section": sec, "metrics_2class": empty, "metrics_3class": empty}

        metrics_2class = self.evaluate(sec["reported_mode"], sec["inferred_agg"], exclude_mixed=True)
        metrics_3class = self.evaluate(sec["reported_mode"], sec["inferred_agg"], exclude_mixed=False)

        sec["match"] = sec["reported_mode"] == sec["inferred_agg"]
        logger.info(
            f"Section-level eval: n={len(sec)}, "
            f"acc_2class={metrics_2class['accuracy']:.3f}, "
            f"acc_3class={metrics_3class['accuracy']:.3f}"
        )
        return {
            "by_section": sec,
            "metrics_2class": metrics_2class,
            "metrics_3class": metrics_3class,
        }

    # ── trial-level evaluation ────────────────────────────────────────────────

    def evaluate_by_trial(
        self,
        df: pd.DataFrame,
        trial_column: str = "Trial",
    ) -> pd.DataFrame:
        """
        Compute per-trial evaluation metrics (section-level within each trial).

        Args:
            df: Full percept DataFrame (must have ``ReportedPercept``,
                ``InferredPerceptAggregated``, ``FixpointSection``, ``Type``).
            trial_column: Column identifying trials.

        Returns:
            DataFrame with one row per trial and columns:
            ``trial``, ``n_sections``, ``accuracy_2class``, ``f1_macro_2class``,
            ``mcc_2class``, ``accuracy_3class``, ``f1_macro_3class``, ``mcc_3class``.
        """
        data = self._rivalry_rows(df)
        rows = []
        for trial_id, trial_df in data.groupby(trial_column, sort=True):
            sec_result = self.evaluate_by_section(trial_df)
            m2 = sec_result["metrics_2class"]
            m3 = sec_result["metrics_3class"]
            rows.append({
                "trial": trial_id,
                "n_sections": len(sec_result["by_section"]),
                "accuracy_2class": m2["accuracy"],
                "f1_macro_2class": m2["f1_macro"],
                "mcc_2class": m2["mcc"],
                "accuracy_3class": m3["accuracy"],
                "f1_macro_3class": m3["f1_macro"],
                "mcc_3class": m3["mcc"],
            })
        return pd.DataFrame(rows)

    # ── run-level convenience ─────────────────────────────────────────────────

    def evaluate_run(self, df: pd.DataFrame) -> dict:
        """
        Run all evaluation levels and return a single result dict.

        Args:
            df: Full percept sidecar DataFrame (with ``ReportedPercept``).

        Returns:
            Dict with keys ``by_section`` (DataFrame), ``by_trial`` (DataFrame),
            ``summary`` (flat metrics dict for logging / JSON serialisation).
        """
        section_result = self.evaluate_by_section(df)
        trial_result = self.evaluate_by_trial(df)

        m2 = section_result["metrics_2class"]
        m3 = section_result["metrics_3class"]
        summary = {
            "n_sections": len(section_result["by_section"]),
            "n_trials": len(trial_result),
            "accuracy_2class": m2["accuracy"],
            "f1_macro_2class": m2["f1_macro"],
            "f1_weighted_2class": m2["f1_weighted"],
            "mcc_2class": m2["mcc"],
            "accuracy_3class": m3["accuracy"],
            "f1_macro_3class": m3["f1_macro"],
            "f1_weighted_3class": m3["f1_weighted"],
            "mcc_3class": m3["mcc"],
        }
        logger.info(
            f"Run evaluation complete — "
            f"2-class acc={m2['accuracy']:.3f} F1={m2['f1_macro']:.3f} MCC={m2['mcc']:.3f} | "
            f"3-class acc={m3['accuracy']:.3f} F1={m3['f1_macro']:.3f} MCC={m3['mcc']:.3f}"
        )
        return {
            "by_section": section_result["by_section"],
            "by_trial": trial_result,
            "summary": summary,
        }
