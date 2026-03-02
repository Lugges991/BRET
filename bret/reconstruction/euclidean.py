"""
Euclidean distance-based percept reconstruction.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class EuclideanReconstructor:
    """
    Reconstruct perceptual states based on gaze proximity to fixation points.

    Implements the heuristic: dominant percept corresponds to the eye whose
    fixation point is closest to current gaze position.
    """

    def __init__(self, include_mixed: bool = True, ratio_threshold: float = 0.0):
        """
        Initialize reconstructor.

        Args:
            include_mixed: Whether to infer "mixed" percepts when center is closest
            ratio_threshold: Threshold τ for log-distance-ratio classifier.
                             0.0 = equivalent to argmin (current behaviour).
                             Values > 0 create a "mixed" band around the decision
                             boundary (only used by ``infer_percept_ratio``).
        """
        self.include_mixed = include_mixed
        self.ratio_threshold = ratio_threshold
        logger.info(
            f"Initialized EuclideanReconstructor "
            f"(mixed={include_mixed}, ratio_threshold={ratio_threshold})"
        )

    # ── sample-level classifiers ──────────────────────────────────────────────

    def infer_percept_from_closest_fixpoint(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Infer percept based on closest fixation point (argmin rule).

        Args:
            df: DataFrame with distance columns

        Returns:
            DataFrame with 'InferredPercept' column
        """
        df["InferredPercept"] = df.apply(
            lambda row: row["Image1"]
            if row["DistanceToFixpoint1"] < row["DistanceToFixpoint2"]
            else ("face" if row["Image1"] == "house" else "house"),
            axis=1,
        )
        return df

    def infer_percept_with_mixed(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Infer percept including "mixed" state when center is closest.

        Args:
            df: DataFrame with distance columns

        Returns:
            DataFrame with 'InferredPerceptMixed' column
        """
        df["InferredPerceptMixed"] = df.apply(
            lambda row: "mixed"
            if row["DistanceToCenter"] < row["DistanceToFixpoint1"]
            and row["DistanceToCenter"] < row["DistanceToFixpoint2"]
            else (
                row["Image1"]
                if row["DistanceToFixpoint1"] < row["DistanceToFixpoint2"]
                else ("face" if row["Image1"] == "house" else "house")
            ),
            axis=1,
        )
        return df

    def infer_percept_ratio(
        self,
        df: pd.DataFrame,
        threshold: float | None = None,
    ) -> pd.DataFrame:
        """Infer percept using the log distance ratio.

        Computes  r = ln(d1 / d2)  for every sample.
        - r < -τ  → Image1 (closer to fixpoint 1)
        - r >  τ  → Image2
        - |r| ≤ τ → "mixed"

        When τ = 0 (default) this is equivalent to argmin, but the continuous
        ratio value is kept for downstream weighting / smoothing.

        Args:
            df: DataFrame with DistanceToFixpoint1, DistanceToFixpoint2, Image1.
            threshold: Override ``self.ratio_threshold`` for this call.

        Returns:
            DataFrame with added columns:
              - LogDistRatio: continuous log(d1/d2) value
              - InferredPerceptRatio: categorical percept label
        """
        tau = threshold if threshold is not None else self.ratio_threshold

        d1 = df["DistanceToFixpoint1"].values.astype(float)
        d2 = df["DistanceToFixpoint2"].values.astype(float)

        # Avoid log(0): clamp distances to a small positive floor
        eps = 1e-6
        ratio = np.log(np.maximum(d1, eps) / np.maximum(d2, eps))

        df = df.copy()
        df["LogDistRatio"] = ratio

        image1 = df["Image1"].values
        image2 = np.where(image1 == "house", "face", "house")

        percept = np.where(
            ratio < -tau,
            image1,
            np.where(ratio > tau, image2, "mixed"),
        )
        df["InferredPerceptRatio"] = percept

        n_mixed = (percept == "mixed").sum()
        logger.info(
            f"Log-ratio classification (τ={tau:.3f}): "
            f"{n_mixed}/{len(df)} samples classified as mixed "
            f"({100 * n_mixed / max(len(df), 1):.1f}%)"
        )
        return df

    # ── peri-jump exclusion ───────────────────────────────────────────────────

    @staticmethod
    def label_fixpoint_sections(df: pd.DataFrame) -> pd.DataFrame:
        """Add ``FixpointSection`` column (int counter starting at 1).

        A new section starts whenever Fixpoint1 or Fixpoint2 changes.
        This is factored out so that other methods can reuse section labels.
        """
        result = df.copy()
        fp1_change = result["Fixpoint1"].ffill() != result["Fixpoint1"].ffill().shift()
        fp2_change = result["Fixpoint2"].ffill() != result["Fixpoint2"].ffill().shift()
        result["FixpointSection"] = (fp1_change | fp2_change).cumsum().astype(int)
        return result

    @staticmethod
    def exclude_peri_jump_samples(
        df: pd.DataFrame,
        margin_ms: float = 50,
    ) -> pd.DataFrame:
        """Mark samples near fixation-section boundaries as unreliable.

        Adds a boolean column ``PeriJump`` that is True for samples within
        ±``margin_ms`` milliseconds of a section boundary.  These samples
        can be excluded before aggregation.

        The method auto-detects whether timestamps are in seconds or
        milliseconds by inspecting the median inter-sample interval.

        Args:
            df: DataFrame with Timestamp and FixpointSection columns.
            margin_ms: Exclusion margin around each boundary (milliseconds).

        Returns:
            DataFrame with added ``PeriJump`` column.
        """
        if "FixpointSection" not in df.columns:
            raise ValueError("Run label_fixpoint_sections() first.")

        result = df.copy()
        ts = result["Timestamp"].values.astype(float)
        sections = result["FixpointSection"].values

        # Auto-detect time unit: if median dt < 0.5, timestamps are in seconds
        median_dt = np.median(np.diff(ts))
        if median_dt < 0.5:
            # Timestamps are in seconds — convert margin to seconds
            margin = margin_ms / 1000.0
        else:
            # Timestamps are in milliseconds
            margin = margin_ms

        # Find boundary timestamps (first sample of each new section)
        boundaries = ts[np.where(np.diff(sections, prepend=sections[0] - 1) != 0)]

        peri = np.zeros(len(ts), dtype=bool)
        for b in boundaries:
            peri |= (ts >= b - margin) & (ts <= b + margin)

        result["PeriJump"] = peri
        n_excluded = peri.sum()
        logger.info(
            f"Peri-jump exclusion (±{margin_ms} ms, unit_margin={margin:.4f}): "
            f"{n_excluded}/{len(ts)} samples marked "
            f"({100 * n_excluded / max(len(ts), 1):.1f}%)"
        )
        return result

    # ── section-level aggregation ─────────────────────────────────────────────

    def aggregate_by_fixation_section(
        self,
        df: pd.DataFrame,
        percept_col: str = "InferredPercept",
        method: str = "mode",
        exclude_peri_jump: bool = False,
    ) -> pd.DataFrame:
        """
        Aggregate inferred percepts within fixation sections.

        A new fixation section starts whenever Fixpoint1 or Fixpoint2 changes
        (i.e. the fixation point jumps). Within each section the dominant percept
        is determined by the chosen aggregation method and written back to every
        row of that section.

        Args:
            df: DataFrame with columns Fixpoint1, Fixpoint2, and percept_col.
            percept_col: Column to aggregate (e.g. 'InferredPercept' or
                         'InferredPerceptWithMixed').
            method: Aggregation method – only 'mode' is supported for categorical
                    percepts. 'mean' / 'median' are accepted when percept_col
                    contains numeric values.
            exclude_peri_jump: If True, ignore PeriJump==True samples when
                    computing the section-level aggregate. The aggregated label
                    is still written back to *all* rows of the section.

        Returns:
            DataFrame with two new columns:
              - FixpointSection: integer section counter (starts at 1)
              - <percept_col>Aggregated: section-level aggregated percept
        """
        if percept_col not in df.columns:
            raise ValueError(
                f"Column '{percept_col}' not found. "
                f"Run infer_percept_from_closest_fixpoint() or "
                f"infer_percept_with_mixed() first."
            )

        result = df.copy()

        # Ensure section labels exist
        if "FixpointSection" not in result.columns:
            result = self.label_fixpoint_sections(result)

        logger.info(
            f"Detected {result['FixpointSection'].nunique()} fixation sections "
            f"across {len(result)} samples."
        )

        out_col = f"{percept_col}Aggregated"

        # Determine which rows participate in aggregation
        if exclude_peri_jump and "PeriJump" in result.columns:
            agg_mask = ~result["PeriJump"]
            agg_df = result.loc[agg_mask]
            logger.info(
                f"Excluding {(~agg_mask).sum()} peri-jump samples from aggregation."
            )
        else:
            agg_df = result

        if method == "mode":
            mode_map = (
                agg_df.groupby("FixpointSection")[percept_col]
                .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
            )
            result[out_col] = result["FixpointSection"].map(mode_map)
        elif method in ("mean", "median"):
            agg_map = agg_df.groupby("FixpointSection")[percept_col].agg(method)
            result[out_col] = result["FixpointSection"].map(agg_map)
        else:
            raise ValueError(
                f"Unknown aggregation method '{method}'. "
                "Use 'mode', 'mean', or 'median'."
            )

        logger.info(f"Aggregated '{percept_col}' → '{out_col}' using method='{method}'.")
        return result
