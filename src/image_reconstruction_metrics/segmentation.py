from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.image_reconstruction_metrics.reconstruction import threshold_regions


@dataclass(frozen=True)
class BinarySegmentationMetrics:
    dice: float
    iou: float


@dataclass(frozen=True)
class ClassSegmentationMetrics:
    class_id: int
    dice: float
    iou: float


@dataclass(frozen=True)
class ChaosSegmentationReport:
    mapped_prediction: np.ndarray
    mapping: dict[int, int]
    per_class: list[ClassSegmentationMetrics]
    mean_dice: float
    mean_iou: float


def _binary_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> BinarySegmentationMetrics:
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)

    intersection = int(np.logical_and(pred, gt).sum())
    pred_count = int(pred.sum())
    gt_count = int(gt.sum())
    union = int(np.logical_or(pred, gt).sum())

    if pred_count == 0 and gt_count == 0:
        return BinarySegmentationMetrics(dice=1.0, iou=1.0)

    dice = (2.0 * intersection) / (pred_count + gt_count) if (pred_count + gt_count) > 0 else 0.0
    iou = intersection / union if union > 0 else 0.0

    return BinarySegmentationMetrics(dice=float(dice), iou=float(iou))


def map_segments_to_ground_truth(
    segmented_regions: np.ndarray,
    ground_truth_mask: np.ndarray,
    include_background: bool = False,
) -> tuple[np.ndarray, dict[int, int]]:
    """Map thresholded region IDs (0..K) to ground-truth class IDs.

    Uses Hungarian assignment on overlap counts to create a one-to-one mapping
    where possible. If the number of predicted regions differs from the number
    of GT classes, unmatched predicted regions are mapped to the GT class with
    maximum overlap.
    """
    if segmented_regions.shape != ground_truth_mask.shape:
        raise ValueError("segmented_regions and ground_truth_mask must have the same shape")

    pred_ids = np.unique(segmented_regions).astype(int)
    gt_ids = np.unique(ground_truth_mask).astype(int)

    if not include_background:
        gt_ids = gt_ids[gt_ids != 0]

    if gt_ids.size == 0:
        raise ValueError("ground_truth_mask has no valid classes after background filtering")

    overlap = np.zeros((pred_ids.size, gt_ids.size), dtype=np.int64)

    for i, pred_id in enumerate(pred_ids):
        pred_mask = segmented_regions == pred_id
        for j, gt_id in enumerate(gt_ids):
            overlap[i, j] = int(np.logical_and(pred_mask, ground_truth_mask == gt_id).sum())

    cost = -overlap
    row_ind, col_ind = linear_sum_assignment(cost)

    mapping: dict[int, int] = {}
    for r, c in zip(row_ind, col_ind):
        mapping[int(pred_ids[r])] = int(gt_ids[c])

    # Handle any unmatched predicted region IDs.
    for i, pred_id in enumerate(pred_ids):
        pred_key = int(pred_id)
        if pred_key in mapping:
            continue

        best_col = int(np.argmax(overlap[i]))
        mapping[pred_key] = int(gt_ids[best_col])

    mapped_prediction = np.zeros_like(segmented_regions, dtype=np.int32)
    for pred_id, gt_id in mapping.items():
        mapped_prediction[segmented_regions == pred_id] = gt_id

    return mapped_prediction, mapping


def evaluate_chaos_segmentation(
    segmented_regions: np.ndarray,
    ground_truth_mask: np.ndarray,
    include_background: bool = False,
) -> ChaosSegmentationReport:
    """Compute CHAOS-style segmentation metrics (DSC and IoU).

    The thresholded output should be region IDs (0..K), not a 0/255 mask.
    """
    mapped_prediction, mapping = map_segments_to_ground_truth(
        segmented_regions=segmented_regions,
        ground_truth_mask=ground_truth_mask,
        include_background=include_background,
    )

    gt_ids = np.unique(ground_truth_mask).astype(int)
    if not include_background:
        gt_ids = gt_ids[gt_ids != 0]

    per_class: list[ClassSegmentationMetrics] = []
    for class_id in gt_ids:
        metrics = _binary_metrics(
            pred_mask=mapped_prediction == class_id,
            gt_mask=ground_truth_mask == class_id,
        )
        per_class.append(
            ClassSegmentationMetrics(
                class_id=int(class_id),
                dice=metrics.dice,
                iou=metrics.iou,
            )
        )

    mean_dice = float(np.mean([item.dice for item in per_class])) if per_class else 0.0
    mean_iou = float(np.mean([item.iou for item in per_class])) if per_class else 0.0

    return ChaosSegmentationReport(
        mapped_prediction=mapped_prediction,
        mapping=mapping,
        per_class=per_class,
        mean_dice=mean_dice,
        mean_iou=mean_iou,
    )


def evaluate_chaos_from_thresholds(
    image: np.ndarray,
    thresholds,
    ground_truth_mask: np.ndarray,
    include_background: bool = False,
) -> ChaosSegmentationReport:
    """Convenience wrapper from thresholds -> region IDs -> CHAOS metrics."""
    regions = threshold_regions(image=image, thresholds=thresholds)
    return evaluate_chaos_segmentation(
        segmented_regions=regions,
        ground_truth_mask=ground_truth_mask,
        include_background=include_background,
    )
