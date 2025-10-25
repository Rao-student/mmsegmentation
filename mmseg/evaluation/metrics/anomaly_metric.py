from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from mmengine.dist import is_main_process
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log
from mmengine.utils import mkdir_or_exist

from mmseg.registry import METRICS

try:
    from sklearn.metrics import (average_precision_score, precision_recall_curve,
                                 roc_auc_score)
except Exception:  # pragma: no cover - sklearn may be unavailable in tests
    average_precision_score = None
    precision_recall_curve = None
    roc_auc_score = None


@METRICS.register_module()
class AnomalyMetric(BaseMetric):
    """Metric for anomaly segmentation tasks.

    The implementation is extended to provide both pixel-level metrics (``-P``)
    and image-level metrics (``-I``). The latter matches the evaluation
    procedure shown in the reference script: anomaly maps are average-pooled and
    the global maximum is used as the image score.

    Args:
        ignore_index (int): Value in the ground-truth mask to ignore.
        num_thresholds (int): Number of thresholds for pixel-level F1/IoU scan.
        collect_device (str): Device for result collection across processes.
        output_dir (str, optional): Directory for dumping predictions when
            ``format_only`` is ``True``.
        format_only (bool): Skip metric computation and only format outputs.
        prefix (str, optional): String prefix for metric keys.
        image_pool_kernel (int): Kernel size of the average pooling used for
            image-level scores.
        image_pool_stride (int): Stride of the average pooling used for
            image-level scores.
        anomaly_label (int, optional): Specific pixel value that should be
            treated as anomaly. When ``None`` (default), any non-zero pixel is
            considered anomalous. This accommodates datasets that encode
            anomalies with either ``1`` or ``255`` without extra preprocessing.
    """

    def __init__(self,
                 ignore_index: int = 255,
                 num_thresholds: int = 101,
                 collect_device: str = 'cpu',
                 output_dir: Optional[str] = None,
                 format_only: bool = False,
                 prefix: Optional[str] = None,
                 image_pool_kernel: int = 21,
                 image_pool_stride: int = 1,
                 anomaly_label: Optional[int] = None,
                 **kwargs) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.ignore_index = ignore_index
        self.num_thresholds = int(num_thresholds)
        self.format_only = format_only
        self.output_dir = output_dir
        self.image_pool_kernel = int(image_pool_kernel)
        self.image_pool_stride = int(image_pool_stride)
        self.image_pool_padding = self.image_pool_kernel // 2
        self.anomaly_label = anomaly_label

        if self.output_dir and is_main_process():
            mkdir_or_exist(self.output_dir)

    def process(self, data_batch: dict,
                data_samples: Sequence[dict]) -> None:  # type: ignore[override]
        """Process a batch of predictions and ground truths."""

        logger = MMLogger.get_current_instance()

        for data_sample in data_samples:
            if self.format_only:
                if is_main_process():
                    logger.warning(
                        'format_only=True, skip metric computation in process().'
                    )
                continue

            if 'gt_sem_seg' in data_sample and 'data' in data_sample['gt_sem_seg']:
                gt = data_sample['gt_sem_seg']['data'].squeeze()
            else:
                gt = data_sample['gt'].squeeze()

            if 'seg_logits' in data_sample and 'data' in data_sample['seg_logits']:
                pred = data_sample['seg_logits']['data'].squeeze()
            else:
                pred = data_sample['pred'].squeeze()

            gt = gt.detach().cpu().numpy() if isinstance(gt, torch.Tensor) else np.asarray(gt)
            pred = pred.detach().cpu().numpy() if isinstance(pred, torch.Tensor) else np.asarray(pred)

            mask = (gt != self.ignore_index)
            if not np.any(mask):
                if is_main_process():
                    logger.warning(
                        "Sample '%s' has no valid pixels. Skipping.",
                        data_sample.get('img_path', 'N/A'))
                continue

            score_map = self._extract_score_map(pred)
            if score_map.shape != gt.shape:
                raise ValueError(
                    f'Shape mismatch between score_map {score_map.shape} '
                    f'and ground truth {gt.shape}.')

            valid_gt = gt[mask]
            binary_gt = self._binarize_gt(valid_gt)
            pixel_scores = score_map[mask].ravel()
            pixel_labels = binary_gt.ravel()

            score_tensor = torch.from_numpy(score_map).float().unsqueeze(0).unsqueeze(0)
            pooled = F.avg_pool2d(
                score_tensor,
                kernel_size=self.image_pool_kernel,
                stride=self.image_pool_stride,
                padding=self.image_pool_padding)
            image_score = float(pooled.max().item())
            image_label = int(binary_gt.any())

            self.results.append(
                dict(pixel_scores=pixel_scores,
                     pixel_labels=pixel_labels,
                     image_score=image_score,
                     image_label=image_label))

    def compute_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        logger = MMLogger.get_current_instance()

        if not is_main_process():
            return {}
        if self.format_only:
            logger.info('format_only=True, skip metric computation.')
            return {}
        if len(results) == 0:
            logger.warning('No results to compute metrics.')
            return {}

        px_scores = np.concatenate([r['pixel_scores'] for r in results],
                                    axis=0).astype(float)
        px_labels = np.concatenate([r['pixel_labels'] for r in results],
                                    axis=0).astype(np.uint8)

        img_scores = np.array([r['image_score'] for r in results], dtype=float)
        img_labels = np.array([r['image_label'] for r in results], dtype=np.uint8)

        px_unique = np.unique(px_scores)
        if px_unique.size == 1:
            px_scores_norm = np.full_like(px_scores, 0.5, dtype=float)
        else:
            smin, smax = float(px_scores.min()), float(px_scores.max())
            if 0.0 <= smin and smax <= 1.0 and px_unique.size > 1:
                px_scores_norm = px_scores
            else:
                px_scores_norm = (px_scores - smin) / (smax - smin + 1e-12)

        img_scores_norm = img_scores

        auroc_i = float('nan')
        ap_i = float('nan')
        f1_max_i = float('nan')
        best_thr_i = float('nan')
        if roc_auc_score is not None and precision_recall_curve is not None:
            if not (np.all(img_labels == 0) or np.all(img_labels == 1)):
                auroc_i = roc_auc_score(img_labels, img_scores_norm)
                ap_i = average_precision_score(img_labels, img_scores_norm)

            precisions, recalls, thresholds = precision_recall_curve(
                img_labels, img_scores_norm)
            if thresholds.size > 0:
                f1_arr = 2 * precisions[:-1] * recalls[:-1] / (
                    precisions[:-1] + recalls[:-1] + 1e-12)
                if f1_arr.size > 0 and np.isfinite(f1_arr).any():
                    best_idx = int(np.nanargmax(f1_arr))
                    f1_max_i = float(f1_arr[best_idx])
                    best_thr_i = float(thresholds[best_idx])

        auroc_p = float('nan')
        ap_p = float('nan')
        if roc_auc_score is not None:
            if not (np.all(px_labels == 0) or np.all(px_labels == 1)):
                auroc_p = roc_auc_score(px_labels, px_scores_norm)
                ap_p = average_precision_score(px_labels, px_scores_norm)

        best_f1_p, best_thr_p = 0.0, 0.5
        max_iou_p, max_iou_thr = 0.0, 0.5
        for thr in np.linspace(0.0, 1.0, self.num_thresholds):
            pred_bin = (px_scores_norm >= thr)
            tp = int(np.sum((pred_bin == 1) & (px_labels == 1)))
            fp = int(np.sum((pred_bin == 1) & (px_labels == 0)))
            fn = int(np.sum((pred_bin == 0) & (px_labels == 1)))

            denom_f1 = (2 * tp + fp + fn)
            if denom_f1 > 0:
                f1 = 2 * tp / denom_f1
                if f1 > best_f1_p:
                    best_f1_p, best_thr_p = f1, thr

            denom_iou = (tp + fp + fn)
            if denom_iou > 0:
                iou = tp / denom_iou
                if iou > max_iou_p:
                    max_iou_p, max_iou_thr = iou, thr

        def to_pct(value: float) -> float:
            if isinstance(value, float) and np.isnan(value):
                return float('nan')
            return value * 100.0

        metrics = OrderedDict()
        metrics['AUROC'] = to_pct(auroc_p)
        metrics['AP'] = to_pct(ap_p)
        metrics['BestF1'] = to_pct(best_f1_p)
        metrics['BestF1Threshold'] = float(best_thr_p)
        metrics['MaxIoU'] = to_pct(max_iou_p)
        metrics['MaxIoUThreshold'] = float(max_iou_thr)

        metrics['AUROC-P'] = to_pct(auroc_p)
        metrics['AP-P'] = to_pct(ap_p)
        metrics['f1_max-P'] = to_pct(best_f1_p)
        metrics['BestF1Threshold-P'] = float(best_thr_p)
        metrics['MaxIoU-P'] = to_pct(max_iou_p)
        metrics['MaxIoUThreshold-P'] = float(max_iou_thr)

        metrics['AUROC-I'] = to_pct(auroc_i)
        metrics['AP-I'] = to_pct(ap_i)
        metrics['f1_max-I'] = to_pct(f1_max_i)
        metrics['BestF1Threshold-I'] = float(best_thr_i)

        print_log('Anomaly results (PIXEL level):', logger)
        for key in [
                'AUROC-P', 'AP-P', 'f1_max-P', 'BestF1Threshold-P', 'MaxIoU-P',
                'MaxIoUThreshold-P'
        ]:
            value = metrics[key]
            msg = (f'{key}: {value:.4f}' if 'Threshold' not in key else
                   f'{key}: {value}')
            print_log(msg, logger=logger)

        print_log('Anomaly results (IMAGE level):', logger)
        for key in ['AUROC-I', 'AP-I', 'f1_max-I', 'BestF1Threshold-I']:
            value = metrics[key]
            msg = (f'{key}: {value:.4f}' if 'Threshold' not in key else
                   f'{key}: {value}')
            print_log(msg, logger=logger)

        return metrics

    @staticmethod
    def _extract_score_map(pred):
        arr = np.array(pred)
        if arr.ndim == 2:
            return arr.astype(float)
        if arr.ndim == 3:
            if arr.shape[0] in (1, 2):
                c, h, w = arr.shape
            elif arr.shape[2] in (1, 2):
                h, w, c = arr.shape
            else:
                raise ValueError(f'Unsupported prediction shape: {arr.shape}')

            if arr.shape[0] == c:  # channel-first
                if c == 1:
                    return arr[0].astype(float)
                exp = np.exp(arr - arr.max(axis=0, keepdims=True))
                probs = exp / (exp.sum(axis=0, keepdims=True) + 1e-12)
                return probs[1].astype(float)
            else:  # channel-last
                if c == 1:
                    return arr[:, :, 0].astype(float)
                exp = np.exp(arr - arr.max(axis=2, keepdims=True))
                probs = exp / (exp.sum(axis=2, keepdims=True) + 1e-12)
                return probs[:, :, 1].astype(float)
        raise ValueError(f'Unsupported prediction shape: {arr.shape}')

    def _binarize_gt(self, values: np.ndarray) -> np.ndarray:
        """Convert raw ground-truth values into binary anomaly labels.

        If ``anomaly_label`` is specified, treat pixels equal to that value as
        anomalous. Otherwise, adopt a generic heuristic that marks any
        non-zero value as anomaly. This makes the metric robust to both
        ``{0, 1}`` and ``{0, 255}`` style masks without additional
        pre-processing.
        """

        if values.dtype == bool:
            return values.astype(np.uint8)
        if self.anomaly_label is not None:
            return (values == self.anomaly_label).astype(np.uint8)
        return (values != 0).astype(np.uint8)
