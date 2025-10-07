from collections import OrderedDict
from typing import List, Optional, Sequence

import numpy as np
import torch

from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log
from mmengine.utils import mkdir_or_exist
from mmengine.dist import is_main_process
from mmseg.registry import METRICS

'''
2025-10-6
sklearn 用于 auc / ap / precision-recall
'''

try:
    from sklearn.metrics import roc_auc_score, average_precision_score
except Exception as e:
    roc_auc_score = None
    average_precision_score = None

@METRICS.register_module()
class AnomalyMetric(BaseMetric):
    '''
    Anomaly detection style metric computed on pixel-level score maps. (docstring as before)
    '''
    def __init__(self,
                ignore_index: int = 255,
                num_thresholds: int = 101,
                collect_device: str = 'cpu',
                output_dir: Optional[str] = None,
                format_only: bool = False,
                prefix: Optional[str] = None,
                **kwargs) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.ignore_index = ignore_index
        self.num_thresholds = int(num_thresholds)
        self.format_only = format_only
        self.output_dir = output_dir
        if self.output_dir and is_main_process():
            mkdir_or_exist(self.output_dir)
        # BaseMetric已经提供了 self.results 列表

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """
        Process a batch of data_samples. Append (scores_flat, labels_flat) for each sample.
        """
        # ===== 强力调试日志: 检查process方法是否被调用以及数据流 =====
        logger: MMLogger = MMLogger.get_current_instance()
        # if is_main_process():
        #      logger.info(f"--- [DEBUG] Entering AnomalyMetric.process() for a batch of {len(data_samples)} samples.")

        for i, data_sample in enumerate(data_samples):
            # format_only 检查
            if self.format_only:
                if is_main_process():
                    logger.warning("[DEBUG] format_only=True, skipping metric processing in .process().")
                continue

            # Extract GT
            try:
                if 'gt_sem_seg' in data_sample and 'data' in data_sample['gt_sem_seg']:
                    gt = data_sample['gt_sem_seg']['data'].squeeze()
                else: # 兼容旧版或不同格式
                    gt = data_sample['gt'].squeeze()
            except KeyError:
                raise KeyError("No ground truth found in data_sample (expect 'gt_sem_seg' or 'gt').")

            # Extract prediction
            try:
                # if 'pred_sem_seg' in data_sample and 'data' in data_sample['pred_sem_seg']:
                #     pred = data_sample['pred_sem_seg']['data'].squeeze()
                if 'seg_logits' in data_sample and 'data' in data_sample['seg_logits']:
                    pred = data_sample['seg_logits']['data'].squeeze()
                else: # 兼容旧版或不同格式
                    pred = data_sample['pred'].squeeze()
            except KeyError:
                raise KeyError("No prediction found in data_sample (expect 'pred_sem_seg' or 'pred').")

            # 转换为 numpy
            gt = gt.detach().cpu().numpy() if isinstance(gt, torch.Tensor) else np.array(gt)
            pred = pred.detach().cpu().numpy() if isinstance(pred, torch.Tensor) else np.array(pred)

            # 创建忽略掩码
            mask = (gt != self.ignore_index)
            
            # ===== 强力调试日志: 检查掩码是否有效 =====
            if is_main_process() and i == 0: # 只对每个batch的第一个样本打印一次，避免刷屏
                logger.info(f"--- [DEBUG] GT shape: {gt.shape}, Pred shape: {pred.shape}")
                logger.info(f"--- [DEBUG] Ignore index: {self.ignore_index}. Number of valid pixels in mask: {np.sum(mask)} / {mask.size}")

            # 如果所有像素都被忽略，这是一个严重问题
            # if not np.any(mask):
            #     if is_main_process():
            #         logger.warning(f"[DEBUG] Sample '{data_sample.get('img_path', 'N/A')}' has no valid pixels. All pixels might be ignore_index ({self.ignore_index}). Skipping append.")
            #     continue

            score_map = self._extract_score_map(pred)

            if score_map.shape != gt.shape:
                raise ValueError(f"Shape mismatch between score_map {score_map.shape} and gt {gt.shape}")

            scores_flat = score_map[mask].ravel()
            labels_flat = (gt[mask] == 1).astype(np.uint8).ravel()

            # ===== 强力调试日志: 检查将要append的数据 =====
            # if is_main_process() and i == 0:
            #      logger.info(f"--- [DEBUG] Appending flattened arrays. scores_flat shape: {list(set(scores_flat))}, labels_flat shape: {labels_flat.shape}")

            self.results.append((scores_flat, labels_flat))

    def compute_metrics(self, results: list) -> dict:
        """
        Compute metrics after all batches have been processed.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        # 这一段由BaseMetric的收集机制保证了只在主进程运行，但为了清晰，我们还是保留
        if not is_main_process():
            return {}
        
        logger.info("--- [DEBUG] Entering AnomalyMetric.compute_metrics()...")
        logger.info(f"--- [DEBUG] Received {len(results)} results to compute.")

        if self.format_only:
            logger.info(f'format_only=True, results are saved to {self.output_dir}')
            return OrderedDict()

        if not results:
            logger.warning("`results` list is empty. No metrics will be computed. Check if .process() appended any data.")
            return OrderedDict()

        # concatenate all arrays
        all_scores = np.concatenate([r[0] for r in results])
        all_labels = np.concatenate([r[1] for r in results])

        # logger.info(f"--- [DEBUG] Concatenated all scores and labels. Total pixels to evaluate: {all_scores}")

        # handle trivial case of no valid pixels across all batches
        if all_labels.size == 0:
            logger.warning("All processed samples resulted in zero valid pixels after masking. Cannot compute metrics.")
            metrics = OrderedDict()
            metrics['AUROC'], metrics['AP'], metrics['BestF1'], metrics['MaxIoU'] = (float('nan'),) * 4
            metrics['BestF1Threshold'], metrics['MaxIoUThreshold'] = (float('nan'),) * 2
            return metrics
        
        # --- 你的验证逻辑（现在应该能打印了）---
        unique_scores = np.unique(all_scores)
        logger.info(f"验证：预测分数中的唯一值 (前20个): {unique_scores[:20]}")
        if len(unique_scores) <= 2:
            logger.warning(
                "检测到预测分数似乎是二值的！AUROC, AP, BestF1, MaxIoU 等指标可能无意义。")
        # ----------------------------------------

        metrics = OrderedDict()
        
        # ... (你的后续计算代码保持不变) ...
        # (代码从 normalize scores to [0,1] 开始)
        smin, smax = all_scores.min(), all_scores.max()
        scores_norm = (all_scores - smin) / (smax - smin) if smax > smin else np.zeros_like(all_scores, dtype=float)

        try:
            if roc_auc_score is None: raise ImportError("sklearn.metrics not available")
            if np.unique(all_labels).size < 2:
                auroc, ap = float('nan'), float('nan')
            else:
                auroc = roc_auc_score(all_labels, scores_norm)
                ap = average_precision_score(all_labels, scores_norm)
        except Exception as e:
            logger.error(f"Error calculating AUROC/AP: {e}")
            auroc, ap = float('nan'), float('nan')

        best_f1, best_f1_thresh = 0.0, 0.5
        max_iou, max_iou_thresh = 0.0, 0.5
        thresholds = np.linspace(0.0, 1.0, self.num_thresholds)

        for t in thresholds:
            pred_bin = (scores_norm >= t)
            tp = np.sum((pred_bin == 1) & (all_labels == 1))
            fp = np.sum((pred_bin == 1) & (all_labels == 0))
            fn = np.sum((pred_bin == 0) & (all_labels == 1))
            
            if (2 * tp + fp + fn) > 0:
                f1 = 2 * tp / (2 * tp + fp + fn)
                if f1 > best_f1: best_f1, best_f1_thresh = f1, t
            
            if (tp + fp + fn) > 0:
                iou = tp / (tp + fp + fn)
                if iou > max_iou: max_iou, max_iou_thresh = iou, t

        to_pct = lambda x: x * 100.0 if not np.isnan(x) else float('nan')
        metrics['AUROC'] = to_pct(auroc)
        metrics['AP'] = to_pct(ap)
        metrics['BestF1'] = to_pct(best_f1)
        metrics['BestF1Threshold'] = best_f1_thresh
        metrics['MaxIoU'] = to_pct(max_iou)
        metrics['MaxIoUThreshold'] = max_iou_thresh

        print_log('Anomaly detection results (pixel-level):', logger)
        for k, v in metrics.items():
            log_str = f'{k}: {v:.4f}' if 'Threshold' not in k else f'{k}: {v}'
            print_log(log_str, logger=logger)

        return metrics


    @staticmethod
    def _extract_score_map(pred):
        # ... (这个静态方法无需改动) ...
        arr = np.array(pred)
        if arr.ndim == 2: return arr.astype(float)
        if arr.ndim == 3:
            if arr.shape[0] in (1, 2): C, H, W = arr.shape
            elif arr.shape[2] in (1, 2): H, W, C = arr.shape
            else: raise ValueError(f"Unsupported prediction shape: {arr.shape}")
            
            if arr.shape[0] == C: # Channel-first
                if C == 1: return arr[0].astype(float)
                exp = np.exp(arr - arr.max(axis=0, keepdims=True))
                probs = exp / (exp.sum(axis=0, keepdims=True) + 1e-12)
                return probs[1].astype(float)
            else: # Channel-last
                if C == 1: return arr[:, :, 0].astype(float)
                exp = np.exp(arr - arr.max(axis=2, keepdims=True))
                probs = exp / (exp.sum(axis=2, keepdims=True) + 1e-12)
                return probs[:, :, 1].astype(float)
        raise ValueError(f"Unsupported prediction shape: {arr.shape}")