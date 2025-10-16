from collections import OrderedDict
from typing import List, Optional, Sequence, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F

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
    from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
except Exception as e:
    roc_auc_score = None
    average_precision_score = None

@METRICS.register_module()
class AnomalyMetric(BaseMetric):
    '''
    Anomaly detection style metric computed on pixel-level score maps. (docstring as before)
    '''
    """
    Evaluate anomaly score maps against binary ground truth masks.

    Args:
        ignore_index (int): Label value in the ground truth mask that should be
            skipped when computing the metrics. Pixels equal to this value are
            treated as *don't care* and will not contribute to AUROC/AP/F1/IoU.
            When your masks already use ``255`` as the anomaly value (e.g.
            masks are ``0`` for normal and ``255`` for anomaly), set
            ``ignore_index`` to a value that is absent in the mask (such as
            ``-1``) or pre-process the mask to map ``255`` to ``1`` before
            evaluation so that anomaly pixels remain valid.
    """
    """
    10-15 新增图像级
    - 像素级（-P）：把所有图的像素分数与像素标签拼接计算 AUROC/AP，阈值扫描取 F1-max、IoU-max。
    - 图像级（-I）：分数=对异常概率图做 21×21 平均池化后取全局最大；标签=是否存在异常像素。
    """
    def __init__(self,
                ignore_index: int = 255,
                num_thresholds: int = 101,
                collect_device: str = 'cpu',
                output_dir: Optional[str] = None,
                format_only: bool = False,
                prefix: Optional[str] = None,
                # 图像级分数的池化配置（复刻你的 test 脚本）
                 image_pool_kernel: int = 21,
                 image_pool_stride: int = 1,
                **kwargs) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.ignore_index = ignore_index
        self.num_thresholds = int(num_thresholds)
        self.format_only = format_only
        self.output_dir = output_dir
        self.image_pool_kernel = int(image_pool_kernel)
        self.image_pool_stride = int(image_pool_stride)
        self.image_pool_padding = self.image_pool_kernel // 2
        if self.output_dir and is_main_process():   # 仅主进程建目录，避免并发冲突
            mkdir_or_exist(self.output_dir)
        # BaseMetric已经提供了 self.results 列表

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """
        Process a batch of data_samples. Append (scores_flat, labels_flat) for each sample.
        """
        """
        在收集完整个数据集后（仅主进程）聚合计算：
          - 归一化分数到 [0,1]
          - AUROC, AP（若二分类标签不齐，则返回 NaN）
          - 在 [0,1] 均匀阈值上扫描，计算 BestF1 / MaxIoU 及对应阈值
          - 不做最终指标计算（在 `compute_metrics` 里统一完成）。
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

            # 创建忽略掩码：等于 ignore_index 的像素会被剔除，不参与统计
            mask = (gt != self.ignore_index)  # 原作者的
            
            # ===== 强力调试日志: 检查掩码是否有效 =====
            # if is_main_process() and i == 0: # 只对每个batch的第一个样本打印一次，避免刷屏
            #     logger.info(f"--- [DEBUG] GT shape: {gt.shape}, Pred shape: {pred.shape}")
            #     logger.info(f"--- [DEBUG] Ignore index: {self.ignore_index}. Number of valid pixels in mask: {np.sum(mask)} / {mask.size}")

            # 如果所有像素都被忽略，这是一个严重问题
            if not np.any(mask):
                if is_main_process():
                    logger.warning(f"[DEBUG] Sample '{data_sample.get('img_path', 'N/A')}' has no valid pixels. All pixels might be ignore_index ({self.ignore_index}). Skipping append.")
                continue

            score_map = self._extract_score_map(pred)

            if score_map.shape != gt.shape:
                raise ValueError(f"Shape mismatch between score_map {score_map.shape} and gt {gt.shape}")
            
            # ---- 像素级：拉平（仅统计有效像素） ----
            scores_flat = score_map[mask].ravel()
            labels_flat = (gt[mask] == 1).astype(np.uint8).ravel()

            # ---- 图像级：avg_pool 后取全局最大（不对 ignore 做 mask，复刻你的脚本）----
            score_t = torch.from_numpy(score_map).float().unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
            pooled = F.avg_pool2d(
                score_t,
                kernel_size=self.image_pool_kernel,
                stride=self.image_pool_stride,
                padding=self.image_pool_padding
            )
            image_score = float(pooled.max().item())
            image_label = int((gt[mask] == 1).any())  # 是否存在任意异常像素


            # ===== 强力调试日志: 检查将要append的数据 =====
            # if is_main_process() and i == 0:
            #      logger.info(f"--- [DEBUG] Appending flattened arrays. scores_flat shape: {list(set(scores_flat))}, labels_flat shape: {labels_flat.shape}")

            # 保存一条复合记录，compute_metrics 再统一汇总
            self.results.append({
                'pixel_scores': scores_flat,
                'pixel_labels': labels_flat,
                'image_score': image_score,
                'image_label': image_label
            })

    # ------------------------- 计算阶段 -------------------------
    def compute_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        logger: MMLogger = MMLogger.get_current_instance()

        if not is_main_process():
            return {}
        if self.format_only:
            logger.info(f'format_only=True, skip metrics. dir={self.output_dir}')
            return {}
        if len(results) == 0:
            logger.warning("No results to compute. Returning empty dict.")
            return {}

        # 1) 汇总像素级/图像级数据
        px_scores = np.concatenate([r['pixel_scores'] for r in results], axis=0).astype(float)
        px_labels = np.concatenate([r['pixel_labels'] for r in results], axis=0).astype(np.uint8)

        img_scores = np.array([r['image_score'] for r in results], dtype=float)
        img_labels = np.array([r['image_label'] for r in results], dtype=np.uint8)

        # 2) 像素级：必要时做 min-max 归一化（softmax 概率则不动）
        px_unique = np.unique(px_scores)
        if px_unique.size == 1:
            px_scores_norm = np.full_like(px_scores, 0.5, dtype=float)
        else:
            smin, smax = float(px_scores.min()), float(px_scores.max())
            if 0.0 <= smin and smax <= 1.0 and px_unique.size > 1:
                px_scores_norm = px_scores
            else:
                px_scores_norm = (px_scores - smin) / (smax - smin + 1e-12)

        # 3) 图像级：分数本就处于 [0,1]（来自 softmax 概率 + avg_pool），保持原样
        img_scores_norm = img_scores

        # ======== 计算图像级指标（-I）========
        auroc_i = float('nan')
        ap_i = float('nan')
        if not (np.all(img_labels == 0) or np.all(img_labels == 1)):
            auroc_i = roc_auc_score(img_labels, img_scores_norm)
            ap_i = average_precision_score(img_labels, img_scores_norm)

        # f1_max-I（按 precision_recall_curve 计算，与你脚本一致）
        precisions, recalls, thresholds = precision_recall_curve(img_labels, img_scores_norm)
        # 注意：thresholds 长度 = len(precisions)-1
        f1_arr = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-12)
        if f1_arr.size > 0 and np.isfinite(f1_arr).any():
            best_idx_i = int(np.nanargmax(f1_arr))
            f1_max_i = float(f1_arr[best_idx_i])
            best_thr_i = float(thresholds[best_idx_i])
        else:
            f1_max_i, best_thr_i = float('nan'), float('nan')

        # ======== 计算像素级指标（-P）========
        auroc_p = float('nan')
        ap_p = float('nan')
        if not (np.all(px_labels == 0) or np.all(px_labels == 1)):
            auroc_p = roc_auc_score(px_labels, px_scores_norm)
            ap_p = average_precision_score(px_labels, px_scores_norm)

        # F1-max（像素级）：均匀阈值扫描，默认 101 个点（可调 num_thresholds）
        best_f1_p, best_thr_p = 0.0, 0.5
        max_iou_p, max_iou_thr = 0.0, 0.5
        for t in np.linspace(0.0, 1.0, self.num_thresholds):
            pred_bin = (px_scores_norm >= t)
            tp = int(np.sum((pred_bin == 1) & (px_labels == 1)))
            fp = int(np.sum((pred_bin == 1) & (px_labels == 0)))
            fn = int(np.sum((pred_bin == 0) & (px_labels == 1)))

            denom_f1 = (2 * tp + fp + fn)
            if denom_f1 > 0:
                f1 = 2 * tp / denom_f1
                if f1 > best_f1_p:
                    best_f1_p, best_thr_p = f1, t

            denom_iou = (tp + fp + fn)
            if denom_iou > 0:
                iou = tp / denom_iou
                if iou > max_iou_p:
                    max_iou_p, max_iou_thr = iou, t

        # 4) 组织输出（百分比化）
        to_pct = lambda x: x * 100.0 if not (isinstance(x, float) and np.isnan(x)) else float('nan')
        metrics = OrderedDict()

        # 旧键（保留向后兼容：像素级）
        metrics['AUROC'] = to_pct(auroc_p)
        metrics['AP'] = to_pct(ap_p)
        metrics['BestF1'] = to_pct(best_f1_p)
        metrics['BestF1Threshold'] = float(best_thr_p)
        metrics['MaxIoU'] = to_pct(max_iou_p)
        metrics['MaxIoUThreshold'] = float(max_iou_thr)

        # 新增显式后缀键（便于与你的 CSV 对齐）
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

        # 5) 日志打印
        print_log('Anomaly results (PIXEL level):', logger)
        for k in ['AUROC-P', 'AP-P', 'f1_max-P', 'BestF1Threshold-P', 'MaxIoU-P', 'MaxIoUThreshold-P']:
            v = metrics[k]
            s = f'{k}: {v:.4f}' if 'Threshold' not in k else f'{k}: {v}'
            print_log(s, logger=logger)

        print_log('Anomaly results (IMAGE level):', logger)
        for k in ['AUROC-I', 'AP-I', 'f1_max-I', 'BestF1Threshold-I']:
            v = metrics[k]
            s = f'{k}: {v:.4f}' if 'Threshold' not in k else f'{k}: {v}'
            print_log(s, logger=logger)

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