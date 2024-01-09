from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import re
import argparse

import numpy as np
import torch
from tqdm import tqdm

from lib.utils.eval_utils import coco_into_labels, Table, pairTab
from sklearn.metrics import roc_auc_score

def load_res(bbox_path, logi_path, cls_path, predict_dir, score_path=None):
    res_bboxs, res_logis, res_clses, res_scores = [], [], [], [] # num of pics
    for file_name in tqdm(os.listdir(os.path.join(predict_dir, 'gt_center'))):
        if 'txt' in file_name:
            bbox_file = os.path.join(bbox_path, file_name)
            logi_file = os.path.join(logi_path, file_name)
            cls_file = os.path.join(cls_path, file_name)
            if score_path != None:
                score_file = os.path.join(score_path, file_name)
            
            bboxs = []
            logis = []
            clses = []
            scores = []
            with open(bbox_file) as f_b:
                bboxs = f_b.readlines()
            with open(logi_file) as f_l:
                logis = f_l.readlines()
            with open(cls_file) as f_c:
                clses = f_c.readlines()
            if score_path != None:
                with open(score_file) as f_s:
                    scores = f_s.readlines()
                
            res_bboxs_single = [] # dets a pic * 8
            res_logis_single = [] # dets a pic * 4
            res_clses_single = [] # dets a pic
            
            for bbox, logi, clskv in zip(bboxs, logis, clses):
                bbox = list(map(float, re.split(';|,',bbox.strip())))
                logi = list(map(int, logi.strip().split(',')))
                clskv = int(clskv.strip())
                
                res_bboxs_single.append(bbox)
                res_logis_single.append(logi)
                res_clses_single.append(clskv)
            
            if score_path != None:
                res_scores_single = [] # dets a pic
                for score in scores:
                    res_scores_single.append(float(score.strip()))
                res_scores.append(torch.Tensor(np.array(res_scores_single)))
            
            res_bboxs.append(torch.Tensor(np.array(res_bboxs_single)))
            res_logis.append(torch.Tensor(np.array(res_logis_single)))
            res_clses.append(torch.Tensor(np.array(res_clses_single)))
            
    if score_path != None:
        return res_bboxs, res_logis, res_clses, res_scores
    else:
        return res_bboxs, res_logis, res_clses

def box_iou(box1, box2, eps=1e-7):
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    box1, box2 = torch.cat((box1[:, :2], box1[:, 4:6]), 1), torch.cat((box2[:, :2], box2[:, 4:6]), 1)
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)
    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)

def correct_martix(pred_bbox, label_bbox, iouv=0.5):
    """
    Return correct prediction matrix
    Arguments:
        pred_bbox (array[N, 8])
        label (array[M, 8])
        iouv (float)
    Returns:
        correct (array[N])
        gt_match_pred (array[num of match, 2])
    """
    
    correct = np.zeros(pred_bbox.shape[0]).astype(bool)
    iou = box_iou(label_bbox, pred_bbox)
    x = torch.where(iou >= iouv)  # IoU > threshold
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou] shape: [num of match, 3]
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]] # 按照iou从大到小排序
            # 一个pred对多个gt的处理，只保留iou最大的那个gt
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]] # np.unique返回的是两个数组，第一个数组是去重后的数组，第二个数组是去重后的数组中的索引, 还会根据数值升序排列对应的索引
            # 一个gt对多个pred的处理，只保留iou最大的那个pred
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        correct[matches[:, 1].astype(int)] = True
        # pred_match_gt[matches[:, 1].astype(int)] = matches[:, 0]
        gt_match_pred = torch.tensor(matches[:, :2], dtype=torch.long)
    else:
        gt_match_pred = torch.empty((0, 2), dtype=torch.long)
    return torch.tensor(correct, dtype=torch.bool), gt_match_pred

def evalBbox(correct, pred_score, num_pred, num_label):
    if num_pred == 0 and num_label == 0:
        return 1.0
    elif num_pred == 0 or num_label == 0:
        return 0.0
    
    # sort by score
    correct = correct[torch.sort(pred_score, descending=True)[1]]
    
    tp = correct
    fp = ~tp
    
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    
    precision = tp_cum / (tp_cum + fp_cum + 1e-7) # shape is num_pred
    recall = tp_cum / num_label
    
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, precision[-1], recall[-1]

def same_col(logi1, logi2):
    if logi1[2] <= logi2[2] <= logi1[3]:
        if ((logi1[0] - logi2[1]) == 1) or ((logi2[0] - logi1[1]) == 1):
            return True
        else:
            return False
    elif logi2[2] <= logi1[2] <= logi2[3]:
        if ((logi1[0] - logi2[1]) == 1) or ((logi2[0] - logi1[1]) == 1):
            return True
        else:
            return False
    else:
        return False

def same_row(logi1, logi2):
    if logi1[0] <= logi2[0] <= logi1[1]:
        if ((logi1[2] - logi2[3]) == 1) or ((logi2[2] - logi1[3]) == 1):
            return True
        else:
            return False
    elif logi2[0] <= logi1[0] <= logi2[1]:
        if ((logi1[2] - logi2[3]) == 1) or ((logi2[2] - logi1[3]) == 1):
            return True
        else:
            return False
    else:
        return False

def evalRelation(pred_logi, gt_logi, gt2pred):
    tp = 0   # num of predict true relations
    allt = 0 # num of label relations
    allp = 0 # num of predicted relations
        
    for i in range(pred_logi.shape[0]):
        for j in range(i, pred_logi.shape[0]):
            wui = pred_logi[i]
            wuj = pred_logi[j]
            if same_row(wui, wuj):
                allp = allp + 1.0
            if same_col(wui, wuj):
                allp = allp + 1.0
    for i in range(gt_logi.shape[0]):
        for j in range(i+1, gt_logi.shape[0]):
            sui = gt_logi[i]
            suj = gt_logi[j]
            
            if same_row(sui, suj):
                allt = allt + 1.0
            if same_col(sui, suj):
                allt = allt + 1.0
            
            if i in gt2pred[:, 0] and j in gt2pred[:, 0]:
                tui = pred_logi[gt2pred[np.argwhere(gt2pred[:, 0] == i)[0], 1].item()]
                tuj = pred_logi[gt2pred[np.argwhere(gt2pred[:, 0] == j)[0], 1].item()]
                
                if same_row(sui, suj) and same_row(tui, tuj):
                    tp = tp + 1.0
                if same_col(sui, suj) and same_col(tui, tuj):
                    tp = tp + 1.0
    
    if allp == 0 or allt == 0:
        return 0.
    else:
        p, r = tp/allp, tp/allt
        f1_score = 2 * p * r / (p + r + 1e-7)
        return f1_score

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type = str)
    parser.add_argument('--predict_dir', type = str)
    parser.add_argument('--iou', type = float, default=0.5)
    args = parser.parse_args()

    coco_into_labels(args.dataset_dir, args.predict_dir)

    gt_bbox_path = os.path.join(args.predict_dir, 'gt_center')
    gt_logi_path = os.path.join(args.predict_dir, 'gt_logi')
    gt_cls_path = os.path.join(args.predict_dir, 'gt_cls')
    
    bbox_path = os.path.join(args.predict_dir, 'center')
    logi_path = os.path.join(args.predict_dir, 'logi')
    cls_path = os.path.join(args.predict_dir, 'cls')
    score_path = os.path.join(args.predict_dir, 'score')
    
    gt_bbox, gt_logi, gt_clse = load_res(gt_bbox_path, gt_logi_path, gt_cls_path, args.predict_dir)
    pred_bbox, pred_logi, pred_cls, pred_score = load_res(bbox_path, logi_path, cls_path, args.predict_dir, score_path)
    
    correct_all = []
    pred_score_all = []
    num_label = 0
    logi_acc = []
    logi_rel_f1_all = []
    kv_acc = []
    pred_cls_all, gt_cls_all = [], []
    for si in range(len(pred_bbox)):
        pred_bbox_single = pred_bbox[si]
        pred_logi_single = pred_logi[si]
        pred_cls_single = pred_cls[si]
        pred_score_single = pred_score[si] # len is pred_num
        
        pred_score_all.append(pred_score_single)
        
        gt_bbox_single = gt_bbox[si]
        gt_logi_single = gt_logi[si]
        gt_cls_single = gt_clse[si]
        num_label += gt_bbox_single.shape[0]
        
        # correct: bbox tp or fp, [pred_num]
        correct, gt2pred = correct_martix(pred_bbox_single, gt_bbox_single, args.iou) # len of correct is pred_num
        correct_all.append(correct)

        logi_acc.append(torch.all(pred_logi_single[gt2pred[:, 1]] == gt_logi_single[gt2pred[:, 0]], dim=1).sum().item()/gt2pred.shape[0])
        logi_rel_f1 = evalRelation(pred_logi_single, gt_logi_single, gt2pred)
        logi_rel_f1_all.append(logi_rel_f1)
        
        # kv_acc.append(torch.sum(pred_cls_single[gt2pred[:, 1]] == gt_cls_single[gt2pred[:, 0]]).item()/gt2pred.shape[0])
        pred_cls_all.append(pred_cls_single[gt2pred[:, 1]])
        gt_cls_all.append(gt_cls_single[gt2pred[:, 0]])
    
    correct_all = torch.cat(correct_all, 0)
    pred_score_all = torch.cat(pred_score_all, 0)
    pred_cls_all = torch.cat(pred_cls_all, 0)
    gt_cls_all = torch.cat(gt_cls_all, 0)
    
    # Bbox metric
    bbox_ap, bbox_p, bbox_r = evalBbox(correct_all, pred_score_all, correct_all.shape[0], num_label)
    print('----------------------------------------')
    print('Bbox metric:')
    print('Bbox Precision(IoU {:.2f}): {:.4f}'.format(args.iou, bbox_p))
    print('Bbox Recall(IoU {:.2f}): {:.4f}'.format(args.iou, bbox_r))
    print('Bbox AP(IoU {:.2f}): {:.4f}'.format(args.iou, bbox_ap))
    
    # Logic metric
    print('----------------------------------------')
    print('Logic metric:')
    print('Accuracy of Logical Location: {:.4f}'.format(np.array(logi_acc).mean()))
    print('F1 score of Logical Relation: {:.4f}'.format(np.array(logi_rel_f1_all).mean()))
    
    # Key-Value classifier metric
    print('----------------------------------------')
    print('KV metric:')
    print('AUC of Key-Value: {:.4f}'.format(roc_auc_score(pred_cls_all, gt_cls_all)))
