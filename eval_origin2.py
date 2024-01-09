import torch
import numpy as np

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
    # pred_logi: [num of pred logi position, 4]
    # gt_logi: [num of gt logi position, 4]
    # gt2pred: [num of gt logi position, 2], 表示gt cell id到pred cell id的映射
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


# pred_bbox_single: pred bbox, [pred_num, 8]
# gt_bbox_single: gt bbox, [gt_num, 8]S

# correct: bbox tp or fp, [pred_num]
# gt2pred: gt cell id to pred cell id, [gt_num, 2]
correct, gt2pred = correct_martix(pred_bbox_single, gt_bbox_single, iou=0.5)
logi_rel_f1 = evalRelation(pred_logi_single, gt_logi_single, gt2pred)