'''
xuyilun, wangziwei

cd:  Cell Detection 		Detection F1 (IoU=0.5)
ctc: Cell Type Classification		Classification F1 (Macro)
tsr_rc / tsr_teds / tsr_ll: Table Structure Recognition/Relationship Classification F1 / TEDS/Logical Location F1

'''

import torch
import numpy as np
from sklearn.metrics import f1_score

import distance
from apted import APTED, Config
from apted.helpers import Tree
from lxml import etree, html
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


class Evaluator():

    def __init__(self, cal_tasks, match_type):
        self.cal_tasks = cal_tasks                 # ['cd', 'ctc', 'tsr_rc', 'tsr_teds', 'tsr_ll']
        self.t2a, self.a2t = False, False
        if match_type == 't2a':
            self.t2a = True
        elif match_type == 'a2t':
            self.a2t = True

        self.iouv = 0.5  # iou threshold value

        ### Task1/2. For Bounding Box Dection; Text Region Detection and Aligned Cell Detection (according to the dataset)
        self.BBD_F1 = 0.  # Bounding box dection
        self.BBD_tp = 0  # tp
        self.BBD_allp = 0  # p = tp / allp     predicted results
        self.BBD_allt = 0  # R = 
        
        # For AP metric compute
        self.correct_all = []
        self.pred_score_all = []

        ### Task 3. For Cell Type Classification

        self.CTC_F1 = 0.
        self.CTC_pred, self.CTC_gt = [], []

        ### Task 4. For Table Structure Recognition

        # (Relationship Classification F1)
        self.RC_F1 = 0.
        self.RC_tp = 0  # num of predict true relations
        self.RC_allp = 0  # num of predicted relations
        self.RC_allt = 0  # num of label relations

        # (Logical Location ACC)
        self.LL_ACC = 0.
        # self.LL_F1 = 0.
        self.LL_tp = 0
        self.LL_allp = 0
        self.LL_allt = 0

        self.LL_rb_tp = 0
        self.LL_re_tp = 0
        self.LL_cb_tp = 0
        self.LL_ce_tp = 0
        
        # (TEDS)
        self.TEDS = []
        self.teds_cal = TEDS(structure_only=True, n_jobs=16)
        ###########################################################

    def cal_box_iou(self, box1, box2, eps=1e-7):  # 计算IoU得分->(N,M) float
        """
        Intersection-over-union (Jaccard index) of boxes.

        Arguments:
        ----------
            box1 : np.ndarray (array[n1, 8]) 
            box2 : np.ndarray (array[n2, 8])

        Returns:
        --------
            iou : torch.Tensor([n1, n2], dtype=float)
                the matrix containing the pairwise IoU values 
                for every element in boxes1 and boxes2.
        """
        # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
        box1, box2 = torch.cat((torch.from_numpy(box1)[:, :2], torch.from_numpy(box1)[:, 4:6]), 1), torch.cat(
            (torch.from_numpy(box2)[:, :2], torch.from_numpy(box2)[:, 4:6]), 1)

        (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)

        inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

        # IoU = inter / (area1 + area2 - inter)
        return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)
    
    def cal_box_iou_t2a(self, box1, box2, eps=1e-7):  # 计算text region和align cell之间的IoU得分->(N,M) float
        """
        Intersection-over-union (Jaccard index) of boxes for text region output and align cell ground truth.
        If a text region bbox is almost in a align cell bbox, we match them. So we change the iou caculation's denominator to the area of text region bbox.

        Arguments:
        ----------
            box1 : np.ndarray (array[N, 8])
                text region bbox
            box2 : np.ndarray (array[M, 8])
                align cell bbox

        Returns:
        --------
            iou : torch.Tensor([N, M], dtype=float)
                the NxM matrix containing the pairwise IoU values 
                for every element in boxes1 and boxes2.
        """
        # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
        box1, box2 = torch.cat((torch.from_numpy(box1)[:, :2], torch.from_numpy(box1)[:, 4:6]), 1), torch.cat(
            (torch.from_numpy(box2)[:, :2], torch.from_numpy(box2)[:, 4:6]), 1)

        (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)

        inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

        # IoU = inter / area1
        return inter / ((a2 - a1).prod(2) + eps)
    
    def cal_box_iou_a2t(self, box1, box2, eps=1e-7):  # 计算align cell和text region之间的IoU得分->(N,M) float
        """
        Intersection-over-union (Jaccard index) of boxes for align cell output and text region ground truth.
        If a align cell bbox almost includes a text region bbox, we match them. So we change the iou caculation's denominator to the area of text region bbox.
        
        Arguments:
        ----------
            box1 : np.ndarray (array[N, 8])
                align cell bbox
            box2 : np.ndarray (array[M, 8])
                text region bbox
        
        Returns:
        --------
            iou : torch.Tensor([N, M], dtype=float)
                the NxM matrix containing the pairwise IoU values 
                for every element in boxes1 and boxes2.
        """
        # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
        box1, box2 = torch.cat((torch.from_numpy(box1)[:, :2], torch.from_numpy(box1)[:, 4:6]), 1), torch.cat(
            (torch.from_numpy(box2)[:, :2], torch.from_numpy(box2)[:, 4:6]), 1)

        (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)

        inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

        # IoU = inter / area2
        return inter / ((b2 - b1).prod(2) + eps)
    
    def correct_martix(self, pred_bbox, gt_bbox): # 计算大于IOU阈值(与gt匹配)的bbox索引
        """
        Return correct prediction matrix

        Arguments:
        ----------
            pred_bbox : np.ndarray(array[N, 8]) 
                predicted bounding boxes of four corners.
            gt : np.ndarray(array[M, 8]) 
                ground truth bounding boxes of four corners.

        Returns:
        --------
            correct : torch.tensor(tensor[N], dtype=int)
                tp or fp for every predicted bounding box.
            gt_match_pred : torch.tensor(tensor[num of match, 2], dtype=int)
                gt cell id to pred cell id mapping.
        """
        iouv = self.iouv
        correct = np.zeros(pred_bbox.shape[0]).astype(bool)
        iou = self.cal_box_iou(gt_bbox, pred_bbox)  # (Tensor[M, N]) float
        x = torch.where(iou >= iouv)  # IoU > Threshold, 返回两个tensor数组表示达到阈值的元素的横纵坐标，即gt_id, pred_id

        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1),
                                 iou[x[0], x[1]][:, None]),
                                1).cpu().numpy()  # [gt_id, pred_id, iou_score] shape: [num of match, 3]

            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]  # 按照iou从大到小排序

                # 一个pred对多个gt的处理，只保留iou最大的那个gt
                # np.unique返回的是两个数组，第一个数组是去重后的数组，第二个数组是去重后的数组中的索引, 还会根据选中维度的数值升序排列对应的索引
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]

                # 按照iou从大到小排序, 因为前面的去重操作会破坏iou的降序
                matches = matches[matches[:, 2].argsort()[::-1]]

                # 一个gt对多个pred的处理，只保留iou最大的那个pred
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]

            correct[matches[:, 1].astype(int)] = True
            # pred_match_gt[matches[:, 1].astype(int)] = matches[:, 0]
            gt_match_pred = torch.tensor(matches[:, :2], dtype=torch.int)
        else:
            gt_match_pred = torch.empty((0, 2), dtype=torch.int)
        
        return torch.tensor(correct, dtype=torch.bool), gt_match_pred
    
    def textregion_2_aligncell(self, test_region_pred, align_cell_gt, pred_logi):
        """
        Matching function for detected text regions and aligned cells groundtruth
        
        Arguments:
        ----------
            test_region_pred : np.ndarray(array[N, 8]) 
                predicted bounding boxes of four corners.
            align_cell_gt : np.ndarray(array[M, 8]) 
                ground truth bounding boxes of four corners.
            pred_logi : np.ndarray(array[N, 4]) 
                predicted logical location of cells.
        
        Returns:
        --------
            correct : torch.tensor(tensor[N], dtype=int)
                tp or fp for every predicted bounding box.
            gt_match_pred : torch.tensor(tensor[num of match, 2], dtype=int)
                gt cell id to pred cell id mapping.
            new_pred_logi : np.ndarray(array[N2, 4])
                predicted logical location of cells after merging.
        """
        correct = np.zeros(test_region_pred.shape[0]).astype(bool)
        iou = self.cal_box_iou_t2a(test_region_pred, align_cell_gt) # (N, M) float
        x = torch.where(iou >= 0.8)  # IoU > Threshold, 返回两个tensor数组表示达到阈值的元素的横纵坐标，即pred_id, gt_id 

        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1),
                                 iou[x[0], x[1]][:, None]),
                                1).cpu().numpy()  # [pred_id, gt_id, iou_score] shape: [num of match, 3]

            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]  # 按照iou从大到小排序
                # 一个pred对多个gt的处理，只保留iou最大的那个gt
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]] 
                
                # 多个pred对一个gt的处理
                new_matches = []
                gt_pred_first = {}
                for i in range(matches.shape[0]):
                    if matches[i, 1] not in gt_pred_first:
                        gt_pred_first[matches[i, 1]] = matches[i, 0]
                        new_matches.append(matches[i])
                    else:
                        base_pid = int(gt_pred_first[matches[i, 1]])
                        add_pid = int(matches[i, 0])
                        pred_logi[base_pid] = np.array([min(pred_logi[base_pid][0], pred_logi[add_pid][0]), 
                                                        max(pred_logi[base_pid][1], pred_logi[add_pid][1]), 
                                                        min(pred_logi[base_pid][2], pred_logi[add_pid][2]), 
                                                        max(pred_logi[base_pid][3], pred_logi[add_pid][3])])
                
                matches = np.array(new_matches)
            correct[matches[:, 0].astype(int)] = True
            gt_match_pred = torch.tensor(matches[:, [1, 0]], dtype=torch.int) # turn [pid, gid] to [gid, pid]
            # print(gt_match_pred)
        else:
            gt_match_pred = torch.empty((0, 2), dtype=torch.int)
        new_pred_logi = pred_logi
        return torch.tensor(correct, dtype=torch.bool), gt_match_pred, new_pred_logi

    def aligncell_2_textregion(self, align_cell_pred, test_region_gt, gt_logi):
        """
        Matching function for detected aligned cells and text regions groundtruth
        
        Arguments:
        ----------
            align_cell_pred : np.ndarray(array[N, 8]) 
                predicted bounding boxes of four corners.
            text_region_gt : np.ndarray(array[M, 8]) 
                ground truth bounding boxes of four corners.
            gt_logi : np.ndarray(array[N, 4]) 
                ground truth logical location of cells.
        
        Returns:
        --------
            correct : torch.tensor(tensor[N], dtype=int)
                tp or fp for every predicted bounding box.
            gt_match_pred : torch.tensor(tensor[num of match, 2], dtype=int)
                gt cell id to pred cell id mapping.
            new_gt_logi : np.ndarray(array[N2, 4])
                ground truth logical location of cells after merging.
        """
        correct = np.zeros(align_cell_pred.shape[0]).astype(bool)
        iou = self.cal_box_iou_a2t(align_cell_pred, test_region_gt) # (N, M) float
        x = torch.where(iou >= 0.8)  # IoU > Threshold, 返回两个tensor数组表示达到阈值的元素的横纵坐标，即pred_id, gt_id 

        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1),
                                 iou[x[0], x[1]][:, None]),
                                1).cpu().numpy()  # [pred_id, gt_id, iou_score] shape: [num of match, 3]

            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]  # 按照iou从大到小排序
                # 多个pred对一个gt的处理，只保留iou最大的那个pred
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                
                # 一个pred对多个gt的处理
                new_matches = []
                pred_gt_first = {}
                for i in range(matches.shape[0]):
                    if matches[i, 0] not in pred_gt_first:
                        pred_gt_first[matches[i, 0]] = matches[i, 1]
                        new_matches.append(matches[i])
                    else:
                        base_gid = int(pred_gt_first[matches[i, 0]])
                        add_gid = int(matches[i, 1])
                        gt_logi[base_gid] = np.array([min(gt_logi[base_gid][0], gt_logi[add_gid][0]),
                                                      max(gt_logi[base_gid][1], gt_logi[add_gid][1]),
                                                      min(gt_logi[base_gid][2], gt_logi[add_gid][2]),
                                                      max(gt_logi[base_gid][3], gt_logi[add_gid][3])])
                
                matches = np.array(new_matches)
            correct[matches[:, 0].astype(int)] = True
            gt_match_pred = torch.tensor(matches[:, [1, 0]], dtype=torch.int) # turn [pid, gid] to [gid, pid]
        else:
            gt_match_pred = torch.empty((0, 2), dtype=torch.int)
        new_gt_logi = gt_logi
        return torch.tensor(correct, dtype=torch.bool), gt_match_pred, new_gt_logi

    def same_col(self, logi1, logi2):
        """
        Judge whether two cells are in the same column
        
        Arguments:
        ----------
            logi1 : np.ndarray(array[4]) 
                logical location of cell1
            logi2 : np.ndarray(array[4]) 
                logical location of cell2
        
        Returns:
        --------
            bool
        """
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

    def same_row(self, logi1, logi2):
        """
        Judge whether two cells are in the same row
        
        Arguments:
        ----------
            logi1 : np.ndarray(array[4]) 
                logical location of cell1
            logi2 : np.ndarray(array[4]) 
                logical location of cell2
        
        Returns:
        --------
            bool
        """
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
    ##############

    def cal_bbox_detection(self, num_pred_bbox, num_gt_bbox, correct): # for Task 1/2. Bounding Box Dection; Text Region Detection and Aligned Cell Detection (according to the dataset)
        """
        Caculate number of true positive, predicted and ground truth bounding boxes

        Args:
        ----------
            num_pred_bbox : int
                num of predicted bounding boxes.
            num_gt_bbox : int
                num of ground truth bounding boxes.
            correct : torch.tensor(tensor[N], dtype=int)
                tp or fp for every predicted bounding box.
                
        Returns:
        --------
            None
        """
        self.BBD_allp += num_pred_bbox
        self.BBD_allt += num_gt_bbox

        tp = int(torch.count_nonzero(correct))
        self.BBD_tp += tp
        
        self.correct_all.append(correct)

    def cal_relation_dection(self, pred_logi, gt_logi, gt2pred):  # for Task 4. Relationship Classification F1
        """
        Caculate number of true positive, predicted and ground truth relations
        
        Args:
        ----------
            pred_logi : np.ndarray(array[N, 4])
                predicted logical location of cells. N is the number of predicted cells.
            gt_logi : np.ndarray(array[M, 4])
                ground truth logical location of cells. M is the number of ground truth cells.
            gt2pred : torch.tensor(tensor[num of match, 2], dtype = int)
                gt cell id to pred cell id mapping.
        
        Returns:
        --------
            None
        """

        # for i in range(pred_logi.shape[0]):  # 根据pred的内容整理出pred的relation数量
        #     for j in range(i, pred_logi.shape[0]):
        #         wui = pred_logi[i]
        #         wuj = pred_logi[j]
                
        #         if self.same_row(wui, wuj):
        #             self.RC_allp += 1.0
        #         if self.same_col(wui, wuj):
        #             self.RC_allp += 1.0

        for i in range(gt_logi.shape[0]):
            for j in range(i + 1, gt_logi.shape[0]):
                sui = gt_logi[i]
                suj = gt_logi[j]
                
                if self.same_row(sui, suj):
                    self.RC_allt += 1.0
                if self.same_col(sui, suj):
                    self.RC_allt += 1.0
                
                if i in gt2pred[:, 0] and j in gt2pred[:, 0]:
                    # gt2pred[np.argwhere(gt2pred[:, 0] == i)[0], 1].item() ---》 检索 与target_i 匹配的 pred_i
                    tui = pred_logi[gt2pred[np.argwhere(gt2pred[:, 0] == i)[0], 1].item()]
                    tuj = pred_logi[gt2pred[np.argwhere(gt2pred[:, 0] == j)[0], 1].item()]
                    
                    # 根据匹配gt的pred框，整理出pred的relation数量，减小task1/2对task4指标评估的影响
                    if self.same_row(tui,tuj):
                        self.RC_allp += 1.0
                    if self.same_col(tui, tuj):
                        self.RC_allp += 1.0

                    if self.same_row(sui, suj) and self.same_row(tui, tuj):
                        self.RC_tp += 1.0
                    if self.same_col(sui, suj) and self.same_col(tui, tuj):
                        self.RC_tp += 1.0

    def cal_logical_location(self, pred_logi, gt_logi, gt2pred):  # For Logical Location AACC
        """
        Caculate number of true positive, predicted and ground truth logical locations
        
        Arguments:
        ----------
            pred_logi : np.ndarray(array[N, 4])
                predicted logical location of cells. N is the number of predicted cells.
            gt_logi : np.ndarray(array[M, 4])
                ground truth logical location of cells. M is the number of ground truth cells.
            gt2pred : torch.tensor(tensor[num of match, 2], dtype=int)
                gt cell id to pred cell id mapping.
        
        Returns:
        --------
            None
        """
        # self.LL_allt += len(gt_logi)
        # self.LL_allp += len(pred_logi)

        for i in range(gt_logi.shape[0]):
            if i in gt2pred[:, 0]:
                # gt中的cell在pred中找到了匹配, 只考虑匹配上的
                self.LL_allt += 1
                self.LL_allp += 1
                
                # gt2pred[np.argwhere(gt2pred[:, 0] == i)[0], 1].item() ---》 检索 与target_i 匹配的 pred_i
                pui = pred_logi[gt2pred[np.argwhere(gt2pred[:, 0] == i)[0], 1].item()]
                tui = gt_logi[i]
                # print(f'pred:{pui}, gt:{tui}')
                if (pui == tui).all():
                    self.LL_tp += 1
                if pui[0] == tui[0]:
                    self.LL_rb_tp += 1
                if pui[1] == tui[1]:
                    self.LL_re_tp += 1
                if pui[2] == tui[2]:
                    self.LL_cb_tp += 1
                if pui[3] == tui[3]:
                    self.LL_ce_tp += 1

    def cal_cell_type_classification(self, pred_cls, gt_cls, gt2pred):  # For Cell Type Classification F1
        """
        Caculate cell type classification F1
        
        Arguments:
        ----------
            pred_cls : List[N]
                list of predicted cell types. N is the number of predicted cells.
            gt_cls : List[M]
                list of ground truth cell types. M is the number of ground truth cells.
            gt2pred : torch.tensor(tensor[num of match, 2], dtype=int)
                ground truth cell id to predicted cell id mapping.
        
        Returns:
        --------
            None
        """
        for i in range(len(gt_cls)):
            gt_cls_matched, pred_cls_matched = [], []
            if i in gt2pred[:, 0]:
                gt_cls_matched.append(gt_cls[i])
                # gt2pred[np.argwhere(gt2pred[:, 0] == i)[0], 1].item() ---》 检索 与target_i 匹配的 pred_i
                pred_class = pred_cls[gt2pred[np.argwhere(gt2pred[:, 0] == i)[0], 1].item()]
                pred_cls_matched.append(pred_class)

        self.CTC_pred = self.CTC_pred + pred_cls_matched
        self.CTC_gt = self.CTC_gt + gt_cls_matched

    def cal_teds(self, pred_html, gt_html):
        """
        Caculate TEDS metric
        
        Args:
        ----------
            pred_html : str
                predicted html string.
            gt_html : str
                ground truth html string.
        
        Returns:
        --------
            None
        """
        self.TEDS.append(self.teds_cal.evaluate(pred_html, gt_html))

    ###############

    # Task 1/2
    def Cal_Bbox_Detection_F1(self):
        """
        Detection metric P, R, F1 for Task1/2
        """
        if self.BBD_allp == 0 or self.BBD_allt == 0:
            return 0., 0., 0.

        P = self.BBD_tp / self.BBD_allp
        R = self.BBD_tp / self.BBD_allt
        F1_score = 2 * P * R / (P + R + 1e-7)

        self.BBD_F1 = F1_score

        return P, R, F1_score

    # Task 3
    def Cal_Cell_Type_Classification_F1(self):
        """
        Classification metric macro F1 for Task3
        """
        return f1_score(self.CTC_gt, self.CTC_pred, average='macro')

    # Task 4
    def Cal_Relation_Classification_F1(self):
        """
        Relation classification P,R,F1 for Task4
        """
        if self.RC_allp == 0 or self.RC_allt == 0:
            return 0., 0., 0.

        P = self.RC_tp / self.RC_allp
        R = self.RC_tp / self.RC_allt
        F1_score = 2 * P * R / (P + R + 1e-7)

        self.RC_F1 = F1_score
        
        return P, R, F1_score

    def Cal_Logical_Location_ACC(self):
        """
        Logical location ACC for Task4
        """
        if self.LL_allp == 0 or self.LL_allt == 0:
            return 0.

        # P = self.LL_tp / self.LL_allp
        # R = self.LL_tp / self.LL_allt
        # F1_score = 2 * P * R / (P + R + 1e-7)
        # self.LL_F1 = F1_score
        
        # Acc_LL_rb = self.LL_rb_tp / self.LL_allp
        # Acc_LL_re = self.LL_re_tp / self.LL_allp
        # Acc_LL_cb = self.LL_cb_tp / self.LL_allp
        # Acc_LL_ce = self.LL_ce_tp / self.LL_allp
        Acc_LL = self.LL_tp / self.LL_allp
        self.LL_ACC = Acc_LL
        # res = f'Acc_LL_rb={Acc_LL_rb}, Acc_LL_re={Acc_LL_re}, Acc_LL_cb={Acc_LL_cb}, Acc_LL_ce={Acc_LL_ce}, Acc_LL={Acc_LL}, LL_tp={self.LL_tp}, LL_allp={self.LL_allp}, LL_allt={self.LL_allt}.\n'
        return Acc_LL

    def Cal_TEDS(self):
        """
        Calculate TEDS metric for Task4
        """
        if len(self.TEDS) == 0:
            return 0.
        else:
            return sum(self.TEDS) * 1.0 / len(self.TEDS)
        
    def Cal_AP(self):
        """
        Calculate AP metric for Task1/2
        """
        if self.BBD_allp == 0 and self.BBD_allt == 0:
            return 1.0
        elif self.BBD_allp == 0 and self.BBD_allt == 0:
            return 0.0
        
        # sort by score
        correct = self.correct_all[torch.sort(self.pred_score_all, descending=True)[1]]
        
        tp = correct
        fp = ~tp
        
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        
        precision = tp_cum / (tp_cum + fp_cum + 1e-7) # shape is num_pred
        recall = tp_cum / self.BBD_allt
        
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([1.0], precision, [0.0]))

        # Compute the precision envelope
        mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

        # Integrate area under curve
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

        return ap

    def run_one_step(self, preds, gts):  # 每次执行一个sample，累加各项指标
        """
        Running one example metric calculation
        Tasks:
        ttd: Text Region Detection		Detection F1 (IoU=0.5)
        acd: Aligned Cell Detection 	Detection F1 (IoU=0.5)
        ctc: Cell Type Classification	Classification F1 (Macro)
        tsr_rc/tsr_teds/tsr_ll: Table Structure Recognition		Relationship Classification F1, TEDS, Logical Location F1
        
        Args:
        ----------
            preds : dict
                predicted results
            gts : dict
                ground truth results
        
        Returns:
        --------
            None
        """
        pred_logi, gt_logi = preds['pred_lloc'], gts['gt_lloc']  # np.ndarray [num, 4]
        pred_bbox, gt_bbox = preds['pred_bbox'], gts['gt_bbox']  # np.ndarray [num, 8]
        
        # 判断是否检测结果为空
        if len(pred_bbox) == 0:
            pred_bbox = np.empty((0, 8))
            pred_logi = np.empty((0, 4))
        
        # check if bbox turing to 8 value type(four corners)
        assert pred_bbox.shape[1] == 8
        assert gt_bbox.shape[1] == 8        
        
        # for text region to align cell, need to change pred_logi
        # for align cell to text region, need to change gt_logi
        if self.t2a:
            correct, gt2pred, pred_logi = self.textregion_2_aligncell(pred_bbox, gt_bbox, pred_logi)
        elif self.a2t:
            correct, gt2pred, gt_logi = self.aligncell_2_textregion(pred_bbox, gt_bbox, gt_logi)
        else:
            correct, gt2pred = self.correct_martix(pred_bbox, gt_bbox)

        if 'cd' in self.cal_tasks:
            self.cal_bbox_detection(len(pred_bbox), len(gt_bbox), correct)

        if 'ctc' in self.cal_tasks:
            pred_cls, gt_cls = preds['pred_cls'], gts['gt_cls']  # list [num]
            self.cal_cell_type_classification(pred_cls, gt_cls, gt2pred)

        if 'tsr_rc' in self.cal_tasks:
            self.cal_relation_dection(pred_logi, gt_logi, gt2pred)

        if 'tsr_teds' in self.cal_tasks:
            pred_html, gt_html = preds['pred_html'], gts['gt_html'] # str
            self.cal_teds(pred_html, gt_html)

        if 'tsr_ll' in self.cal_tasks:
            self.cal_logical_location(pred_logi, gt_logi, gt2pred)

    def summary_for_final_results(self):
        
        res = dict()
        
        if 'cd' in self.cal_tasks:
            BBD_P, BBD_R, BBD_F1 = self.Cal_Bbox_Detection_F1()
            res['detect_f1'] = BBD_F1
            
            # self.correct_all = torch.cat(self.correct_all, 0)
            # self.pred_score_all = torch.cat(self.pred_score_all, 0)
            # AP = self.Cal_AP()
            # res['detect_ap'] = AP
            
            # print(f'Cell Detection:   Precision:{BBD_P}, Recall:{BBD_R}, F1:{BBD_F1}')

        if 'ctc' in self.cal_tasks:
            CLS_F1 = self.Cal_Cell_Type_Classification_F1()
            res['ctc_f1'] = CLS_F1
            # print(f'Cell Type Classification:  F1:{CLS_F1}')

        if 'tsr_rc' in self.cal_tasks:
            RC_P, RC_R, RC_F1 = self.Cal_Relation_Classification_F1()
            res['rel_f1'] = RC_F1
            # print(f'TSR-Relationship Classification:   Precision:{RC_P}, Recall:{RC_R}, F1:{RC_F1}')

        if 'tsr_teds' in self.cal_tasks:
            TEDS_total = self.Cal_TEDS()
            res['ted'] = TEDS_total
            # print(f'TSR-TEDS: {TEDS_total}')

        if 'tsr_ll' in self.cal_tasks:
            LL_ACC = self.Cal_Logical_Location_ACC()
            res['loc_acc'] = LL_ACC
            # print(f'TSR-Logical Location:  ACC:{LL_ACC}\n' + location_res)

        return res


def parallel_process(array, function, n_jobs=16, use_kwargs=False, front_num=0):
    """
        A parallel version of the map function with a progress bar.

        Args:
            array (array-like): An array to iterate over.
            function (function): A python function to apply to the elements of array
            n_jobs (int, default=16): The number of cores to use
            use_kwargs (boolean, default=False): Whether to consider the elements of array as dictionaries of
                keyword arguments to function
            front_num (int, default=3): The number of iterations to run serially before kicking off the parallel job.
                Useful for catching bugs
        Returns:
            [function(array[0]), function(array[1]), ...]
    """
    # We run the first few iterations serially to catch bugs
    if front_num > 0:
        front = [function(**a) if use_kwargs else function(a) for a in array[:front_num]]
    else:
        front = []

    # If we set n_jobs to 1, just run a list comprehension. This is useful for benchmarking and debugging.
    if n_jobs == 1:
        return front + [function(**a) if use_kwargs else function(a) for a in tqdm(array[front_num:])]
    # Assemble the workers
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        # Pass the elements of array into function
        if use_kwargs:
            futures = [pool.submit(function, **a) for a in array[front_num:]]
        else:
            futures = [pool.submit(function, a) for a in array[front_num:]]
        kwargs = {
            'total': len(futures),
            'unit': 'it',
            'unit_scale': True,
            'leave': True
        }
        # Print out the progress as tasks complete
        for f in tqdm(as_completed(futures), **kwargs):
            pass
    out = []
    # Get the results from the futures.
    for i, future in tqdm(enumerate(futures)):
        try:
            out.append(future.result())
        except Exception as e:
            out.append(e)
    return front + out


class TableTree(Tree):
    def __init__(self, tag, colspan=None, rowspan=None, content=None, *children):
        self.tag = tag
        self.colspan = colspan
        self.rowspan = rowspan
        self.content = content
        self.children = list(children)

    def bracket(self):
        """Show tree using brackets notation"""
        if self.tag == 'td':
            result = '"tag": %s, "colspan": %d, "rowspan": %d, "text": %s' % \
                     (self.tag, self.colspan, self.rowspan, self.content)
        else:
            result = '"tag": %s' % self.tag
        for child in self.children:
            result += child.bracket()
        return "{{{}}}".format(result)


class CustomConfig(Config):
    @staticmethod
    def maximum(*sequences):
        """Get maximum possible value
        """
        return max(map(len, sequences))

    def normalized_distance(self, *sequences):
        """Get distance from 0 to 1
        """
        return float(distance.levenshtein(*sequences)) / self.maximum(*sequences)

    def rename(self, node1, node2):
        """Compares attributes of trees"""
        if (node1.tag != node2.tag) or (node1.colspan != node2.colspan) or (node1.rowspan != node2.rowspan):
            return 1.
        if node1.tag == 'td':
            if node1.content or node2.content:
                return self.normalized_distance(node1.content, node2.content)
        return 0.


class TEDS(object):
    ''' Tree Edit Distance basead Similarity
    '''
    def __init__(self, structure_only=False, n_jobs=1, ignore_nodes=None):
        assert isinstance(n_jobs, int) and (n_jobs >= 1), 'n_jobs must be an integer greather than 1'
        self.structure_only = structure_only
        self.n_jobs = n_jobs
        self.ignore_nodes = ignore_nodes
        self.__tokens__ = []

    def tokenize(self, node):
        ''' Tokenizes table cells
        '''
        self.__tokens__.append('<%s>' % node.tag)
        if node.text is not None:
            self.__tokens__ += list(node.text)
        for n in node.getchildren():
            self.tokenize(n)
        if node.tag != 'unk':
            self.__tokens__.append('</%s>' % node.tag)
        if node.tag != 'td' and node.tail is not None:
            self.__tokens__ += list(node.tail)

    def load_html_tree(self, node, parent=None):
        ''' Converts HTML tree to the format required by apted
        '''
        global __tokens__
        if node.tag == 'td':
            if self.structure_only:
                cell = []
            else:
                self.__tokens__ = []
                self.tokenize(node)
                cell = self.__tokens__[1:-1].copy()
            new_node = TableTree(node.tag,
                                 int(node.attrib.get('colspan', '1')),
                                 int(node.attrib.get('rowspan', '1')),
                                 cell, *deque())
        else:
            new_node = TableTree(node.tag, None, None, None, *deque())
        if parent is not None:
            parent.children.append(new_node)
        if node.tag != 'td':
            for n in node.getchildren():
                self.load_html_tree(n, new_node)
        if parent is None:
            return new_node

    def evaluate(self, pred, true):
        ''' Computes TEDS score between the prediction and the ground truth of a
            given sample
        '''
        if (not pred) or (not true):
            return 0.0
        parser = html.HTMLParser(remove_comments=True, encoding='utf-8')
        pred = html.fromstring(pred, parser=parser)
        true = html.fromstring(true, parser=parser)
        if pred.xpath('body/table') and true.xpath('body/table'):
            pred = pred.xpath('body/table')[0]
            true = true.xpath('body/table')[0]
            if self.ignore_nodes:
                etree.strip_tags(pred, *self.ignore_nodes)
                etree.strip_tags(true, *self.ignore_nodes)
            n_nodes_pred = len(pred.xpath(".//*"))
            n_nodes_true = len(true.xpath(".//*"))
            n_nodes = max(n_nodes_pred, n_nodes_true)
            tree_pred = self.load_html_tree(pred)
            tree_true = self.load_html_tree(true)
            distance = APTED(tree_pred, tree_true, CustomConfig()).compute_edit_distance()
            return 1.0 - (float(distance) / n_nodes)
        else:
            return 0.0

    def batch_evaluate(self, pred_json, true_json):
        ''' Computes TEDS score between the prediction and the ground truth of
            a batch of samples
            @params pred_json: {'FILENAME': 'HTML CODE', ...}
            @params true_json: {'FILENAME': {'html': 'HTML CODE'}, ...}
            @output: {'FILENAME': 'TEDS SCORE', ...}
        '''
        samples = true_json.keys()
        if self.n_jobs == 1:
            scores = [self.evaluate(pred_json.get(filename, ''), true_json[filename]['html']) for filename in tqdm(samples)]
        else:
            inputs = [{'pred': pred_json.get(filename, ''), 'true': true_json[filename]['html']} for filename in samples]
            scores = parallel_process(inputs, self.evaluate, use_kwargs=True, n_jobs=self.n_jobs, front_num=1)
        scores = dict(zip(samples, scores))
        return scores


if __name__ == '__main__':
    '''
    def test(args):
        model = model.load(ckpt)
        evaluator = Evaluator(args)
        batches = DataLoader(args.data_path, col_fn=col_fn)

        for data in batches:
            gts = data['label']
            preds = model(data)
            evaluator.run_one_step(preds, gts)

        res = evaluator.summary_for_final_results()
        print(res)
    '''
