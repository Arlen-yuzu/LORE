import re
import os
import numpy as np
import scipy.spatial
from tqdm import tqdm
import jsonlines


import pycocotools.coco as coco
from glob import glob

def coco_into_labels(annot_path, label_path, need_teds=False):
    #annot_path = 'rev_scitsr_st_full/test_full.json'
    coco_data = coco.COCO(annot_path)
    images = coco_data.getImgIds()

    gt_center_dir = '{}/gt_center/'.format(label_path)
    gt_logi_dir = '{}/gt_logi/'.format(label_path)
    gt_cls_dir = '{}/gt_cls/'.format(label_path)        
    
    if not os.path.exists(gt_center_dir):
        os.mkdir(gt_center_dir)

    if not os.path.exists(gt_logi_dir):
        os.mkdir(gt_logi_dir)
    
    if not os.path.exists(gt_cls_dir):
        os.mkdir(gt_cls_dir)

    print('Changing COCO Labels into TXT Labels...')
    for i in tqdm(range(len(images))):
        img_id = images[i]
        file_name = coco_data.loadImgs(ids=[img_id])[0]['file_name']
        #file_names.append(file_name)

        # using this for your dataset
        center_file = '{}/gt_center/'.format(label_path) + file_name +'.txt'
        logi_file = '{}/gt_logi/'.format(label_path) + file_name +'.txt'
        cls_file = '{}/gt_cls/'.format(label_path) + file_name +'.txt'
        # if need_teds:
        #     html_file = '{}/gt_html/'.format(label_path) + file_name +'.txt'

        #TODO: revise the file names in the annotation of PubTabNet
        # center_file = gt_center_dir + file_name.replace('.jpg', '.png') +'.txt'
        # logi_file = gt_logi_dir + file_name.replace('.jpg', '.png') +'.txt'
        # cls_file = gt_cls_dir + file_name.replace('.jpg', '.png') +'.txt'
        
        
        ann_ids = coco_data.getAnnIds(imgIds=[img_id])
        anns = coco_data.loadAnns(ids=ann_ids)
        
        fc = open(center_file, 'w')
        fl = open(logi_file, 'w')
        fcls = open(cls_file, 'w')
        for j in range(len(anns)):
            ann = anns[j]
            bbox = ann['segmentation'][0]
            logi = ann['logic_axis'][0]
            cls_kv = ann['category_id']
            for i in range(0,3):
                fc.write(str(bbox[2*i])+','+str(bbox[2*i+1])+';')
                fl.write(str(int(logi[i]))+',')
                        
            fc.write(str(bbox[6])+','+str(bbox[7])+'\n')
            fl.write(str(int(logi[3]))+'\n')
            fcls.write(str(cls_kv)+'\n')
        fc.close()
        fl.close()
        fcls.close()
        
        # if need_teds:
            
        #     with open(html_file, 'w') as fhtml:
        #         fhtml.write()
        
    print('Finished: Changing COCO Labels into TXT Labels!')

class pairTab():
    def __init__(self, pred_table, gt_table):
        self.gt_list = gt_table.ulist
        self.pred_list = pred_table.ulist
        
        self.match_list = []
        self.matching()
        
    def matching(self):
        for tunit in self.gt_list:
            if_find = 0
            for sunit in self.pred_list:
                #TODO: Adding Parameters for IOU threshold
                #Using IOU=0.5 as Default
                if self.compute_IOU(tunit.bbox, sunit.bbox) >= 0.5:
                    self.match_list.append(sunit)
                    if_find = 1
                    break
            if if_find == 0:
                self.match_list.append('empty')
    
    def evalBbox(self, eval_type):
        tp = 0
        for u in self.match_list:
            if u != 'empty':
                tp = tp + 1.0
        
        ap = len(self.pred_list)
        at = len(self.gt_list)
        if eval_type == 'recall':
            if at == 0:
                return 'null'
            else:
                return tp/at
        elif eval_type == 'precision':
            if ap == 0:
                return 'null'
            else:
                return tp/ap

    def compute_IOU(self, bbox1,bbox2):
        rec1 = (bbox1.point1[0][0], bbox1.point1[0][1], bbox1.point3[0][0], bbox1.point3[0][1])
        rec2 = (bbox2.point1[0][0], bbox2.point1[0][1], bbox2.point3[0][0], bbox2.point3[0][1])

        left_column_max = max(rec1[0],rec2[0])
        right_column_min = min(rec1[2],rec2[2])
        up_row_max = max(rec1[1],rec2[1])
        down_row_min = min(rec1[3],rec2[3])
    
        if left_column_max>=right_column_min or down_row_min<=up_row_max:
            return 0
        else:
            S1 = (rec1[2]-rec1[0])*(rec1[3]-rec1[1])
            S2 = (rec2[2]-rec2[0])*(rec2[3]-rec2[1])
            S_cross = (down_row_min-up_row_max)*(right_column_min-left_column_max)

            return S_cross/(S1+S2-S_cross)
    
    def evalAxis(self):
        
        tp = 0
        for u in self.match_list:
            if u != 'empty':
                tp = tp + 1.0
                
        truep = 0
        for i in range(len(self.gt_list)):
            sunit = self.match_list[i]
            if sunit != 'empty':
                tunit = self.gt_list[i]
                
                saxis = sunit.axis
                taxis= tunit.axis
                
                flag = 1
                for j in range(4):
                    if saxis[j] != taxis[j]:
                        flag = 0
                        break
                
                #all four axis are correctly predicted
                if flag == 1:
                    truep = truep + 1.0

        if len(self.gt_list) == 0:
            #return 0
            return 'null'
        else:
            if tp == 0:
                #return 0
                return 'null'
            else:
                return truep/tp
    
    def same_col(self, unit1, unit2):
        if unit1.axis[2] <= unit2.axis[2] <= unit1.axis[3]:
            if ((unit1.axis[0] - unit2.axis[1]) == 1) or ((unit2.axis[0] - unit1.axis[1]) == 1):
                return True
            else:
                return False
        elif unit2.axis[2] <= unit1.axis[2] <= unit2.axis[3]:
            if ((unit1.axis[0] - unit2.axis[1]) == 1) or ((unit2.axis[0] - unit1.axis[1]) == 1):
                return True
            else:
                return False
        else:
            return False
        
    def same_row(self, unit1, unit2):
        if unit1.axis[0] <= unit2.axis[0] <= unit1.axis[1]:
            if ((unit1.axis[2] - unit2.axis[3]) == 1) or ((unit2.axis[2] - unit1.axis[3]) == 1):
                return True
            else:
                return False
        elif unit2.axis[0] <= unit1.axis[0] <= unit2.axis[1]:
            if ((unit1.axis[2] - unit2.axis[3]) == 1) or ((unit2.axis[2] - unit1.axis[3]) == 1):
                return True
            else:
                return False
        else:
            return False
    
    def evalRelation(self, eval_type):
        tp = 0   # num of predict true relations
        allt = 0 # num of label relations
        allp = 0 # num of predicted relations
        
        for i in range(len(self.pred_list)):
            for j in range(i, len(self.pred_list)):
                wui = self.pred_list[i]
                wuj = self.pred_list[j]
                
                if self.same_row(wui, wuj):
                    allp = allp + 1.0
                     
                if self.same_col(wui, wuj):
                    allp = allp + 1.0
        
        for i in range(len(self.gt_list)):
            for j in range(i, len(self.gt_list)):
                sui = self.gt_list[i]
                suj = self.gt_list[j]
                
                tui = self.match_list[i]
                tuj = self.match_list[j]
                
                if self.same_row(sui, suj):
                    allt = allt + 1.0

                if self.same_col(sui, suj):
                    allt = allt + 1.0
                        
                if tui != 'empty' and tuj != 'empty':   
                    if self.same_row(tui, tuj):
                        allp = allp + 1.0

                    if self.same_col(tui, tuj):
                        allp = allp + 1.0
                    
                    if self.same_row(sui, suj):
                        if self.same_row(tui, tuj):
                            tp = tp + 1.0

                    if self.same_col(sui, suj):
                        if self.same_col(tui, tuj):
                            tp = tp + 1.0

        if eval_type == 'precision':
            if allp == 0:
                return 'null'
            else:
                return tp/allp
        elif eval_type == 'recall':
            if allt == 0:
                return 'null'
            else:
                return tp/allt
    
class Table():
    def __init__(self, bbox_path, axis_path, cls_path, file_name):
        self.bbox_dir = os.path.join(bbox_path, file_name) 
        self.axis_dir = os.path.join(axis_path, file_name)
        self.cls_dir = os.path.join(cls_path, file_name) 
        
        self.ulist = []
        self.load_tabu(self.bbox_dir, self.axis_dir, self.cls_dir)
        self.ulist = self.bubble_sort(self.ulist)
    
    def load_tabu(self, bbox_dir, axis_dir, cls_dir):
        
        f_b = open(self.bbox_dir)
        f_a = open(self.axis_dir)
        f_c = open(self.cls_dir)
        bboxs = f_b.readlines()
        axiss = f_a.readlines()
        clses = f_c.readlines()
        
        for bbox, axis, clskv in zip(bboxs, axiss, clses):
            bbox = list(map(float, re.split(';|,',bbox.strip())))
            axis = list(map(int, axis.strip().split(',')))
            clskv = int(clskv.strip())
            unit = TabUnit_eval(bbox, axis, clskv)

            self.ulist.append(unit)
        f_b.close()
        f_a.close()
        f_c.close()
    
    def compute_IOU(self, bbox1, bbox2):
        rec1 = (bbox1.point1[0][0], bbox1.point1[0][1], bbox1.point3[0][0], bbox1.point3[0][1])
        rec2 = (bbox2.point1[0][0], bbox2.point1[0][1], bbox2.point3[0][0], bbox2.point3[0][1])
        left_column_max = max(rec1[0],rec2[0])
        right_column_min = min(rec1[2],rec2[2])
        up_row_max = max(rec1[1],rec2[1])
        down_row_min = min(rec1[3],rec2[3])
        
        if left_column_max>=right_column_min or down_row_min<=up_row_max:
            return 0
        else:
            S1 = (rec1[2]-rec1[0])*(rec1[3]-rec1[1])
            S2 = (rec2[2]-rec2[0])*(rec2[3]-rec2[1])
            S_cross = (down_row_min-up_row_max)*(right_column_min-left_column_max)
            return S_cross/(S1+S2-S_cross)
        
    def bubble_sort(self, unit_list):
        length = len(unit_list)
        for index in range(length):
            for j in range(1, length-index):
                if self.is_priori(unit_list[j], unit_list[j-1]):
                    unit_list[j-1], unit_list[j] = unit_list[j], unit_list[j-1]
        return unit_list
    
    def is_priori(self, unit_a, unit_b):
        if unit_a.top_idx < unit_b.top_idx :
            return True
        elif unit_a.top_idx > unit_b.top_idx :
            return False
        if unit_a.left_idx < unit_b.left_idx :
            return True
        elif unit_a.left_idx > unit_b.left_idx :
            return False
        if unit_a.bottom_idx < unit_b.bottom_idx :
            return True
        elif unit_a.bottom_idx > unit_b.bottom_idx :
            return False
        if unit_a.right_idx < unit_b.right_idx :
            return True
        elif unit_a.right_idx > unit_b.right_idx :
            return False 

class TabUnit():
    def __init__(self, bbox, axis, 
                 cls_cell, score, text=None
                 ):
        
        self.bbox = BBox(bbox)
        self.cls_cell = cls_cell
        self.score = score
        self.text = text
        self.axis = axis
        self.top_idx = int(axis[0])
        self.bottom_idx = int(axis[1])
        self.left_idx = int(axis[2])
        self.right_idx = int(axis[3])
        # self.top_idx = axis[2]
        # self.bottom_idx = axis[3]
        # self.left_idx = axis[0]
        # self.right_idx = axis[1]

class TabUnit_eval():
    def __init__(self, bbox, axis, cls_cell):
        
        self.bbox = BBox(bbox)
        self.axis = axis
        self.cls_cell = cls_cell
        self.top_idx = int(axis[0])
        self.bottom_idx = int(axis[1])
        self.left_idx = int(axis[2])
        self.right_idx = int(axis[3])
        # self.top_idx = axis[2]
        # self.bottom_idx = axis[3]
        # self.left_idx = axis[0]
        # self.right_idx = axis[1]
        
class BBox():
    def __init__(self, bbox):
        self.point1 = np.array([[bbox[0], bbox[1]]])
        self.point2 = np.array([[bbox[2], bbox[3]]])
        self.point3 = np.array([[bbox[4], bbox[5]]])
        self.point4 = np.array([[bbox[6], bbox[7]]])
        
        self.col_span = (self.computing_span(self.point1, self.point2) + self.computing_span(self.point3, self.point4))/2
        self.row_span = (self.computing_span(self.point1, self.point3) + self.computing_span(self.point2, self.point4))/2
        
    def computing_span(self, pointa, pointb):
        span = scipy.spatial.distance.cdist(pointa, pointb, metric = "euclidean")
        return span
 
    
  
 
  