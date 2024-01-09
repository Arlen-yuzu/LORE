from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import re
import argparse

import numpy as np
from tqdm import tqdm

from lib.utils.eval_utils import coco_into_labels, Table, pairTab
from Evaluator import Evaluator

def load_res(predict_dir, bbox_path, logi_path, cls_path, html_path=None):
    res_bboxs, res_logis, res_clses, res_htmls = [], [], [], [] # num of pics
    for file_name in tqdm(os.listdir(os.path.join(predict_dir, 'gt_center'))):
        if 'txt' in file_name:
            bbox_file = os.path.join(bbox_path, file_name)
            logi_file = os.path.join(logi_path, file_name)
            cls_file = os.path.join(cls_path, file_name)
            if html_path is not None:
                html_file = os.path.join(html_path, file_name)
            
            bboxs = []
            logis = []
            clses = []
            htmls = None
            with open(bbox_file) as f_b:
                bboxs = f_b.readlines()
            with open(logi_file) as f_l:
                logis = f_l.readlines()
            with open(cls_file) as f_c:
                clses = f_c.readlines()
            if html_path is not None:
                with open(html_file) as f_h:
                    htmls = f_h.read()
                
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
            
            res_bboxs.append((np.array(res_bboxs_single)))
            res_logis.append((np.array(res_logis_single)))
            res_clses.append(((res_clses_single)))
            res_htmls.append(htmls)
    if html_path is not None:
        return res_bboxs, res_logis, res_clses, res_htmls
    else:
        return res_bboxs, res_logis, res_clses

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type = str)
    parser.add_argument('--predict_dir', type = str)
    parser.add_argument('--dataset_name', type = str, default='icdar13')
    parser.add_argument('--iou', type = float, default=0.5)
    args = parser.parse_args()
    
    # make sure the tasks the chosen dataset sppports
    if args.dataset_name == 'ptn':
        cal_tasks = ['acd', 'tsr_rc', 'tsr_teds', 'tsr_ll']
    elif args.dataset_name == 'comfintab':
        cal_tasks = ['acd', 'ctc', 'tsr_rc', 'tsr_ll']
    else:
        cal_tasks = ['acd', 'tsr_rc', 'tsr_ll']

    if args.dataset_name == 'ptn':
        need_teds = True
    else:
        need_teds = False
    coco_into_labels(args.dataset_dir, args.predict_dir, need_teds)

    gt_bbox_path = os.path.join(args.predict_dir, 'gt_center')
    gt_logi_path = os.path.join(args.predict_dir, 'gt_logi')
    gt_cls_path = os.path.join(args.predict_dir, 'gt_cls')
    gt_html_path = None
    if args.dataset_name == 'ptn':
        gt_html_path = os.path.join(args.predict_dir, 'gt_html')
    
    bbox_path = os.path.join(args.predict_dir, 'center')
    logi_path = os.path.join(args.predict_dir, 'logi')
    cls_path = os.path.join(args.predict_dir, 'cls')
    html_path = None
    if args.dataset_name == 'ptn':
        html_path = os.path.join(args.predict_dir, 'html')
    
    if args.dataset_name == 'ptn':
        gt_bbox, gt_logi, gt_clse, gt_htmls = load_res(args.predict_dir, gt_bbox_path, gt_logi_path, gt_cls_path, gt_html_path)
        pred_bbox, pred_logi, pred_cls, pred_htmls = load_res(args.predict_dir, bbox_path, logi_path, cls_path, html_path)
    else:
        gt_bbox, gt_logi, gt_clse = load_res(args.predict_dir, gt_bbox_path, gt_logi_path, gt_cls_path)
        pred_bbox, pred_logi, pred_cls = load_res(args.predict_dir, bbox_path, logi_path, cls_path)
    
    evaluator = Evaluator(cal_tasks, text_match_aligh=True)
    
    for i in range(len(pred_bbox)):
        pred_bbox_single = pred_bbox[i]
        pred_logi_single = pred_logi[i]
        pred_cls_single = pred_cls[i]
        preds = {'pred_bbox': pred_bbox_single, 'pred_lloc': pred_logi_single, 'pred_cls': pred_cls_single}
        
        gt_bbox_single = gt_bbox[i]
        gt_logi_single = gt_logi[i]
        gt_cls_single = gt_clse[i]
        gts = {'gt_bbox': gt_bbox_single, 'gt_lloc': gt_logi_single, 'gt_cls': gt_cls_single}
        
        if args.dataset_name == 'ptn':
            pred_html_single = pred_htmls[i]
            gt_html_single = gt_htmls[i]
            preds['pred_html'] = pred_html_single
            gts['gt_html'] = gt_html_single
        
        evaluator.run_one_step(preds, gts)
    
    res = evaluator.summary_for_final_results()
    print(res)
