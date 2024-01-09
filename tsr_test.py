# 原始版本：表格图输入，仅返回tsr结果和保存用于后续指标计算的txt文件
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
from tqdm import tqdm
import numpy as np

import cv2
import jsonlines
import pycocotools.coco as coco
from opts import opts
from detectors.detector_factory import detector_factory
from res_visual import show_results, show_results_with_gt
from utils.html_utils import res2html_teds, format_html

from cal_tasks import model_dataset_info
from Evaluator import Evaluator

image_ext = ['jpg', 'jpeg', 'png', 'webp', 'bmp', 'tiff']

def demo(opt):
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.debug = max(opt.debug, 1)
  Detector = detector_factory[opt.task]
  detector = Detector(opt)
  
  image_names = []
  image_annos = []
  coco_data = coco.COCO(opt.anno_path)
  images = coco_data.getImgIds()
  teds_ann = []
  
  if opt.dataset_name == 'pubtabnet':
    teds = {}
    with jsonlines.open('/data/xuyilun/project/LORE-TSR/data/pubtabnet_lore/PubTabNet_2.0.0_val.jsonl', 'r') as f:
      for tab in f:
        teds[tab['filename']] = format_html(tab)

  for i in range(len(images)):
    img_id = images[i]
    file_name = coco_data.loadImgs(ids=[img_id])[0]['file_name']

    ann_ids = coco_data.getAnnIds(imgIds=[img_id])
    anns = coco_data.loadAnns(ids=ann_ids)

    image_names.append(os.path.join(opt.img_dir, file_name))
    image_annos.append(anns)
    if opt.dataset_name == 'pubtabnet':
      teds_ann.append(teds[file_name])

  # # 保存结果以用于指标计算用
  # if not os.path.exists(opt.output_dir + opt.demo_name):
  #   os.makedirs(opt.output_dir + opt.demo_name +'/center/')
  #   os.makedirs(opt.output_dir + opt.demo_name +'/logi/')
  #   os.makedirs(opt.output_dir + opt.demo_name +'/cls/')
  #   os.makedirs(opt.output_dir + opt.demo_name +'/html/')
  #   os.makedirs(opt.output_dir + opt.demo_name +'/gt_html/')
  
  if not os.path.exists(opt.vis_dir):
    os.makedirs(opt.vis_dir)
  
  cal_tasks = model_dataset_info[opt.dataset_name]['cal_tasks']
  evaluator = Evaluator(cal_tasks, match_type=model_dataset_info[opt.dataset_name]['match'])

  for i in tqdm(range(len(image_names))):
    image_name = image_names[i]
    
    # # debug
    # if image_name.split('/')[-1] != 'cTDaR_t00748_2.jpg':
    #   continue
    
    image_anno = image_annos[i]
    if not opt.wiz_detect:
      # 用于只衡量logic regression的效果
      ret = detector.run(opt, image_name, image_anno)
    else:
      ret = detector.run(opt, image_name)

    spatial = ret['spatial'] # [num_valid, 8]
    logi = ret['logi'] # [num_valid, 4]
    cell_cls = ret['cls'] # [num_valid]
    
    if opt.dataset_name == 'pubtabnet':
      pred_html = res2html_teds(spatial, logi, cell_cls) # str
    
    image = cv2.imread(image_name)
    image_name = image_name.split('/')[-1]
    show_results(image_name, image, spatial, logi, opt)
    
    spatial_gt = []
    for j in range(len(image_anno)):
      spatial_gt.append(image_anno[j]['segmentation'][0])
    spatial_gt = np.array(spatial_gt)
    # show_results_with_gt(image_name, image, spatial, spatial_gt, opt)
    
    pred_dict = dict()
    pred_dict['pred_bbox'] = np.array(spatial)
    pred_dict['pred_lloc'] = np.array(logi)
    pred_dict['pred_cls'] = cell_cls
    if opt.dataset_name == 'pubtabnet':
      pred_dict['pred_html'] = pred_html
    
    gt_dict = dict()
    gt_dict['gt_bbox'] = np.array([image_anno[j]['segmentation'][0] for j in range(len(image_anno))])
    gt_dict['gt_lloc'] = np.array([image_anno[j]['logic_axis'][0] for j in range(len(image_anno))])
    gt_dict['gt_cls'] = [image_anno[j]['category_id'] for j in range(len(image_anno))]
    if opt.dataset_name == 'pubtabnet':
        gt_dict['gt_html'] = teds_ann[i]
    
    evaluator.run_one_step(pred_dict, gt_dict)
    
    # # debug
    # if image_name == 'cTDaR_t00748_2.jpg':
    #   break
    
  res = evaluator.summary_for_final_results()
  print(res)

if __name__ == '__main__':
  opt = opts().init()
  demo(opt)
