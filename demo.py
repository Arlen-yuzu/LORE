# 原始版本：表格图输入，仅返回tsr结果和保存用于后续指标计算的txt文件
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
from tqdm import tqdm
import numpy as np

import cv2
import pycocotools.coco as coco
from opts import opts
from detectors.detector_factory import detector_factory
from utils.utils import show_results
image_ext = ['jpg', 'jpeg', 'png', 'webp', 'bmp', 'tiff']

def demo(opt):
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.debug = max(opt.debug, 1)
  Detector = detector_factory[opt.task]
  detector = Detector(opt)
  
  if not opt.anno_path == '':

    image_names = []
    image_annos = []
    coco_data = coco.COCO(opt.anno_path)
    images = coco_data.getImgIds()
  
    for i in range(len(images)):
      img_id = images[i]
      if opt.dataset_name == 'WTW':
        file_name = coco_data.loadImgs(ids=[img_id])[0]['file_name']
      elif opt.dataset_name == 'PTN':
        file_name = coco_data.loadImgs(ids=[img_id])[0]['file_name'].replace('.jpg', '.png')
      else:
        file_name = coco_data.loadImgs(ids=[img_id])[0]['file_name']

      ann_ids = coco_data.getAnnIds(imgIds=[img_id])
      anns = coco_data.loadAnns(ids=ann_ids)

      image_names.append(os.path.join(opt.demo, file_name))
      image_annos.append(anns)
  
  elif os.path.isdir(opt.demo):

    image_names = []
    ls = os.listdir(opt.demo)
    for file_name in sorted(ls):
        ext = file_name[file_name.rfind('.') + 1:].lower()
        if ext in image_ext:
            image_names.append(os.path.join(opt.demo, file_name))
  else:

    image_names = [opt.demo]

  # 保存结果以用于指标计算用
  if not os.path.exists(opt.output_dir + opt.demo_name):
      os.makedirs(opt.output_dir + opt.demo_name +'/center/')
      # os.makedirs(opt.output_dir + opt.demo_name +'/corner/')
      os.makedirs(opt.output_dir + opt.demo_name +'/logi/')
      os.makedirs(opt.output_dir + opt.demo_name +'/cls/')
      os.makedirs(opt.output_dir + opt.demo_name +'/score/')

  if not os.path.exists(opt.demo_dir):
    os.makedirs(opt.demo_dir)

  for i in tqdm(range(len(image_names))):
      image_name = image_names[i]
      if not opt.wiz_detect:
        # 用于只衡量logic regression的效果
        image_anno = image_annos[i]
        ret = detector.run(opt, image_name, image_anno)
      else:
        #image_anno = image_annos[i]
        #print(image_name)
      
        ret = detector.run(opt, image_name)

        spatial = ret['spatial'] # [num_valid, 8]
        logi = ret['logi'] # [num_valid, 4]
        cell_cls = ret['cls'] # [num_valid]
        cell_score = ret['score'] # [num_valid]

        image = cv2.imread(image_name)
        image_name = image_name.split('/')[-1]
        show_results(image, spatial, image_name, logi, cell_cls, cell_score, opt)


if __name__ == '__main__':
  opt = opts().init()
  demo(opt)
