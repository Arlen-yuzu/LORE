# 从全图根据TD坐标切割出table区域，然后进行TSR
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
from tqdm import tqdm

import numpy as np
import cv2
import pycocotools.coco as coco
import xml.etree.ElementTree as ET
from opts import opts
from detectors.detector_factory import detector_factory
from utils.eval_utils import TabUnit
from utils.html_utils import json2html, cells_row_convert

image_ext = ['jpg', 'jpeg', 'png', 'webp', 'bmp', 'tiff']
color_list = np.array(
        [
            1.000, 1.000, 1.000,
            0.850, 0.325, 0.098,
            0.929, 0.694, 0.125,
        ]).astype(np.float32)
color_list = color_list.reshape((-1, 3)) * 255


def add_4ps_coco_bbox(img, bbox, cat, logi):
    bbox = np.array(bbox, dtype=np.int32)
    cat = int(cat)

    colors = [(color_list[_]).astype(np.uint8) \
            for _ in range(len(color_list))]
    colors = np.array(colors, dtype=np.uint8).reshape(len(colors), 1, 1, 3)
    colors = colors.reshape(-1)[::-1].reshape(len(colors), 1, 1, 3)
    colors = np.clip(colors, 0., 0.6 * 255).astype(np.uint8)
    c = colors[cat][0][0].tolist()
    c = (255 - np.array(c)).tolist()
    
    if not logi is None:
      txt = '{:.0f},{:.0f},{:.0f},{:.0f}'.format(logi[0], logi[1], logi[2], logi[3])
    font = cv2.FONT_HERSHEY_SIMPLEX
    cat_size = cv2.getTextSize(txt, font, 0.3, 2)[0]
    cv2.line(img,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,0,255),2)
    cv2.line(img,(bbox[2],bbox[3]),(bbox[4],bbox[5]),(0,0,255),2)
    cv2.line(img,(bbox[4],bbox[5]),(bbox[6],bbox[7]),(0,0,255),2)
    cv2.line(img,(bbox[6],bbox[7]),(bbox[0],bbox[1]),(0,0,255),2) # 1 - 5
  
    if not logi is None:
      cv2.rectangle(img,
                    (bbox[0], bbox[1] - cat_size[1] - 2),
                    (bbox[0] + cat_size[0], bbox[1] - 2), c, -1)
      cv2.putText(img, txt, (bbox[0], bbox[1] - 2), 
                  font, 0.30, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA) #1 - 5 # 0.20 _ 0.60


def show_results(image, results, image_name, logi, opt):
    for j in range(1, opt.num_classes + 1):
      k = 0
      for m in range(len(results[j])):
        bbox = results[j][m]
        k = k + 1
        if bbox[8] > opt.vis_thresh:
       
          if len(logi.shape) == 1:
            add_4ps_coco_bbox(image, bbox[:8], j-1, logi)
          else:
            add_4ps_coco_bbox(image, bbox[:8], j-1, logi[m,:])
    
    # 保存bbox框+逻辑坐标到原图
    cv2.imwrite(os.path.join(opt.demo_dir, image_name), image)


def demo(opt):
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.debug = max(opt.debug, 1)
  Detector = detector_factory[opt.task]
  detector = Detector(opt)

  if not os.path.exists(opt.demo_dir):
    os.makedirs(opt.demo_dir)
  
  if os.path.isdir(opt.demo):
    TD_label_dir = os.path.join(opt.demo, 'labels')
    image_dir = os.path.join(opt.demo, 'images')
    assert os.path.isdir(TD_label_dir), 'Error: {} is not a directory'.format(TD_label_dir)
    assert os.path.isdir(image_dir), 'Error: {} is not a directory'.format(image_dir)

    ls = os.listdir(TD_label_dir)
    for file_name in tqdm(sorted(ls)):
        TD_label_path = os.path.join(TD_label_dir, file_name)
        # 解析标注表格区域的XML文件
        tree = ET.parse(TD_label_path)
        root = tree.getroot()

        # 获取对应图像文件名
        image_filename = root.find("filename").text
        ext = image_filename[image_filename.rfind('.') + 1:].lower()
        if ext not in image_ext:
            print('Ignore invalid image file: {}'.format(image_filename))
            continue
        
        # 读取图像，根据table的坐标，将图像切割为table区域
        image_path = os.path.join(image_dir, image_filename)
        image = cv2.imread(image_path)

        for i, obj in enumerate(root.findall("object")):
            # 获取对象的边界框信息
            bbox = obj.find("bndbox")
            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)
            
            # 根据边界框坐标裁剪图像
            table_region = image[ymin:ymax, xmin:xmax]

            ret = detector.run(opt, table_region)
            results = ret['4ps'] # dict: key表示目标类别，value: [K, 9]
            slct_logi = ret['logi'] # [batch, num_valid, 4]

            for k, v in results.items():
               results[k][..., :8] += [xmin, ymin] * 4

            show_results(image, results, image_filename, slct_logi.squeeze(), opt)





if __name__ == '__main__':
  opt = opts().init()
  demo(opt)
