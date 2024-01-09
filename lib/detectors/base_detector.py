from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import numpy as np
from progress.bar import Bar
import torch

from models.model import create_model, load_model
from models.classifier import Processor, load_processor
from utils.utils import make_batch
from utils.image import get_affine_transform, get_affine_transform_upper_left
from utils.nms_utils import nms

from utils.debugger import Debugger

class BaseDetector(object):
  def __init__(self, opt):
    if opt.gpus[0] >= 0:
      opt.device = torch.device('cuda')
    else:
      opt.device = torch.device('cpu')

    self.model = create_model(opt.arch, opt.heads, opt.head_conv)
    self.model = load_model(self.model, opt.load_model, opt.device)
    self.model = self.model.to(opt.device)
    self.model.eval()
    
    self.processor = Processor(opt)
    self.processor = load_model(self.processor, opt.load_processor, opt.device)
    self.processor.cuda()
    self.processor.eval()

    self.mean = np.array(opt.mean, dtype=np.float32).reshape(1, 1, 3)
    self.std = np.array(opt.std, dtype=np.float32).reshape(1, 1, 3)
    self.max_per_image = opt.K
    self.num_classes = opt.num_classes
    self.scales = opt.test_scales
    self.opt = opt
    self.pause = True

  def pre_process(self, image, scale, meta=None):
    height, width = image.shape[0:2]
    # scale 尺度变换比例，默认不进行变换
    new_height = int(height * scale)
    new_width  = int(width * scale)
    if self.opt.fix_res:
      inp_height, inp_width = self.opt.input_h, self.opt.input_w
      c = np.array([new_width / 2., new_height / 2.], dtype=np.float32) # 原图像中心点位置
      s = max(height, width) * 1.0 # 原图最大边长
    else:
      inp_height = (new_height | self.opt.pad) #+ 1
      inp_width = (new_width | self.opt.pad) #+ 1
      c = np.array([new_width // 2, new_height // 2], dtype=np.float32)
      s = np.array([inp_width, inp_height], dtype=np.float32)

    # 仿射变换，图片尺寸变为(inp_height, inp_width)
    if self.opt.upper_left:
      c = np.array([0, 0], dtype=np.float32)
      s = max(height, width) * 1.0
      trans_input = get_affine_transform_upper_left(c, s, 0, [inp_width, inp_height])
    else:
      trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
    resized_image = cv2.resize(image, (new_width, new_height))
    inp_image = cv2.warpAffine(
      resized_image, trans_input, (inp_width, inp_height),
      flags=cv2.INTER_LINEAR)

    # rgb值归一化
    inp_image = ((inp_image / 255. - self.mean) / self.std).astype(np.float32)
    
    # reshape to (1, 3, inp_height, inp_width)
    images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)

    # 推理期间是否需要翻转数据增强，flip_test默认false
    if self.opt.flip_test:
      images = np.concatenate((images, images[:, :, :, ::-1]), axis=0)
    images = torch.from_numpy(images)
    meta = {'c': c, 's': s, 
            'input_height':inp_height,
            'input_width':inp_width,
            'out_height': inp_height // self.opt.down_ratio, 
            'out_width': inp_width // self.opt.down_ratio}
    return images, meta

  def save_img_txt(self,img):
    shape = list(img.shape)
    f1 = open('/home/rujiao.lrj/CenterNet_cell_Coord/src/img.txt','w')
    for i in range(shape[0]):
      for j in range(shape[1]):
        for k in range(shape[2]):
          data = img[i][j][k].item()
          f1.write(str(data)+'\n')
    f1.close()

  def Duplicate_removal(self, results, corners):
    bbox = []
    for j in range(len(results)):
      box = results[j]
      if box[-1] > self.opt.scores_thresh:
        for i in range(8):
          if box[i]<0:
            box[i]=0
          if box[i]>1024:
            box[i]=1024
        def dist(p1,p2):
            return ((p1[0]-p2[0])*(p1[0]-p2[0])+(p1[1]-p2[1])*(p1[1]-p2[1]))**0.5
        p1,p2,p3,p4 = [box[0],box[1]],[box[2],box[3]],[box[4],box[5]],[box[6],box[7]]
        if dist(p1,p2)>3 and dist(p2,p3)>3 and dist(p3,p4)>3 and dist(p4,p1)>3:
            bbox.append(box)
        else:
            continue

    corner = []
    for i in range(len(corners)):
        if corners[i][-1] > self.opt.vis_thresh_corner:
            corner.append(corners[i])
    return np.array(bbox),np.array(corner)
  
  def calculate_iou_with_points(self, box, boxes):
    x1, y1, x2, y2, x3, y3, x4, y4 = box
    x1s, y1s, x2s, y2s, x3s, y3s, x4s, y4s = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4], boxes[:, 5], boxes[:, 6], boxes[:, 7]

    # 计算交集的坐标
    intersection_x1 = np.maximum(x1, x1s)
    intersection_y1 = np.maximum(y1, y1s)
    intersection_x2 = np.minimum(x3, x3s)
    intersection_y2 = np.minimum(y3, y3s)

    # 计算交集面积
    intersection_area = np.maximum(0, intersection_x2 - intersection_x1) * np.maximum(0, intersection_y2 - intersection_y1)

    # 计算每个框的面积
    box_area = (x3 - x1) * (y3 - y1)
    boxes_area = (x3s - x1s) * (y3s - y1s)

    # 计算并返回交并比和面积比
    # iou = intersection_area / (box_area + boxes_area - intersection_area)
    iou = intersection_area / np.minimum(box_area, boxes_area)
    return iou
   
  def filter(self, logi, hmdet, results_batch):
    # 根据cell的置信度筛选
    # logi: tensor(batch, K, 256);  
    # hmdet: tensor(batch, K, 8) heatmap尺寸下的坐标
    # results_batch: ndarray(batch, K, 10) 10=8+score+cls
    # this function select boxes
    batch_size, feat_dim = logi.shape[0], logi.shape[2]
    # 目前单元格cell只有一个类别，不作区分，所以results只选[0]
    # num_valid = sum(results[0][:,8] >= self.opt.vis_thresh)
    num_valid = sum(results_batch[0, :, 8] >= self.opt.vis_thresh) # 目前只考虑单张图片推理，所以选第一个样本来进行阈值过滤
    
    #if num_valid <= 900 : #opt.max_objs
    slct_logi = np.zeros((batch_size, num_valid, feat_dim), dtype=np.float32)
    # heatmap尺寸下的坐标
    slct_dets = np.zeros((batch_size, num_valid, 8), dtype=np.float32)
    # 原图尺寸下的坐标
    slct_dets_ori = np.zeros((batch_size, num_valid, 8), dtype=np.int32)
    # cell class
    slct_det_cls = np.zeros((batch_size, num_valid), dtype=np.int32)
    # cell confidence score
    slct_det_score = np.zeros((batch_size, num_valid), dtype=np.float32)
    
    for i in range(batch_size):
      for j in range(num_valid):
        slct_logi[i,j,:] = logi[i,j,:].cpu()
        slct_dets[i,j,:] = hmdet[i,j,:].cpu()
        slct_dets_ori[i,j,:] = results_batch[i,j,:8]
        slct_det_cls[i,j] = results_batch[i,j,-1]
        slct_det_score[i,j] = results_batch[i,j,-2]

    #else:
      #print('Error: Number of Detected Boxes Exceed the Model Defaults.')
      #quit()
    
    # 前两个是用于后续逻辑坐标预测需要的输入故tensor化，后两个是已经得到的输出结果
    return torch.Tensor(slct_logi).cuda(), torch.Tensor(slct_dets).cuda(), slct_dets_ori, slct_det_cls, slct_det_score
  
  def filter_nms(self, logi, hmdet, results_batch):
    # 根据cell的置信度筛选, 并执行nms, 这是单张推理版本, 所谓的batch维度
    # logi: tensor(batch, K, 256)
    # det: tensor(batch, K, 8) 基于heatmap尺寸的坐标
    # results_batch: ndarray(batch, K, 10) 基于原图的坐标 8+score+cls
    # 单张图版nms, 放在centernet之后（first stage结果）
    if hmdet.shape[1] == 0:
      return
    batch_size, feat_dim = logi.shape[0], logi.shape[2]
    num_valid = sum(results_batch[0, :, 8] >= self.opt.vis_thresh) # 目前只考虑单张图片推理，所以选第一个样本来进行阈值过滤
    
    nms_spatial, nms_clses, nms_scores = [], [], []
    slct_logi, slct_dets = [], []
    
    bboxes_single = results_batch[0, :, :8] # [K, 8]
    scores_single = results_batch[0, :, 8] # [K]
    clses_single = results_batch[0, :, 9] # [K]
    
    logi_embed_single = logi[0] # [K, 256]
    det_hm_single = hmdet[0] # [K, 8]

    # 首先根据置信度筛选
    selected_indices = np.where(scores_single > self.opt.vis_thresh)[0]
    assert(len(selected_indices) == num_valid)
    if num_valid == 0:
      slct_logi = np.zeros((batch_size, num_valid, feat_dim), dtype=np.float32)
      slct_dets = np.zeros((batch_size, num_valid, 8), dtype=np.float32)
      nms_spatial.append(np.zeros((num_valid, 8), dtype=np.int32))
      nms_clses.append(np.zeros((num_valid), dtype=np.int32))
      nms_scores.append(np.zeros((num_valid), dtype=np.float32))
      return torch.Tensor(slct_logi).cuda(), torch.Tensor(slct_dets).cuda(), nms_spatial, nms_clses, nms_scores

    keep_indices = []
    
    sorted_indices = np.argsort(scores_single[selected_indices])[::-1]
    while len(sorted_indices) > 0:
      best_idx = sorted_indices[0]
      keep_indices.append(selected_indices[best_idx])

      best_box = bboxes_single[selected_indices[best_idx]]
      other_boxes = bboxes_single[selected_indices[sorted_indices[1:]]]
      iou = self.calculate_iou_with_points(best_box, other_boxes)

      overlapping_indices = np.where(iou <= self.opt.thresh_min)[0]
      sorted_indices = sorted_indices[overlapping_indices + 1]
    
    keep_indices = np.sort(keep_indices) # 原本索引的顺序就是按照score从大到小的，为了恢复score从大到小顺序，只要增序排一下索引就行
    
    nms_spatial.append(bboxes_single[keep_indices])
    nms_clses.append(clses_single[keep_indices])
    nms_scores.append(scores_single[keep_indices])
    
    slct_logi.append(logi_embed_single[keep_indices])
    slct_dets.append(det_hm_single[keep_indices])
    
    return torch.stack(slct_logi).cuda(), torch.stack(slct_dets).cuda(), nms_spatial, nms_clses, nms_scores

  def process_logi(self, logi):
    logi_floor = logi.floor()
    dev = logi - logi_floor
    logi = torch.where(dev>0.5, logi_floor+1, logi_floor)
    
    return logi

  def _normalized_ps(self, ps, vocab_size):
    ps = torch.round(ps).to(torch.int64)
    ps = torch.where(ps < vocab_size, ps, (vocab_size-1) * torch.ones(ps.shape).to(torch.int64).cuda())
    ps = torch.where(ps >= 0, ps, torch.zeros(ps.shape).to(torch.int64).cuda())
    return ps

  def resize(self,image):
    h,w,_ = image.shape
    scale = 1024/(max(w,h)+1e-4)
    image = cv2.resize(image,(int(w*scale),int(h*scale)))
    image = cv2.copyMakeBorder(image,0,1024 - int(h*scale), 0, 1024 - int(w*scale),cv2.BORDER_CONSTANT, value=[0,0,0])
    return image,scale

  def process(self, images, return_time=False):
    raise NotImplementedError

  def post_process(self, dets, meta, scale=1):
    raise NotImplementedError

  def merge_outputs(self, detections):
    raise NotImplementedError

  def debug(self, debugger, images, dets, output, scale=1):
    raise NotImplementedError

  def show_results(self, debugger, image, results):
   raise NotImplementedError

  def ps_convert_minmax(self, results):
    detection = {}
    # change 1-n to 0-(n-1)
    for j in range(0,self.num_classes):
      detection[j]=[]
    # change 1-n to 0-(n-1)
    for j in range(0,self.num_classes):
      for bbox in results[j]:
        minx = min(bbox[0],bbox[2],bbox[4],bbox[6])
        miny = min(bbox[1],bbox[3],bbox[5],bbox[7])
        maxx = max(bbox[0],bbox[2],bbox[4],bbox[6])
        maxy = max(bbox[1],bbox[3],bbox[5],bbox[7])
        detection[j].append([minx,miny,maxx,maxy,bbox[-1]])
    # change 1-n to 0-(n-1)
    for j in range(0,self.num_classes):
      detection[j] = np.array(detection[j])
    return detection

  def run(self, opt, image_or_path_or_tensor, image_anno=None, meta=None):
   
    debugger = Debugger(dataset=self.opt.dataset, ipynb=(self.opt.debug==3),
                        theme=self.opt.debugger_theme)
 
    pre_processed = False
    if isinstance(image_or_path_or_tensor, np.ndarray):
      image = image_or_path_or_tensor
    elif type(image_or_path_or_tensor) == type (''): 
      # cv2.imread读取返回的就是np.ndarray格式
      image = cv2.imread(image_or_path_or_tensor)
    else:
      image = image_or_path_or_tensor['image'][0].numpy()
      pre_processed_images = image_or_path_or_tensor
      pre_processed = True


    if not opt.wiz_detect:
      batch = make_batch(opt, image_or_path_or_tensor, image_anno)
  
    detections = [] # (scale, class, K, 9(8+1))
    batch_detections = [] # (scale, batch, K, 10)

    hm = []
    corner_st = []
    if self.opt.img_dir != '' and type(image_or_path_or_tensor) == type (''):
      image_name = image_or_path_or_tensor.split('/')[-1]
      
    for scale in self.scales:
      # 图像预处理
      # images是预处理好的图像，image是原始图像
      # meta是图像的一些信息，包括 c: resize后图像中心点位置; s: 原图最长边; out_height: 输出heatmap高度; out_width: 输出heatmap图像宽度
      if not pre_processed:
        images, meta = self.pre_process(image, scale, meta)
    
      else:
        images = pre_processed_images['images'][scale][0]
        meta = pre_processed_images['meta'][scale]
        meta = {k: v.numpy()[0] for k, v in meta.items()}
        
      images = images.to(self.opt.device) # (batch, 3, H, W)

      torch.cuda.synchronize()

      # dets (batch, K, 10=4*2+score+cls) keep (batch, 1, H, W) logi (batch, K, 256) cr (batch, K, 256) corner_st_reg (batch, K, 11)
      if self.opt.wiz_detect:
        outputs, output, dets, corner_st_reg, forward_time, logi, cr, keep = self.process(images, image, return_time=True)
      else:
        outputs, output, dets, corner_st_reg, forward_time, logi, cr, keep = self.process(images, image, return_time=True, batch=batch)

      # 保留相对于heatmap的坐标
      raw_dets = dets

      torch.cuda.synchronize()
      
      if self.opt.debug >= 2:
        self.debug(debugger, images, dets, output, scale)
      
      # 后处理成符合原图尺寸的坐标，这个函数目前只考虑单张图片推理，dets返回的是batch里的[0]第一个样本
      # dets 是 dict，key表示class，value 是 list，shape 为 [k, 9], 每个value的长度 k 加起来是K
      # corner_st_reg: (batch, K, 11)
      # dets_ori: (batch, K, 10) 8+score+cls
      dets, corner_st_reg, dets_ori = self.post_process(dets, meta, corner_st_reg, scale)
      torch.cuda.synchronize()

      detections.append(dets)
      batch_detections.append(dets_ori)
      hm.append(keep)
       
    if self.opt.wiz_2dpe or self.opt.wiz_4ps:
      logi = logi + cr

    # class, scale * K, 9(8+1)
    # merge目的是合并多个scale
    results = self.merge_outputs(detections)
    # batch, scale * K, 10
    results_batch = self.merge_outputs2(batch_detections)
    torch.cuda.synchronize()

    # 对单元格目标根据置信度进行阈值筛选，采用opt.vis_thresh作为阈值
    # Tensor: slct_logi (batch, num_valid, 256) slct_dets (batch, num_valid, 8), slct_dets是heatmap尺寸下的坐标，这两个用于后续逻辑位置预测输入
    # Ndarray: slct_dets_ori (batch, num_valid, 8) slct_dets_ori是原图尺寸下的cell坐标
    # Ndarray: slct_dets_cls (batch, num_valid) slct_dets_cls是cell的类别
    # slct_logi, slct_dets, slct_dets_ori, slct_dets_cls, slct_dets_score = self.filter(logi, raw_dets[:,:,:8], results_batch)
    slct_logi, slct_dets, slct_dets_ori, slct_dets_cls, slct_dets_score = self.filter_nms(logi, raw_dets[:,:,:8], results_batch)
    slct_dets = self._normalized_ps(slct_dets, 256)
    
    if self.opt.wiz_2dpe:
      if self.opt.wiz_stacking:
        _, slct_logi = self.processor(slct_logi, dets = slct_dets)
      else:
        slct_logi = self.processor(slct_logi, dets = slct_dets)
    else:
      if self.opt.wiz_stacking:
        _, slct_logi = self.processor(slct_logi)
      else:
        slct_logi = self.processor(slct_logi)
    
    # slct_logi (batch, num_valid, 4)
    slct_logi = self.process_logi(slct_logi) # 把float格式的逻辑坐标近似到最接近的int
    slct_logi = slct_logi.int()
    slct_logi = slct_logi.detach().cpu().numpy()

    # NMS 由于不同图像检测的目标数量不同, 因此放到第二阶段后做
    # 返回的是list的第一个元素, 元素是ndarray: (num_valid, x)
    # if self.opt.nms:
    #   slct_dets_ori, slct_logi, slct_dets_cls, slct_dets_score = nms(slct_dets_ori, slct_logi, slct_dets_cls, slct_dets_score, self.opt, prior='score')

    torch.cuda.synchronize()

    # if self.opt.debug >= 1 and type(image_or_path_or_tensor) == type (''):
    #   self.show_results(debugger, image, results, corner_st_reg, image_name, slct_logi[0])
    
    # 四点坐标转换成对角点坐标, 这里的Result和result都是没有根据置信度阈值筛的，所以比logic多很多框; corner_st_reg作为角点坐标也没有进行置信度过滤
    Results = self.ps_convert_minmax(results)
    
    # 目前仅支持单张推理，返回的是batch里面的第一个元素
    return {'results': Results,'4ps':results,'corner_st_reg':corner_st_reg, 'hm': hm, 
            'logi': slct_logi[0], 'spatial': slct_dets_ori[0], 
            'cls': slct_dets_cls[0], 'score': slct_dets_score[0]}