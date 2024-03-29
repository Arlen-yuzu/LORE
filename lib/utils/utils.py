from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
import numpy as np
import cv2
from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform, get_affine_transform_upper_left
from utils.image import gaussian_radius, draw_umich_gaussian, draw_umich_gaussian_wh, draw_msra_gaussian
from utils.image import draw_dense_reg
from utils.adjacency import adjacency
import math

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):

        self.val = val
        self.sum += val * n
        if self.val != 0:
            self.count += n
        if self.count > 0:
          self.avg = self.sum / self.count

def _judge(box):
    countx = len(list(set([box[0],box[2],box[4],box[6]]))) 
    county = len(list(set([box[1],box[3],box[5],box[7]]))) 
    if countx<2 or county<2:
        return False
    
    return True

def make_batch(opt, path, anns):
    max_objs = 300
    max_cors = 1200
    num_objs = min(len(anns), max_objs)
    if opt.kvobj:
        num_classes = 3
        _valid_ids = [0,1,2] # 原来是[1,2]
    else:
        num_classes = 2
        _valid_ids = [0,1]
    #print(anns[1])
    cat_ids = {v: i for i, v in enumerate(_valid_ids)}
    img_path = path
    
    mean = np.array([0.40789654, 0.44719302, 0.47026115], dtype=np.float32).reshape(1, 1, 3)
    std  = np.array([0.28863828, 0.27408164, 0.27809835], dtype=np.float32).reshape(1, 1, 3)

    img = cv2.imread(img_path)
    img_size = img.shape

    height, width = img.shape[0], img.shape[1]
    
    if opt.upper_left:
      c = np.array([0, 0], dtype=np.float32)
    else:
      c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
    
    if opt.keep_res:
      input_h = (height | opt.pad)# + 1
      input_w = (width | opt.pad)# + 1
      s = np.array([input_w, input_h], dtype=np.float32)
    else:
      s = max(img.shape[0], img.shape[1]) * 1.0
      input_h, input_w = opt.input_h, opt.input_w
    
    flipped = False

    rot = 0
    if opt.rotate==1:
      print('----rotate----')
      rot = np.random.randint(-15,15)

    output_h = input_h // opt.down_ratio
    output_w = input_w // opt.down_ratio
    
    if opt.upper_left:
      trans_input = get_affine_transform_upper_left(c, s, rot, [input_w, input_h])
      trans_output = get_affine_transform_upper_left(c, s, rot, [output_w, output_h])
      trans_output_mk = get_affine_transform_upper_left(c, s, rot, [output_w, output_h])
    else:
      trans_input = get_affine_transform(c, s, rot, [input_w, input_h])
      trans_output = get_affine_transform(c, s, rot, [output_w, output_h])
      trans_output_mk = get_affine_transform(c, s, rot, [output_w, output_h])

    hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
    wh = np.zeros((max_objs, 8), dtype=np.float32)
    reg = np.zeros((max_objs*5, 2), dtype=np.float32)
    st = np.zeros((max_cors, 8), dtype=np.float32)
    hm_ctxy = np.zeros((max_objs, 2), dtype=np.float32)
    hm_xy = np.zeros((max_objs, 10), dtype=np.float32)
    hm_ind = np.zeros((max_objs), dtype=np.int64)
    hm_mask = np.zeros((max_objs), dtype=np.uint8)
    mk_ind = np.zeros((max_cors), dtype=np.int64)
    mk_mask = np.zeros((max_cors), dtype=np.uint8)
    reg_ind = np.zeros((max_objs*5), dtype=np.int64)
    reg_mask = np.zeros((max_objs*5), dtype=np.uint8)
    ctr_cro_ind = np.zeros((max_objs*4), dtype=np.int64)
    log_ax = np.zeros((max_objs, 4), dtype=np.int64)
    box_ind = np.zeros((max_objs, 8), dtype=np.int64)
    cc_match = np.zeros((max_objs, 4), dtype=np.int64)
    adjacent = np.zeros((max_objs, max_objs), dtype=np.int64)
    adj_mask = np.zeros((max_objs, max_objs), dtype=np.int64)
    center = np.zeros((max_objs, 2), dtype=np.float32)

    draw_gaussian = draw_msra_gaussian if opt.mse_loss else \
                    draw_umich_gaussian
    gt_det = []
    corList = []
    point = []
    inp = cv2.warpAffine(img, trans_input, (input_w, input_h),flags=cv2.INTER_LINEAR)
    
    for k in range(num_objs):
      ann = anns[k]
      #bbox = self._coco_box_to_bbox(ann['bbox'])
      seg_mask = ann['segmentation'][0] #[[351.0, 73.0, 172.0, 70.0, 174.0, 127.0, 351.0, 129.0, 351.0, 73.0]]
      x1,y1 = seg_mask[0],seg_mask[1]
      x2,y2 = seg_mask[2],seg_mask[3]
      x3,y3 = seg_mask[4],seg_mask[5]
      x4,y4 = seg_mask[6],seg_mask[7]

      CorNer = np.array([x1,y1,x2,y2,x3,y3,x4,y4], dtype=np.float32)
      boxes = [[CorNer[0],CorNer[1]],[CorNer[2],CorNer[3]],\
               [CorNer[4],CorNer[5]],[CorNer[6],CorNer[7]]]
      cls_id = int(cat_ids[ann['category_id']])
      if flipped:
        #bbox[[0, 2]] = width - bbox[[2, 0]] - 1
        CorNer[[0,2,4,6]] = width - CorNer[[2,0,6,4]] - 1

      CorNer[0:2] = affine_transform(CorNer[0:2], trans_output_mk)
      CorNer[2:4] = affine_transform(CorNer[2:4], trans_output_mk)
      CorNer[4:6] = affine_transform(CorNer[4:6], trans_output_mk)
      CorNer[6:8] = affine_transform(CorNer[6:8], trans_output_mk)
      CorNer[[0,2,4,6]] = np.clip(CorNer[[0,2,4,6]], 0, output_w - 1)
      CorNer[[1,3,5,7]] = np.clip(CorNer[[1,3,5,7]], 0, output_h - 1)
      if not _judge(CorNer):
          continue
 
      maxx = max([CorNer[2*I] for I in range(0,4)])
      minx = min([CorNer[2*I] for I in range(0,4)])
      maxy = max([CorNer[2*I+1] for I in range(0,4)])
      miny = min([CorNer[2*I+1] for I in range(0,4)])
      h, w = maxy-miny,maxx-minx#bbox[3] - bbox[1], bbox[2] - bbox[0]
      if h > 0 and w > 0:
        
        radius = gaussian_radius((math.ceil(h), math.ceil(w)))
        radius = max(0, int(radius))
        radius = opt.hm_gauss if opt.mse_loss else radius
        
        ct = np.array([(maxx+minx)/2.0,(maxy+miny)/2.0], dtype=np.float32)
        ct_int = ct.astype(np.int32)
        draw_gaussian(hm[cls_id], ct_int, radius)

        for i in range(4):
          Cor = np.array([CorNer[2*i],CorNer[2*i+1]], dtype=np.float32)
          Cor_int = Cor.astype(np.int32)
          Cor_key = str(Cor_int[0])+"_"+str(Cor_int[1])
          if Cor_key not in corList:
            
            corNum = len(corList)
            
            corList.append(Cor_key)
            reg[max_objs+corNum] = np.array([abs(Cor[0]-Cor_int[0]),abs(Cor[1]-Cor_int[1])])
            mk_ind[corNum] = Cor_int[1]*output_w + Cor_int[0]
            cc_match[k][i] = mk_ind[corNum]
            box_ind[k][i*2:(i+1)*2] = np.array((corNum, corNum))
            reg_ind[max_objs+corNum] = Cor_int[1]*output_w + Cor_int[0]
            mk_mask[corNum] = 1
            reg_mask[max_objs+corNum] = 1
            draw_gaussian(hm[num_classes-1], Cor_int, 2)
            st[corNum][i*2:(i+1)*2] = np.array([Cor[0]-ct[0],Cor[1]-ct[1]])
            ctr_cro_ind[4*k+i] = corNum*4 + i
            
          else:
            index_of_key = corList.index(Cor_key)
            cc_match[k][i] = mk_ind[index_of_key]
            box_ind[k][i*2:(i+1)*2] = np.array((index_of_key, index_of_key))
            st[index_of_key][i*2:(i+1)*2] = np.array([Cor[0]-ct[0],Cor[1]-ct[1]])
            ctr_cro_ind[4*k+i] = index_of_key*4 + i

        wh[k] = ct[0] - 1. * CorNer[0], ct[1] - 1. * CorNer[1], \
                ct[0] - 1. * CorNer[2], ct[1] - 1. * CorNer[3], \
                ct[0] - 1. * CorNer[4], ct[1] - 1. * CorNer[5], \
                ct[0] - 1. * CorNer[6], ct[1] - 1. * CorNer[7]
        
        hm_ind[k] = ct_int[1] * output_w + ct_int[0]
        hm_mask[k] = 1
        reg_ind[k] = ct_int[1] * output_w + ct_int[0]
        reg_mask[k] = 1
        reg[k] = ct - ct_int
        hm_ctxy[k] = ct[0],ct[1]
        hm_xy[k] = ct[0], ct[1], CorNer[0], CorNer[1], CorNer[2], CorNer[3], CorNer[4], CorNer[5], CorNer[6], CorNer[7]
        center[k] = (x1 + x2 + x3 + x4)/4, (y1 + y2 + y3 + y4)/4
        log_ax[k] = ann['logic_axis'][0][0], ann['logic_axis'][0][1], ann['logic_axis'][0][2], ann['logic_axis'][0][3]
        gt_det.append([ct[0] - 1. * CorNer[0], ct[1] - 1. * CorNer[1],
                       ct[0] - 1. * CorNer[2], ct[1] - 1. * CorNer[3],
                       ct[0] - 1. * CorNer[4], ct[1] - 1. * CorNer[5], 
                       ct[0] - 1. * CorNer[6], ct[1] - 1. * CorNer[7], 1, cls_id])
        for j in range(k):
          box1 = log_ax[k]
          box2 = log_ax[j]
          if adjacency(box1, box2):
            adjacent[j][k] = adjacent[k][j] = 1
          
          if j == k:
            adjacent[j][k] = 1
    
    hm_mask_v = hm_mask.reshape(1, hm_mask.shape[0])
    adj_mask = np.dot(np.transpose(hm_mask_v), hm_mask_v)
    inp = (inp.astype(np.float32) / 255.)
    
    inp = (inp - mean) / std
    inp = inp.transpose(2, 0, 1)
    hm_xy = torch.FloatTensor(hm_xy)
    ret = {'input': inp, 'hm': hm, 'hm_ind':hm_ind, 'hm_mask':hm_mask, 'mk_ind':mk_ind, 'mk_mask':mk_mask, 'reg':reg,'reg_ind':reg_ind,'reg_mask': reg_mask, 'wh': wh,'st':st, 'ctr_cro_ind':ctr_cro_ind,'hm_ctxy':hm_ctxy, 'logic': log_ax, 'adj': adjacent, 'cc_match': cc_match, 'box_ind': box_ind, 'adj_mask': adj_mask, 'hm_xy': hm_xy, 'center': center}
    if opt.dense_wh:
      hm_a = hm.max(axis=0, keepdims=True)
      dense_wh_mask = np.concatenate([hm_a, hm_a], axis=0)
      ret.update({'dense_wh': dense_wh, 'dense_wh_mask': dense_wh_mask})
      del ret['wh']
    elif opt.cat_spec_wh:
      ret.update({'cat_spec_wh': cat_spec_wh, 'cat_spec_mask': cat_spec_mask})
      del ret['wh']
    if opt.reg_offset:
      ret.update({'reg': reg})
    
    gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else \
            np.zeros((1, 10), dtype=np.float32)
    meta = {'c': c, 's': s, 'rot':rot, 'gt_det': gt_det}
    ret['meta'] = meta
    return ret

color_list = np.array(
        [
            0.850, 0.325, 0.098,
            0.929, 0.694, 0.125,
            1.000, 1.000, 1.000,
        ]).astype(np.float32)
color_list = color_list.reshape((-1, 3)) * 255

def add_4ps_coco_bbox(img, bbox, cat, score, logi):
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
      txt = '{:.0f},{:.0f},{:.0f},{:.0f}: {:.2f}'.format(logi[0], logi[1], logi[2], logi[3], score)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cat_size = cv2.getTextSize(txt, font, 0.3, 2)[0]
    cv2.line(img,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,0,255),1)
    cv2.line(img,(bbox[2],bbox[3]),(bbox[4],bbox[5]),(0,0,255),1)
    cv2.line(img,(bbox[4],bbox[5]),(bbox[6],bbox[7]),(0,0,255),1)
    cv2.line(img,(bbox[6],bbox[7]),(bbox[0],bbox[1]),(0,0,255),1) # 1 - 5
  
    if not logi is None:
      cv2.rectangle(img,
                    (bbox[0], bbox[1] - cat_size[1] - 2),
                    (bbox[0] + cat_size[0], bbox[1] - 2), c, -1)
      cv2.putText(img, txt, (bbox[0], bbox[1] - 2), 
                  font, 0.30, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA) #1 - 5 # 0.20 _ 0.60

def show_results(image, spatial, image_name, logi, clses, scores, opt):
    if not os.path.exists(opt.output_dir + opt.demo_name):
      os.makedirs(opt.output_dir + opt.demo_name +'/center/')
      # os.makedirs(opt.output_dir + opt.demo_name +'/corner/')
      os.makedirs(opt.output_dir + opt.demo_name +'/logi/')
      os.makedirs(opt.output_dir + opt.demo_name +'/cls/')
      os.makedirs(opt.output_dir + opt.demo_name +'/score/')
      os.makedirs(opt.output_dir + opt.demo_name +'/html/')
    fc = open(opt.output_dir + opt.demo_name +'/center/'+image_name+'.txt','w') #bounding boxes saved
    fl = open(opt.output_dir + opt.demo_name +'/logi/'+image_name+'.txt','w') #logic axis saved
    fcls = open(opt.output_dir + opt.demo_name +'/cls/'+image_name+'.txt','w') #class saved
    fs = open(opt.output_dir + opt.demo_name +'/score/'+image_name+'.txt','w') #score saved
    for i in range(len(spatial)):
      bbox = spatial[i]
      add_4ps_coco_bbox(image, bbox, clses[i], scores[i], logi[i])
      for j in range(0,3):
        fc.write(str(bbox[2*j])+','+str(bbox[2*j+1])+';')
        if not logi is None:
          fl.write(str(int(logi[i, j]))+',')
      fc.write(str(bbox[6])+','+str(bbox[7])+'\n')
      if not logi is None:
        fl.write(str(int(logi[i, 3]))+'\n')
      fcls.write(str(int(clses[i]))+'\n')
      fs.write(str(scores[i])+'\n')
    if not os.path.exists(os.path.join(opt.demo_dir, 'tsr')):
      os.makedirs(os.path.join(opt.demo_dir, 'tsr'))
    # 保存bbox框+逻辑坐标到原图
    cv2.imwrite(os.path.join(os.path.join(opt.demo_dir, 'tsr'), image_name), image)

def compute_IOU(bbox1, bbox2):
    rec1 = (bbox1[0], bbox1[1], bbox1[4], bbox1[5])
    rec2 = (bbox2[0], bbox2[1], bbox2[4], bbox2[5])
    left_column_max = max(rec1[0],rec2[0])
    right_column_min = min(rec1[2],rec2[2])
    up_row_max = max(rec1[1],rec2[1])
    down_row_min = min(rec1[3],rec2[3])
    
    if left_column_max>=right_column_min or down_row_min<=up_row_max:
        return 0.
    else:
        S1 = (rec1[2]-rec1[0])*(rec1[3]-rec1[1])
        S2 = (rec2[2]-rec2[0])*(rec2[3]-rec2[1])
        S_cross = (down_row_min-up_row_max)*(right_column_min-left_column_max)
        return S_cross / S1

def textmatch(ocr_bbox, ocr_text, cell_bbox, cell_text):
    # 单张图的cell-text匹配
    # ocr_box: list of numpy array, ndarray of shape [4, 2]
    # ocr_text: list of tuple, tuple of (text, score)
    # cell_bbox: [num_valid, 8]
    # cell_text: [num_valid]
    for i in range(len(ocr_bbox)):
        ocr_box = ocr_bbox[i].reshape(-1)
        ocr_txt = ocr_text[i][0]
        max_iou = 0
        max_idx = -1
        for j in range(len(cell_bbox)):
            cell_box = cell_bbox[j]
            iou = compute_IOU(ocr_box, cell_box)
            if iou > max_iou:
                max_iou = iou
                max_idx = j
        if max_idx != -1:
            cell_text[max_idx] += ocr_txt    
    return cell_text
