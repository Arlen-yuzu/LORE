import numpy as np


def nms(spatial, logi, clses, scores, opt, prior='score'):
  if prior == 'score':
    return nms_score(spatial, logi, clses, scores, opt)
  elif prior == 'area':
    return nms_area(spatial, logi, clses, scores, opt)

def nms_area(spatial, logi, clses, scores, opt):
  # batch版nms, 放在processer输出之后
  # spatial: ndarray(batch, num_valid, 8) slct_dets_ori是原图尺寸下的cell坐标
  # logi:    ndarray(batch, num_valid, 4) slct_logi是逻辑坐标
  # clses:   ndarray(batch, num_valid) slct_dets_cls是cell的类别
  # scores:  ndarray(batch, num_valid) slct_dets_score是cell的置信度
  if spatial.shape[1] == 0:
    return
  # each example in batch has different number of boxes, so use list of ndarray to store
  nms_spatial, nms_logi, nms_clses, nms_scores = [], [], [], []
  batch_size = logi.shape[0]
  for i in range(batch_size):
    bboxes_single = spatial[i] # [K, 8]
    scores_single = scores[i] # [K]
    areaes_single = np.array((bboxes_single[:, 4] - bboxes_single[:, 0]) * (bboxes_single[:, 5] - bboxes_single[:, 1]), dtype=np.float32) # [K]

    # 首先根据置信度筛选
    selected_indices = np.where(scores_single > opt.vis_thresh)[0]
    # bboxes_single = bboxes_single[selected_indices]
    # scores_single = scores_single[selected_indices]
    areaes_single = areaes_single[selected_indices]

    keep_indices = []
    count_nms = 0
    count_wrong_nms = 0
    
    sorted_indices = np.argsort(areaes_single)[::-1]# 根据面积倒序, 目前就是把面积作为优先级进行nms，保留面积大的
    # sorted_indices的索引域是[0, len(selected_indices)-1], 和selected_indices的索引域不一致
    while len(sorted_indices) > 0:
      best_idx = sorted_indices[0]
      keep_indices.append(selected_indices[best_idx]) # keep_indices索引域与selected_indices一致

      # best_box = bboxes_single[best_idx]
      # other_boxes = bboxes_single[sorted_indices[1:]]
      best_box = bboxes_single[selected_indices[best_idx]]
      other_boxes = bboxes_single[selected_indices[sorted_indices[1:]]]
      iou = calculate_iou_with_points(best_box, other_boxes)

      overlapping_indices = np.where(iou <= opt.thresh_min)[0]
      
      # delete_det_indices = np.where(iou <= opt.thresh_min)[0]
      # if len(delete_det_indices):
      #   count_nms += 1
      #   keep_score = scores_single[selected_indices[best_idx]]
      #   out_score = scores_single[selected_indices[sorted_indices[delete_det_indices+1]]].min()
      #   if keep_score < out_score:
      #     count_wrong_nms += 1
      
      sorted_indices = sorted_indices[overlapping_indices + 1]
    
    # print(f'{count_wrong_nms}/{count_nms}')
    keep_indices = np.sort(keep_indices) # 原本索引的顺序就是按照score从大到小的，为了恢复score从大到小顺序，只要增序排一下索引就行
    
    nms_spatial.append(bboxes_single[keep_indices])
    nms_logi.append(logi[i][keep_indices])
    nms_clses.append(clses[i][keep_indices])
    nms_scores.append(scores_single[keep_indices])

  return nms_spatial, nms_logi, nms_clses, nms_scores

def nms_score(spatial, logi, clses, scores, opt):
  # batch版nms, 放在processer输出之后
  # spatial: ndarray(batch, num_valid, 8) slct_dets_ori是原图尺寸下的cell坐标
  # logi:    ndarray(batch, num_valid, 4) slct_logi是逻辑坐标
  # clses:   ndarray(batch, num_valid) slct_dets_cls是cell的类别
  # scores:  ndarray(batch, num_valid) slct_dets_score是cell的置信度
  if spatial.shape[1] == 0:
    return
  # each example in batch has different number of boxes, so use list of ndarray to store
  nms_spatial, nms_logi, nms_clses, nms_scores = [], [], [], []
  batch_size = logi.shape[0]
  for i in range(batch_size):
    bboxes_single = spatial[i] # [K, 8]
    scores_single = scores[i] # [K]

    # 首先根据置信度筛选
    selected_indices = np.where(scores_single > opt.vis_thresh)[0]
    # bboxes_single = bboxes_single[selected_indices]
    # scores_single = scores_single[selected_indices]

    keep_indices = []
    
    sorted_indices = np.argsort(scores_single[selected_indices])[::-1]
    # sorted_indices的索引域是[0, len(selected_indices)-1], 和selected_indices的索引域不一致
    while len(sorted_indices) > 0:
      best_idx = sorted_indices[0]
      keep_indices.append(selected_indices[best_idx]) # keep_indices索引域与selected_indices一致

      # best_box = bboxes_single[best_idx]
      # other_boxes = bboxes_single[sorted_indices[1:]]
      best_box = bboxes_single[selected_indices[best_idx]]
      other_boxes = bboxes_single[selected_indices[sorted_indices[1:]]]
      iou = calculate_iou_with_points(best_box, other_boxes)

      overlapping_indices = np.where(iou <= opt.thresh_min)[0]      
      sorted_indices = sorted_indices[overlapping_indices + 1]
    
    keep_indices = np.sort(keep_indices) # 原本索引的顺序就是按照score从大到小的，为了恢复score从大到小顺序，只要增序排一下索引就行
    
    nms_spatial.append(bboxes_single[keep_indices])
    nms_logi.append(logi[i][keep_indices])
    nms_clses.append(clses[i][keep_indices])
    nms_scores.append(scores_single[keep_indices])

  return nms_spatial, nms_logi, nms_clses, nms_scores

def calculate_iou_with_points(box, boxes):
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
