import numpy as np
import cv2
import os


color_list = np.array(
        [
            0.850, 0.325, 0.098,
            0.929, 0.694, 0.125,
            1.000, 1.000, 1.000,
        ]).astype(np.float32)
color_list = color_list.reshape((-1, 3)) * 255

def add_bbox(img, bbox, logi):
    bbox = np.array(bbox, dtype=np.int32)

    colors = [(color_list[_]).astype(np.uint8) \
            for _ in range(len(color_list))]
    colors = np.array(colors, dtype=np.uint8).reshape(len(colors), 1, 1, 3)
    colors = colors.reshape(-1)[::-1].reshape(len(colors), 1, 1, 3)
    colors = np.clip(colors, 0., 0.6 * 255).astype(np.uint8)
    c = colors[0][0][0].tolist()
    c = (255 - np.array(c)).tolist()
    
    if not logi is None:
      txt = '{:.0f},{:.0f},{:.0f},{:.0f}'.format(logi[0], logi[1], logi[2], logi[3])
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

def add_bbox2(img, bbox, color=(0,0,255)):
    bbox = np.array(bbox, dtype=np.int32)
    
    cv2.line(img,(bbox[0],bbox[1]),(bbox[2],bbox[3]),color,1)
    cv2.line(img,(bbox[2],bbox[3]),(bbox[4],bbox[5]),color,1)
    cv2.line(img,(bbox[4],bbox[5]),(bbox[6],bbox[7]),color,1)
    cv2.line(img,(bbox[6],bbox[7]),(bbox[0],bbox[1]),color,1) # 1 - 5

def show_results(image_name, image, bboxes, logi, opt):
    for i in range(len(bboxes)):
      bbox = bboxes[i]
      add_bbox(image, bbox, logi[i])
    if not os.path.exists(os.path.join(opt.vis_dir, 'tsr')):
      os.makedirs(os.path.join(opt.vis_dir, 'tsr'))
    # 保存bbox框+逻辑坐标到原图
    cv2.imwrite(os.path.join(os.path.join(opt.vis_dir, 'tsr'), image_name), image)

def show_results_with_gt(image_name, image, bboxes_pred, bbox_gt, opt):
    for i in range(len(bboxes_pred)):
      bbox = bboxes_pred[i]
      add_bbox2(image, bbox, (0,0,255))
    for i in range(len(bbox_gt)):
      bbox = bbox_gt[i]
      add_bbox2(image, bbox, (0,255,0))
    if not os.path.exists(os.path.join(opt.vis_dir, 'tsr')):
      os.makedirs(os.path.join(opt.vis_dir, 'tsr'))
    # 保存到原图
    cv2.imwrite(os.path.join(os.path.join(opt.vis_dir, 'tsr'), image_name), image)
