python demo.py ctdet \
        --dataset table \
        --demo ../data/WTW/images/ \
        --demo_name wtw_metric \
        --debug 1 \
        --dataset_name WTW \
        --arch dla_34  \
        --K 3000 \
        --MK 5000 \
        --tsfm_layers 4\
        --stacking_layers 4 \
        --gpus 0\
        --wiz_4ps \
        --wiz_detect \
        --wiz_stacking \
        --convert_onnx 0 \
        --vis_thresh_corner 0.3 \
        --vis_thresh 0.2 \
        --scores_thresh 0.2 \
        --nms \
        --demo_dir ../visualization_wtw_metric/ \
        --anno_path ../data/WTW/json/test.json \
        --load_model ../dir_of_ckpt/ckpt_wtw/model_best.pth \
        --load_processor ../dir_of_ckpt/ckpt_wtw/processor_best.pth