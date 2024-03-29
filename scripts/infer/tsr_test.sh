export CUDA_VISIBLE_DEVICES="0" 
python tsr_test.py ctdet_mid \
        --dataset table_mid \
        --demo /shared/aia/alg/xyl/tsrdataset/model/lore/comfintab/image \
        --demo_name lore_comfintab \
        --debug 1 \
        --dataset_name comfintab \
        --arch resfpnhalf_18  \
        --K 3000 \
        --MK 5000 \
        --upper_left \
        --tsfm_layers 4\
        --stacking_layers 4 \
        --gpus 0\
        --wiz_2dpe \
        --wiz_detect \
        --wiz_stacking \
        --convert_onnx 0 \
        --vis_thresh_corner 0.3 \
        --vis_thresh 0.2 \
        --scores_thresh 0.2 \
        --nms \
        --demo_dir ./visualization_metric12/ \
        --anno_path /shared/aia/alg/xyl/tsrdataset/model/lore/comfintab/json/test.json \
        --load_model ./dir_of_ckpt/ckpt_wireless/model_best.pth \
        --load_processor ./dir_of_ckpt/ckpt_wireless/processor_best.pth