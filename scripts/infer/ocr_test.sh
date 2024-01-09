export CUDA_VISIBLE_DEVICES="4,5,6,7" 
python ocr_test.py ctdet_mid \
        --dataset table_mid \
        --demo ../input_images/crop_realtable \
        --demo_name tsr_infer \
        --debug 1 \
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
        --kvobj \
        --demo_dir ../visualization_ocr_test3/ \
        --load_model ../exp/ctdet_mid/train_syntable_1019/model_best.pth \
        --load_processor ../exp/ctdet_mid/train_syntable_1019/processor_best.pth