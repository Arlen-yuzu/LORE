export CUDA_VISIBLE_DEVICES="1" 
python tsr_test.py ctdet_mid \
        --dataset table_mid \
        --demo_name lore_pubtabnet \
        --dataset_name pubtabnet \
        --img_dir /shared/aia/alg/xyl/tsrdataset/unify/pubtabnet/image \
        --anno_path /shared/aia/alg/xyl/tsrdataset/model/lore/pubtabnet/json/val.json \
        --vis_dir ./visualization_pubtabnet/ \
        --load_model /data/xuyilun/project/LORE/exp/ctdet_mid/train_sym/model_best.pth \
        --load_processor /data/xuyilun/project/LORE/exp/ctdet_mid/train_sym/processor_best.pth \
        --debug 1 \
        --class_num 4 \
        --arch dla_34 \
        --K 3000 \
        --MK 5000 \
        --tsfm_layers 4 \
        --stacking_layers 4 \
        --gpus 0 \
        --wiz_4ps \
        --wiz_2dpe \
        --wiz_detect \
        --wiz_stacking \
        --convert_onnx 0 \
        --vis_thresh_corner 0.3 \
        --vis_thresh 0.2 \
        --scores_thresh 0.2 \
        --nms
