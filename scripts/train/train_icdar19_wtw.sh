export CUDA_VISIBLE_DEVICES="1,2,3,4"

port=`shuf -i 29500-29600 -n1`
res=`lsof -i:${port}`
while [[ -n ${res} ]]; do
    port=$((port + 1))
    res=`lsof -i:${port}`
done

PORT=${PORT:-${port}}

echo $PORT

torchrun --nproc_per_node=4 --master_port=$PORT main_ddp.py ctdet \
	--dataset table \
	--exp_id train_icdar19_wtw_only_processor_1226 \
	--dataset_name icdar19 \
	--image_dir /shared/aia/alg/xyl/tsrdataset/model/lore/icdar19/image \
	--anno_path /shared/aia/alg/xyl/tsrdataset/model/lore/icdar19/json \
	--wiz_2dpe \
	--wiz_4ps \
	--wiz_stacking \
	--wiz_pairloss \
	--tsfm_layers 4 \
	--stacking_layers 4 \
	--batch_size 26 \
	--master_batch 26 \
	--arch dla_34 \
	--class_num 1 \
	--lr 5e-4 \
	--K 3000 \
	--MK 5000 \
	--num_epochs 400 \
	--lr_step '70, 150, 250' \
	--gpus 0,1 \
	--num_workers 16 \
	--val_intervals 5 \
	--load_model /data/xuyilun/project/LORE/exp/ctdet/train_icdar19_wtw_only_detector_1221/model_best.pth \
	--only_processor