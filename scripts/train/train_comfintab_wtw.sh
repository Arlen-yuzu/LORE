export CUDA_VISIBLE_DEVICES="1,3" 

port=`shuf -i 29500-29600 -n1`
res=`lsof -i:${port}`
while [[ -n ${res} ]]; do
    port=$((port + 1))
    res=`lsof -i:${port}`
done

PORT=${PORT:-${port}}

echo $PORT

torchrun --nproc_per_node=2 --master_port=$PORT main_ddp.py ctdet_mid \
	--dataset table \
	--exp_id train_comfintab_wtw \
	--dataset_name comfintab \
	--image_dir /shared/aia/alg/xyl/tsrdataset/model/lore/comfintab/image \
	--anno_path /shared/aia/alg/xyl/tsrdataset/model/lore/comfintab/json \
	--wiz_2dpe \
	--wiz_4ps \
	--wiz_stacking \
	--wiz_pairloss \
	--tsfm_layers 3 \
	--stacking_layers 3 \
	--batch_size 8 \
	--master_batch 8 \
	--arch dla_34 \
    --class_num 5 \
	--lr 1e-4 \
	--K 1500 \
	--MK 2500 \
	--num_epochs 200 \
	--lr_step '70, 90' \
	--gpus 0,1 \
	--num_workers 16 \
	--val_intervals 10