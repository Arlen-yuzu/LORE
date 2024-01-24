export CUDA_VISIBLE_DEVICES="0,1,2,3"

port=`shuf -i 29500-29600 -n1`
res=`lsof -i:${port}`
while [[ -n ${res} ]]; do
    port=$((port + 1))
    res=`lsof -i:${port}`
done

PORT=${PORT:-${port}}

echo $PORT

torchrun --nproc_per_node=4 --master_port=$PORT main_ddp.py ctdet_mid \
	--dataset table_mid \
	--exp_id train_sym_align \
	--dataset_name sym_align \
    --root_dir /shared/aia/alg/xyl/tsrckpt/lore \
	--image_dir /shared/aia/alg/xyl/tsrdataset/unify/sym/image \
	--anno_path /shared/aia/alg/xyl/tsrdataset/model/lore/sym_align/json \
	--wiz_2dpe \
	--wiz_4ps \
	--wiz_stacking \
	--wiz_pairloss \
	--tsfm_layers 4 \
	--stacking_layers 4 \
	--batch_size 8 \
	--master_batch 8 \
	--arch dla_34 \
	--class_num 4 \
	--lr 1e-4 \
	--K 500 \
	--MK 1000 \
	--num_epochs 100 \
	--lr_step '70, 90' \
	--gpus 0,1 \
	--num_workers 16 \
	--val_intervals 10