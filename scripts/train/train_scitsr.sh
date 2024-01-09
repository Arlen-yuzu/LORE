export CUDA_VISIBLE_DEVICES="4,5"

port=`shuf -i 29500-29600 -n1`
res=`lsof -i:${port}`
while [[ -n ${res} ]]; do
    port=$((port + 1))
    res=`lsof -i:${port}`
done

PORT=${PORT:-${port}}

echo $PORT

torchrun --nproc_per_node=2 --master_port=$PORT main_ddp.py ctdet_mid \
	--dataset table_mid \
	--exp_id train_scitsr \
	--dataset_name scitsr \
	--image_dir /shared/aia/alg/xyl/tsrdataset/unify/scitsr/image \
	--anno_path /shared/aia/alg/xyl/tsrdataset/model/lore/scitsr/json \
	--wiz_2dpe \
	--wiz_stacking \
	--tsfm_layers 3 \
	--stacking_layers 3 \
	--batch_size 12 \
	--master_batch 12 \
	--arch dla_34 \
	--class_num 1 \
	--lr 1e-4 \
	--K 500 \
	--MK 1000 \
	--num_epochs 100 \
	--lr_step '70, 90' \
	--gpus 0,1 \
	--num_workers 16 \
	--val_intervals 10 \
	--load_model /data/xuyilun/project/LORE/dir_of_ckpt/ckpt_ptn/model_best.pth \
	--load_processor /data/xuyilun/project/LORE/dir_of_ckpt/ckpt_ptn/processor_best.pth