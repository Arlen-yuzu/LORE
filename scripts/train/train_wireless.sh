python main.py ctdet_mid \
	--dataset table_mid \
	--exp_id train_wireless \
	--dataset_name gentable \
	--image_dir ../data/gentable/images \
	--wiz_2dpe \
	--wiz_stacking \
	--tsfm_layers 4 \
	--stacking_layers 4 \
	--batch_size 1 \
	--master_batch 12 \
	--arch dla_34 \
	--lr 1e-4 \
	--K 500 \
	--MK 1000 \
	--num_epochs 5 \
	--lr_step '100, 160' \
	--gpus 0\
	--num_workers 16 \
	--val_intervals 10 