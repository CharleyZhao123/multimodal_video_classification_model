CUDA_VISIBLE_DEVICES=1 python train.py \
               --root_path ~/brain4cars/bky40/code \
               --video_path brain4cars_data \
               --annotation_path datasets/annotation \
			   --result_path results \
			   --dataset Brain4cars_Unit \
			   --batch_size 16 \
			   --n_threads 4 \
			   --checkpoint 5  \
			   --n_epochs 100 \
			   --begin_epoch 1 \
			   --sample_duration 10 \
			   --end_second 0 \
			   --interval 16 \
			   --n_scales 1 \
			   --learning_rate 0.001 \
			   --norm_value 255 \
			   --n_fold 0 \
			   --train_from_scratch 0 \

