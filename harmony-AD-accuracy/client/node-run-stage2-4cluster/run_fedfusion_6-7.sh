CUDA_VISIBLE_DEVICES=2 python3 ./main_fusion_3modal.py --usr_id 6 --local_modality all --batch_size 16 --num_workers 16 &
CUDA_VISIBLE_DEVICES=3 python3 ./main_fusion_3modal.py --usr_id 7 --local_modality all --batch_size 16 --num_workers 16 &

