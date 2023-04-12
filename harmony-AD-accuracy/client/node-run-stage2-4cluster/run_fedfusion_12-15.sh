CUDA_VISIBLE_DEVICES=0 python3 ./main_fusion_3modal.py --usr_id 12 --local_modality all --batch_size 16 --num_workers 16 &
CUDA_VISIBLE_DEVICES=1 python3 ./main_fusion_3modal.py --usr_id 13 --local_modality all --batch_size 16 --num_workers 16 &
CUDA_VISIBLE_DEVICES=2 python3 ./main_fusion_3modal.py --usr_id 14 --local_modality all --batch_size 16 --num_workers 16 &
CUDA_VISIBLE_DEVICES=3 python3 ./main_fusion_3modal.py --usr_id 15 --local_modality all --batch_size 16 --num_workers 16 &
