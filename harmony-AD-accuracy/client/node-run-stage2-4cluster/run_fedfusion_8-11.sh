CUDA_VISIBLE_DEVICES=0 python3 ./main_fusion_3modal.py --usr_id 8 --local_modality all --batch_size 16 --num_workers 16 &
CUDA_VISIBLE_DEVICES=1 python3 ./main_fusion_3modal.py --usr_id 9 --local_modality all --batch_size 16 --num_workers 16 &
CUDA_VISIBLE_DEVICES=2 python3 ./main_fusion_3modal.py --usr_id 10 --local_modality all --batch_size 16 --num_workers 16 &
CUDA_VISIBLE_DEVICES=3 python3 ./main_fusion_3modal.py --usr_id 11 --local_modality all --batch_size 16 --num_workers 16 &

