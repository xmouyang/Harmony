CUDA_VISIBLE_DEVICES=0 python3 ./main_fusion.py --usr_id 0 --local_modality both &
CUDA_VISIBLE_DEVICES=1 python3 ./main_fusion.py --usr_id 1 --local_modality both &
CUDA_VISIBLE_DEVICES=2 python3 ./main_fusion.py --usr_id 2 --local_modality both &
CUDA_VISIBLE_DEVICES=3 python3 ./main_fusion.py --usr_id 3 --local_modality both &
CUDA_VISIBLE_DEVICES=1 python3 ./main_fusion.py --usr_id 4 --local_modality both &
CUDA_VISIBLE_DEVICES=2 python3 ./main_fusion.py --usr_id 5 --local_modality both &
