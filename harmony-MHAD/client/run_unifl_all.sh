CUDA_VISIBLE_DEVICES=0 python3 ./main_unimodal.py --usr_id 0 --local_modality acc &
CUDA_VISIBLE_DEVICES=0 python3 ./main_unimodal.py --usr_id 0 --local_modality skeleton &
CUDA_VISIBLE_DEVICES=1 python3 ./main_unimodal.py --usr_id 1 --local_modality acc &
CUDA_VISIBLE_DEVICES=1 python3 ./main_unimodal.py --usr_id 1 --local_modality skeleton &
CUDA_VISIBLE_DEVICES=2 python3 ./main_unimodal.py --usr_id 2 --local_modality acc &
CUDA_VISIBLE_DEVICES=2 python3 ./main_unimodal.py --usr_id 2 --local_modality skeleton &
CUDA_VISIBLE_DEVICES=3 python3 ./main_unimodal.py --usr_id 3 --local_modality acc &
CUDA_VISIBLE_DEVICES=3 python3 ./main_unimodal.py --usr_id 3 --local_modality skeleton &
CUDA_VISIBLE_DEVICES=4 python3 ./main_unimodal.py --usr_id 4 --local_modality acc &
CUDA_VISIBLE_DEVICES=4 python3 ./main_unimodal.py --usr_id 4 --local_modality skeleton &
CUDA_VISIBLE_DEVICES=5 python3 ./main_unimodal.py --usr_id 5 --local_modality acc &
CUDA_VISIBLE_DEVICES=5 python3 ./main_unimodal.py --usr_id 5 --local_modality skeleton &
CUDA_VISIBLE_DEVICES=6 python3 ./main_unimodal.py --usr_id 6 --local_modality acc &
CUDA_VISIBLE_DEVICES=6 python3 ./main_unimodal.py --usr_id 7 --local_modality acc &
CUDA_VISIBLE_DEVICES=7 python3 ./main_unimodal.py --usr_id 8 --local_modality acc &
CUDA_VISIBLE_DEVICES=7 python3 ./main_unimodal.py --usr_id 9 --local_modality skeleton &
CUDA_VISIBLE_DEVICES=8 python3 ./main_unimodal.py --usr_id 10 --local_modality skeleton &
CUDA_VISIBLE_DEVICES=8 python3 ./main_unimodal.py --usr_id 11 --local_modality skeleton &
