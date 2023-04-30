CUDA_VISIBLE_DEVICES=0 python3 ./main_unimodal.py --usr_id 4 --local_modality radar --batch_size 8 --num_workers 8 &
CUDA_VISIBLE_DEVICES=1 python3 ./main_unimodal.py --usr_id 5 --local_modality radar --batch_size 8 --num_workers 8 &
CUDA_VISIBLE_DEVICES=2 python3 ./main_unimodal.py --usr_id 6 --local_modality audio --batch_size 8 --num_workers 8 &
CUDA_VISIBLE_DEVICES=2 python3 ./main_unimodal.py --usr_id 6 --local_modality depth --batch_size 8 --num_workers 8 &
CUDA_VISIBLE_DEVICES=2 python3 ./main_unimodal.py --usr_id 6 --local_modality radar --batch_size 8 --num_workers 8 &
CUDA_VISIBLE_DEVICES=3 python3 ./main_unimodal.py --usr_id 7 --local_modality audio --batch_size 8 --num_workers 8 &
CUDA_VISIBLE_DEVICES=3 python3 ./main_unimodal.py --usr_id 7 --local_modality depth --batch_size 8 --num_workers 8 &
CUDA_VISIBLE_DEVICES=3 python3 ./main_unimodal.py --usr_id 7 --local_modality radar --batch_size 8 --num_workers 8 &