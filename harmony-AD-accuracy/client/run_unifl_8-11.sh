CUDA_VISIBLE_DEVICES=0 python3 ./main_unimodal.py --usr_id 8 --local_modality audio --batch_size 8 --num_workers 8 &
CUDA_VISIBLE_DEVICES=0 python3 ./main_unimodal.py --usr_id 8 --local_modality depth --batch_size 8 --num_workers 8 &
CUDA_VISIBLE_DEVICES=0 python3 ./main_unimodal.py --usr_id 8 --local_modality radar --batch_size 8 --num_workers 8 &
CUDA_VISIBLE_DEVICES=1 python3 ./main_unimodal.py --usr_id 9 --local_modality audio --batch_size 8 --num_workers 8 &
CUDA_VISIBLE_DEVICES=1 python3 ./main_unimodal.py --usr_id 9 --local_modality depth --batch_size 8 --num_workers 8 &
CUDA_VISIBLE_DEVICES=1 python3 ./main_unimodal.py --usr_id 9 --local_modality radar --batch_size 8 --num_workers 8 &
CUDA_VISIBLE_DEVICES=2 python3 ./main_unimodal.py --usr_id 10 --local_modality audio --batch_size 8 --num_workers 8 &
CUDA_VISIBLE_DEVICES=2 python3 ./main_unimodal.py --usr_id 10 --local_modality depth --batch_size 8 --num_workers 8 &
CUDA_VISIBLE_DEVICES=2 python3 ./main_unimodal.py --usr_id 10 --local_modality radar --batch_size 8 --num_workers 8 &
CUDA_VISIBLE_DEVICES=3 python3 ./main_unimodal.py --usr_id 11 --local_modality audio --batch_size 8 --num_workers 8 &
CUDA_VISIBLE_DEVICES=3 python3 ./main_unimodal.py --usr_id 11 --local_modality depth --batch_size 8 --num_workers 8 &
CUDA_VISIBLE_DEVICES=3 python3 ./main_unimodal.py --usr_id 11 --local_modality radar --batch_size 8 --num_workers 8 &

