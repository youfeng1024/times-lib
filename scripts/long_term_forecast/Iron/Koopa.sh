export CUDA_VISIBLE_DEVICES=0

model_name=Koopa

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/Iron/ \
  --data_path Iron.csv \
  --model_id Iron_96_48 \
  --model $model_name \
  --data custom \
  --target close \
  --features MS \
  --seq_len 96 \
  --pred_len 48 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --learning_rate 0.001 \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/Iron/ \
  --data_path Iron.csv \
  --model_id Iron_192_96 \
  --model $model_name \
  --data custom \
  --target close \
  --features MS \
  --seq_len 192 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --learning_rate 0.001 \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/Iron/ \
  --data_path Iron.csv \
  --model_id Iron_288_144 \
  --model $model_name \
  --data custom \
  --target close \
  --features MS \
  --seq_len 288 \
  --pred_len 144 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --learning_rate 0.001 \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/Iron/ \
  --data_path Iron.csv \
  --model_id Iron_384_192 \
  --model $model_name \
  --data custom \
  --target close \
  --features MS \
  --seq_len 384 \
  --pred_len 192 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --learning_rate 0.001 \
  --itr 1