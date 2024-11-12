 

model_name=FiLM

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/Iron/ \
  --data_path Iron.csv \
  --model_id Iron_96_96 \
  --model $model_name \
  --data custom \
  --target close \
  --features MS \
  --seq_len 192 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --batch_size 4 \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/Iron/ \
  --data_path Iron.csv \
  --model_id Iron_96_192 \
  --model $model_name \
  --data custom \
  --target close \
  --features MS \
  --seq_len 192 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --batch_size 4 \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/Iron/ \
  --data_path Iron.csv \
  --model_id Iron_96_336 \
  --model $model_name \
  --data custom \
  --target close \
  --features MS \
  --seq_len 192 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --batch_size 4 \
  --itr 1

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
  --seq_len 192 \
  --label_len 48 \
  --pred_len 48 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --batch_size 4 \
  --itr 1