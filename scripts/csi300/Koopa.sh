 

model_name=Koopa

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/csi300_close/ \
  --data_path csi300_close.csv \
  --model_id csi300_close_96_48 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 21 \
  --pred_len 48 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 215 \
  --dec_in 215 \
  --c_out 215 \
  --des 'Exp' \
  --learning_rate 0.001 \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/csi300_close/ \
  --data_path csi300_close.csv \
  --model_id csi300_close_192_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 192 \
  --pred_len 7 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 215 \
  --dec_in 215 \
  --c_out 215 \
  --des 'Exp' \
  --learning_rate 0.001 \
  --itr 1
