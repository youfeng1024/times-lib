 

model_name=TimeXer

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/csi300_close/ \
  --data_path csi300_close.csv \
  --model_id csi300_close_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 21 \
  --label_len 10 \
  --pred_len 7 \
  --e_layers 3 \
  --factor 3 \
  --enc_in 215 \
  --dec_in 215 \
  --c_out 215 \
  --d_model 512 \
  --d_ff 512 \
  --des 'Exp' \
  --batch_size 16 \
  --learning_rate 0.001 \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/csi300_close/ \
  --data_path csi300_close.csv \
  --model_id csi300_close_96_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 21 \
  --label_len 10 \
  --pred_len 21 \
  --e_layers 3 \
  --factor 3 \
  --enc_in 215 \
  --dec_in 215 \
  --c_out 215 \
  --d_model 512 \
  --d_ff 512 \
  --des 'Exp' \
  --batch_size 16 \
  --learning_rate 0.001 \
  --itr 1
