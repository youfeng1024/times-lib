
model_name=TSMixer
learning_rate=0.001

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
  --label_len 96 \
  --pred_len 7 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 215 \
  --dec_in 215 \
  --c_out 215 \
  --d_model 512 \
  --d_ff 32 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate $learning_rate \


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
  --label_len 96 \
  --pred_len 21 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 215 \
  --dec_in 215 \
  --c_out 215 \
  --d_model 512 \
  --d_ff 32 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate $learning_rate \
