 

model_name=iTransformer

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/csi300/ \
  --data_path csi300.csv \
  --model_id csi300_80_1 \
  --model $model_name \
  --data custom \
  --features M \
  --freq d \
  --target close \
  --seq_len 80 \
  --label_len 40 \
  --pred_len 1 \
  --e_layers 4 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 6 \
  --dec_in 6 \
  --c_out 6 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --learning_rate 0.001 \
  --itr 1 \
  --batch_size 256 \
  --train_epoch 15

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/csi300/ \
  --data_path csi300.csv \
  --model_id csi300_80_7 \
  --model $model_name \
  --data custom \
  --features M \
  --freq d \
  --target close \
  --seq_len 80 \
  --label_len 40 \
  --pred_len 7 \
  --e_layers 4 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 6 \
  --dec_in 6 \
  --c_out 6 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --learning_rate 0.001 \
  --itr 1 \
  --batch_size 256 \
  --train_epoch 15

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/csi300/ \
  --data_path csi300.csv \
  --model_id csi300_80_80 \
  --model $model_name \
  --data custom \
  --features M \
  --freq d \
  --target close \
  --seq_len 80 \
  --label_len 40 \
  --pred_len 20 \
  --e_layers 4 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 6 \
  --dec_in 6 \
  --c_out 6 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --learning_rate 0.001 \
  --itr 1 \
  --batch_size 256
