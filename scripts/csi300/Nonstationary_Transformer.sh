 

model_name=Nonstationary_Transformer

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
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 6 \
  --dec_in 6 \
  --c_out 6 \
  --des 'Exp' \
  --itr 1 \
  --batch_size 256 \
  --train_epoch 15\
  --p_hidden_dims 128 128 \
  --p_hidden_layers 2

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
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 6 \
  --dec_in 6 \
  --c_out 6 \
  --des 'Exp' \
  --itr 1 \
  --batch_size 256 \
  --train_epoch 15\
  --p_hidden_dims 128 128 \
  --p_hidden_layers 2

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
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 6 \
  --dec_in 6 \
  --c_out 6 \
  --des 'Exp' \
  --itr 1 \
  --batch_size 256 \
  --train_epoch 15\
  --p_hidden_dims 128 128 \
  --p_hidden_layers 2
