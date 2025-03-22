 

model_name=TimeMixer

seq_len=80
e_layers=3
down_sampling_layers=3
down_sampling_window=2
learning_rate=0.01
d_model=32
d_ff=64
batch_size=128

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/csi300/ \
  --data_path csi300.csv \
  --model_id csi300_$seq_len'_'1 \
  --model $model_name \
  --data custom \
  --features M \
  --freq d \
  --target close \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 1 \
  --e_layers $e_layers \
  --d_layers 1 \
  --factor 3 \
  --enc_in 6 \
  --dec_in 6 \
  --c_out 6 \
  --des 'Exp' \
  --itr 1 \
  --batch_size 256 \
  --train_epoch 15\
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/csi300/ \
  --data_path csi300.csv \
  --model_id csi300_$seq_len'_'7 \
  --model $model_name \
  --data custom \
  --features M \
  --freq d \
  --target close \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 7 \
  --e_layers $e_layers \
  --d_layers 1 \
  --factor 3 \
  --enc_in 6 \
  --dec_in 6 \
  --c_out 6 \
  --des 'Exp' \
  --itr 1 \
  --batch_size 256 \
  --train_epoch 15\
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/csi300/ \
  --data_path csi300.csv \
  --model_id csi300_$seq_len'_'80 \
  --model $model_name \
  --data custom \
  --features M \
  --freq d \
  --target close \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 20 \
  --e_layers $e_layers \
  --d_layers 1 \
  --factor 3 \
  --enc_in 6 \
  --dec_in 6 \
  --c_out 6 \
  --des 'Exp' \
  --itr 1 \
  --batch_size 256 \
  --train_epoch 15\
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window
