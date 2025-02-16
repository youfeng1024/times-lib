 

model_name=TimeMixer

seq_len=96
e_layers=3
down_sampling_layers=3
down_sampling_window=2
learning_rate=0.01
d_model=32
d_ff=64
batch_size=8


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/csi300_close/ \
  --data_path csi300_close.csv \
  --model_id csi300_close_$seq_len'_'96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 7 \
  --e_layers $e_layers \
  --d_layers 1 \
  --factor 3 \
  --enc_in 215 \
  --dec_in 215 \
  --c_out 215 \
  --des 'Exp' \
  --itr 1 \
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
  --root_path ./dataset/csi300_close/ \
  --data_path csi300_close.csv \
  --model_id csi300_close_$seq_len'_'192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 21 \
  --e_layers $e_layers \
  --d_layers 1 \
  --factor 3 \
  --enc_in 215 \
  --dec_in 215 \
  --c_out 215 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window
