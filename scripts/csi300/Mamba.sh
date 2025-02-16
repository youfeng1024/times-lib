model_name=Mamba

for pred_len in 96 192 
do

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/csi300_close/ \
  --data_path csi300_close.csv \
  --model_id csi300_close_$pred_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $pred_len \
  --label_len 10 \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --enc_in 215 \
  --expand 2 \
  --d_ff 16 \
  --d_conv 4 \
  --c_out 215 \
  --d_model 128 \
  --des 'Exp' \
  --itr 1 \

done