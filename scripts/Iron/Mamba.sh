model_name=Mamba

for pred_len in 1 2 6 10
# for pred_len in 336 48
do


python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/Iron/ \
  --data_path Iron.csv \
  --model_id Iron_$pred_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --target close \
  --features MS \
  --seq_len $((2 * pred_len)) \
  --label_len $pred_len \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --enc_in 7 \
  --expand 2 \
  --d_ff 16 \
  --d_conv 4 \
  --c_out 7 \
  --d_model 128 \
  --des 'Exp' \
  --itr 1 \

done
