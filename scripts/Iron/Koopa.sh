 

model_name=Koopa

for pred_len in 1 2 6 10
# for pred_len in 336 48
do

python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/Iron/ \
  --data_path Iron.csv \
  --model_id Iron_$pred_len \
  --model $model_name \
  --data custom \
  --target close \
  --features MS \
  --seq_len $((2 * pred_len)) \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --learning_rate 0.001 \
  --itr 1
