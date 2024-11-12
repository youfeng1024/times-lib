 

model_name=Nonstationary_Transformer

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
  --label_len $pred_len \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --p_hidden_dims 256 256 \
  --p_hidden_layers 2 \
  --d_model 2048

python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/Iron/ \
  --data_path Iron.csv \
  --model_id Iron_96_192 \
  --model $model_name \
  --data custom \
  --target close \
  --features MS \
  --seq_len $((2 * pred_len)) \
  --label_len $pred_len \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --p_hidden_dims 256 256 \
  --p_hidden_layers 2 \
  --d_model 2048

python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/Iron/ \
  --data_path Iron.csv \
  --model_id Iron_96_336 \
  --model $model_name \
  --data custom \
  --target close \
  --features MS \
  --seq_len $((2 * pred_len)) \
  --label_len $pred_len \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --p_hidden_dims 256 256 \
  --p_hidden_layers 2 \
  --d_model 2048

python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/Iron/ \
  --data_path Iron.csv \
  --model_id Iron_96_48 \
  --model $model_name \
  --data custom \
  --target close \
  --features MS \
  --seq_len $((2 * pred_len)) \
  --label_len $pred_len \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --p_hidden_dims 256 256 \
  --p_hidden_layers 2 \
  --d_model 2048
