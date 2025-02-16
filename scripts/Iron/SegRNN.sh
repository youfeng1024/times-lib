 

model_name=SegRNN

for pred_len in 1 2 6 10
# for pred_len in 336 48
do

python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/Iron/ \
  --data_path Iron.csv \
  --model_id Iron_$seq_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --target close \
  --features MS \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --seg_len 24 \
  --enc_in 7 \
  --d_model 512 \
  --dropout 0 \
  --learning_rate 0.001 \
  --des 'Exp' \
  --itr 1
done

