 

model_name=PatchTST

for seq_len in 10 15 20 25 30
do
  for label_len in 1 4 8 16 32
  do
    for pred_len in 1 2 4
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
        --seq_len $seq_len \
        --label_len $label_len \
        --pred_len $pred_len \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --des 'Exp' \
        --itr 1 \
        --loss 'SMAPE'
    done
  done
done
