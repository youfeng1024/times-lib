 

model_name=PatchTST

for seq_len in 14 28 56
do
    for pred_len in 7 14 21 28
    do
      python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path ./dataset/Iron/ \
        --data_path Iron.csv \
        --model_id Iron_${seq_len}_${pred_len}_${pred_len} \
        --model $model_name \
        --data custom \
        --target close \
        --features MS \
        --seq_len $seq_len \
        --label_len $pred_len \
        --pred_len $pred_len \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --des 'Exp' \
        --batch_size 16 \
        --itr 1 \
        --learning_rate 0.001 \
        --loss 'SMAPE'\
        --embed 'fixed'
  done
done
