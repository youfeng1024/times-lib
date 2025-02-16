 

model_name=MultiPatchFormer

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/csi300_close/ \
  --data_path csi300_close.csv \
  --model_id csi300_close_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 21 \
  --label_len 10 \
  --pred_len 7 \
  --e_layers 1 \
  --enc_in 215 \
  --dec_in 215 \
  --c_out 215 \
  --d_model 256 \
  --d_ff 512 \
  --des 'Exp' \
  --n_heads 8 \
  --batch_size 32 \
  --itr 1


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/csi300_close/ \
  --data_path csi300_close.csv \
  --model_id csi300_close_96_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 21 \
  --label_len 10 \
  --pred_len 21 \
  --e_layers 1 \
  --enc_in 215 \
  --dec_in 215 \
  --c_out 215 \
  --d_model 256 \
  --d_ff 512 \
  --des 'Exp' \
  --n_heads 8 \
  --batch_size 32 \
  --itr 1
