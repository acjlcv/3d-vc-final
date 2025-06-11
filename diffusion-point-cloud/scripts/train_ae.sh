python ../train_ae.py \
    --dataset_path "$HOME/autodl-tmp/ShapeNetCore.v2/ShapeNetCore.v2/" \
    --max_iters 10000 \
    --val_freq 1000 \
    --categories car \
    --log_root "$HOME/tf-logs" \
    --train_batch_size 128 \

