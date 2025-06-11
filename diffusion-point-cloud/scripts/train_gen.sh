python ../train_gen.py \
    --dataset_path "$HOME/autodl-tmp/ShapeNetCore.v2/ShapeNetCore.v2/" \
    --max_iters 20000 \
    --val_freq 5000 \
    --log_root "$HOME/tf-logs" \
    --train_batch_size 128 \
    --categories airplane \
    --tag airplane