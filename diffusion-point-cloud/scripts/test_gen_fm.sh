python ../test_gen.py \
    --dataset_path "$HOME/autodl-tmp/ShapeNetCore.v2/ShapeNetCore.v2/" \
    --save_dir "$HOME/tf-logs"\
    --batch_size 128 \
    --ckpt "../pretrained/ckpt_fm_table.pt" \
    --categories table \
    --tag fm
