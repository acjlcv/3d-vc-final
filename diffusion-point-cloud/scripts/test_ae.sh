python ../test_ae.py \
    --ckpt "../pretrained/ckpt_ae_car.pt" \
    --categories car \
    --dataset_path "$HOME/autodl-tmp/ShapeNetCore.v2/ShapeNetCore.v2/" \
    --save_dir "$HOME/tf-logs"\
    --batch_size 128