python ../test_gen.py \
    --dataset_path "$HOME/autodl-tmp/ShapeNetCore.v2/ShapeNetCore.v2/" \
    --save_dir "$HOME/autodl-tmp/3d-vc-final/diffusion-point-cloud/tf-logs"\
    --batch_size 128 \
    --ckpt "../pretrained/ckpt_fm_car.pt" \
    --categories car \
    --tag fm
