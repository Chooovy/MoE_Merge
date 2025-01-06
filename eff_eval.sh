CUDA_VISIBLE_DEVICES=1 python eff_eval.py \
    --device gpu \
    --batch_size 8 \
    --original_len 512 \
    --generated_len 128 \