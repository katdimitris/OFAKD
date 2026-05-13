python -m torch.distributed.launch --nproc_per_node=4 --use_env train_distributed.py \
  /home/jupyter/imagenet \
  --config configs/imagenet/vit_mlp.yaml \
  --model deit_tiny_patch16_224 \
  --teacher resnet50 \
  --teacher-pretrained ./teacher_weights/resnet50_a1.pth \
  --distiller kd \
  --kd-temperature 1 \
  --gt-loss-weight 1 \
  --kd-loss-weight 1 \
  --batch-size 256 \
  --amp \
  --output ./output \
  --experiment t=resnet50_s=deit_tiny_patch16_224_kd