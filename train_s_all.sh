python -m torch.distributed.launch --nproc_per_node=4 --use_env train_distributed_conceptkd.py \
  /home/jupyter/imagenet \
  --config configs/imagenet/vit_mlp.yaml \
  --model deit_tiny_patch16_224 \
  --teacher resnet50 \
  --teacher-pretrained ./teacher_weights/resnet50_a1.pth \
  --distiller conceptkd \
  --concept-stages 1 2 3 4 \
  --concept-mapping-stages stage1 stage2 stage3 stage4 \
  --concept-mapping-temps 0.2 1.0 1.0 0.1 \
  --batch-size 256 \
  --amp \
  --output ./output \
  --experiment t=resnet50_s=deit_tiny_patch16_224_conceptkd
