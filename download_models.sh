# venv
python3 -m venv .venv
source .venv/bin/activate
pip install tensorboard
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install timm==0.6.5 pyyaml

# resnet50
mkdir -p teacher_weights
cd teacher_weights
wget -O resnet50_a1.pth https://huggingface.co/timm/resnet50.a1_in1k/resolve/main/pytorch_model.bin