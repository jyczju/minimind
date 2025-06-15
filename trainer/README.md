cd ~/CodeProjects/minimind

source .venv/bin/activate

cd trainer

python train_pretrain.py --from_checkpoint "../out/pretrain_512.pth"