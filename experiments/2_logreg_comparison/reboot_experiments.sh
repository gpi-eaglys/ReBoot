# MLP-1
python3 reboot_plain.py MNIST --resize 14 --num-layers 2 --num-hidden 32 --lr 1e-3 --weight-decay 0.0 --num-runs 10

# MLP-2
python3 reboot_plain.py MNIST --resize 14 --num-layers 3 --num-hidden 64 32 --lr 1e-3 --weight-decay 5e-3 --num-runs 10

# MLP-3
python3 reboot_plain.py MNIST --resize 14 --num-layers 4 --num-hidden 128 64 32 --lr 1e-3 --weight-decay 1e-2 --num-runs 10
