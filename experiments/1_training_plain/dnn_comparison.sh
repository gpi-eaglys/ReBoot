# Mihara et al.
python3 reboot_plain.py IRIS --num-layers 2 --num-hidden 10 --batch-size 8 --lr 5e-3 --weight-decay 0.0 --num-runs 10
python3 reboot_plain.py IRIS --num-layers 2 --num-hidden 32 --batch-size 8 --lr 5e-3 --weight-decay 0.0 --num-runs 10

# Montero et al.
python3 reboot_plain.py BREAST_CANCER --num-layers 2 --num-hidden 29 --batch-size 16 --lr 5e-3 --weight-decay 0.0 --num-runs 10
python3 reboot_plain.py BREAST_CANCER --num-layers 2 --num-hidden 32 --batch-size 16 --lr 1e-3 --weight-decay 1e-2 --num-runs 10

# Lou et al.
python3 reboot_plain.py MNIST --num-layers 3 --num-hidden 128 32 --batch-size 60 --lr 5e-4 --weight-decay 1e-3 --num-runs 10

# Nandakumar et al.
python3 reboot_plain.py MNIST --resize 8 --num-layers 3 --num-hidden 32 16 --lr 5e-4 --weight-decay 1e-3 --num-runs 10
python3 reboot_plain.py MNIST --resize 8 --num-layers 3 --num-hidden 64 32 --batch-size 60 --lr 5e-4 --weight-decay 1e-3 --num-runs 10

# Colombo et al.
python3 reboot_plain.py TMNIST --num-layers 3 --num-hidden 4 2 --batch-size 10 --lr 1e-3 --weight-decay 1e-3  --num-runs 10 
python3 reboot_plain.py TMNIST --num-layers 2 --num-hidden 32 --batch-size 8 --lr 1e-3 --weight-decay 0.0 --num-runs 10 
python3 reboot_plain.py FASHION_MNIST --num-layers 2 --num-hidden 200 --lr 5e-4 --weight-decay 0.0 --num-runs 10
python3 reboot_plain.py PENGUINS --num-layers 2 --num-hidden 2 --batch-size 16 --lr 5e-3 --weight-decay 1e-3 --num-runs 10
python3 reboot_plain.py PENGUINS --num-layers 2 --num-hidden 32 --batch-size 8 --lr 5e-4 --weight-decay 0.0 --num-runs 10