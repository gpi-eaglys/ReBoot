# MLP-1
python3 reboot_plain.py MNIST --subsample 10000 --resize 14 --num-layers 2 --num-hidden 32 --lr 1e-3 --weight-decay 1e-3 --num-runs 10
python3 reboot_plain.py FASHION_MNIST --subsample 10000 --resize 14 --num-layers 2 --num-hidden 32 --lr 5e-4 --weight-decay 0.0 --num-runs 10
python3 reboot_plain.py KUZUSHIJI_MNIST --subsample 10000 --resize 14 --num-layers 2 --num-hidden 32 --lr 5e-4 --weight-decay 1e-2 --num-runs 10
python3 reboot_plain.py LETTER_RECOGNITION --num-layers 2 --num-hidden 32 --lr 1e-3 --weight-decay 1e-3 --num-runs 10
python3 reboot_plain.py BREAST_CANCER --num-layers 2 --num-hidden 32 --batch-size 16 --lr 5e-3 --weight-decay 0.0  --num-runs 10
python3 reboot_plain.py HEART_DISEASE --subsample 10000 --resize 14 --num-layers 4 --num-hidden 128 64 32 --lr 1e-4 --weight-decay 1e-2 --num-runs 10

# MLP-2
python3 reboot_plain.py MNIST --subsample 10000 --resize 14 --num-layers 3 --num-hidden 64 32 --lr 1e-3 --weight-decay 5e-3 --num-runs 10
python3 reboot_plain.py FASHION_MNIST --subsample 10000 --resize 14 --num-layers 3 --num-hidden 64 32 --lr 5e-4 --weight-decay 0.0 --num-runs 10
python3 reboot_plain.py KUZUSHIJI_MNIST --subsample 10000 --resize 14 --num-layers 3 --num-hidden 64 32 --lr 5e-4 --weight-decay 1e-2 --num-runs 10
python3 reboot_plain.py LETTER_RECOGNITION --num-layers 3 --num-hidden 64 32 --lr 5e-3 --weight-decay 1e-3 --num-runs 10
python3 reboot_plain.py BREAST_CANCER --num-layers 3 --num-hidden 64 32 --batch-size 8 --lr 5e-3 --weight-decay 0.0 --num-runs 10
python3 reboot_plain.py HEART_DISEASE --num-layers 3 --num-hidden 64 32 --batch-size 8 --lr 1e-2 --weight-decay 0.0 --num-runs 10

# MLP-3
python3 reboot_plain.py MNIST --subsample 10000 --resize 14 --num-layers 4 --num-hidden 128 64 32 --lr 5e-4 --weight-decay 1e-3 --num-runs 10
python3 reboot_plain.py FASHION_MNIST --subsample 10000 --resize 14 --num-layers 4 --num-hidden 128 64 32 --lr 1e-3 --weight-decay 5e-3 --num-runs 10
python3 reboot_plain.py KUZUSHIJI_MNIST --subsample 10000 --resize 14 --num-layers 4 --num-hidden 128 64 32 --lr 1e-4 --weight-decay 1e-2 --num-runs 10
python3 reboot_plain.py LETTER_RECOGNITION --num-layers 4 --num-hidden 128 64 32 --lr 1e-3 --weight-decay 0.0 --num-runs 10
python3 reboot_plain.py BREAST_CANCER --num-layers 4 --num-hidden 128 64 32 --batch-size 8 --lr 5e-3 --weight-decay 0.0 --num-runs 10
python3 reboot_plain.py HEART_DISEASE --num-layers 4 --num-hidden 128 64 32 --batch-size 8 --lr 5e-3 --weight-decay 0.0 --num-runs 10
