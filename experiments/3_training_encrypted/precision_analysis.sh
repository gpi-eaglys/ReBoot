# MLP-1
python3 precision_analysis.py MNIST --subsample 4800 --num-layers 2 --num-hidden 32 --batch-size 48  --num-epochs 10 --seed 0
python3 precision_analysis.py MNIST --subsample 4800 --num-layers 2 --num-hidden 32 --batch-size 48  --num-epochs 10 --seed 1
python3 precision_analysis.py MNIST --subsample 4800 --num-layers 2 --num-hidden 32 --batch-size 48  --num-epochs 10 --seed 2

# MLP-2
python3 precision_analysis.py MNIST --subsample 4800 --num-layers 3 --num-hidden 64 32 --batch-size 48 --num-epochs 10 --seed 0
python3 precision_analysis.py MNIST --subsample 4800 --num-layers 3 --num-hidden 64 32 --batch-size 48 --num-epochs 10 --seed 1
python3 precision_analysis.py MNIST --subsample 4800 --num-layers 3 --num-hidden 64 32 --batch-size 48 --num-epochs 10 --seed 2

# MLP-3
python3 precision_analysis.py MNIST --subsample 4800 --num-layers 4 --num-hidden 128 64 32 --batch-size 48 --num-epochs 10 --seed 0
python3 precision_analysis.py MNIST --subsample 4800 --num-layers 4 --num-hidden 128 64 32 --batch-size 48 --num-epochs 10 --seed 1
python3 precision_analysis.py MNIST --subsample 4800 --num-layers 4 --num-hidden 128 64 32 --batch-size 48 --num-epochs 10 --seed 2
