# Yoo & Yoon
python3 performance_analysis.py ONE_CLASS --num-layers 2 --num-hidden 1 --batch-size 1 --num-runs 1
python3 performance_analysis.py ONE_CLASS --num-layers 2 --num-hidden 32 --batch-size 1 --num-runs 1

# Colombo et al.
python3 performance_analysis.py PENGUINS --num-layers 2 --num-hidden 2 --batch-size 1 --num-runs 1
python3 performance_analysis.py PENGUINS --num-layers 2 --num-hidden 32 --batch-size 1 --num-runs 1

python3 performance_analysis.py TMNIST --num-layers 3 --num-hidden 4 2 --batch-size 1 --num-runs 1
python3 performance_analysis.py TMNIST --num-layers 3 --num-hidden 64 32 --batch-size 1 --num-runs 1