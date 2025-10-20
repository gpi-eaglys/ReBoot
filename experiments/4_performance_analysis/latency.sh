# Note: running these experiments require manually changing ring_dimension and security_level in fhe_config_mlp_i.yaml files
# Ring 32k
python3 performance_analysis.py CUSTOM --input-dim 64 --output-dim 32 --num-layers 2 --num-hidden 32 --batch-size 1 --num-runs 1 
python3 performance_analysis.py CUSTOM --input-dim 128 --output-dim 64 --num-layers 2 --num-hidden 64 --batch-size 1 --num-runs 1

python3 performance_analysis.py CUSTOM --input-dim 64 --output-dim 32 --num-layers 3 --num-hidden 32 32 --batch-size 1 --num-runs 1 
python3 performance_analysis.py CUSTOM --input-dim 128 --output-dim 64 --num-layers 3 --num-hidden 64 64 --batch-size 1 --num-runs 1

# Ring 64k
python3 performance_analysis.py CUSTOM --input-dim 64 --output-dim 32 --num-layers 2 --num-hidden 32 --batch-size 1 --num-runs 1
python3 performance_analysis.py CUSTOM --input-dim 128 --output-dim 64 --num-layers 2 --num-hidden 64 --batch-size 1 --num-runs 1
python3 performance_analysis.py CUSTOM --input-dim 256 --output-dim 128 --num-layers 2 --num-hidden 128 --batch-size 1 --num-runs 1
python3 performance_analysis.py CUSTOM --input-dim 512 --output-dim 64 --num-layers 2 --num-hidden 64 --batch-size 1 --num-runs 1

python3 performance_analysis.py CUSTOM --input-dim 64 --output-dim 32 --num-layers 3 --num-hidden 32 32 --batch-size 1 --num-runs 1
python3 performance_analysis.py CUSTOM --input-dim 128 --output-dim 64 --num-layers 3 --num-hidden 64 64 --batch-size 1 --num-runs 1
python3 performance_analysis.py CUSTOM --input-dim 256 --output-dim 128 --num-layers 3 --num-hidden 128 128 --batch-size 1 --num-runs 1
python3 performance_analysis.py CUSTOM --input-dim 512 --output-dim 64 --num-layers 3 --num-hidden 64 64 --batch-size 1 --num-runs 1

# Ring 128k
python3 performance_analysis.py CUSTOM --input-dim 256 --output-dim 256 --num-layers 2 --num-hidden 256 --batch-size 1 --num-runs 1
python3 performance_analysis.py CUSTOM --input-dim 512 --output-dim 128 --num-layers 2 --num-hidden 128 --batch-size 1 --num-runs 1

python3 performance_analysis.py CUSTOM --input-dim 256 --output-dim 256 --num-layers 3 --num-hidden 256 256 --batch-size 1 --num-runs 1
python3 performance_analysis.py CUSTOM --input-dim 512 --output-dim 128 --num-layers 3 --num-hidden 128 128 --batch-size 1 --num-runs 1