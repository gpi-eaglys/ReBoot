#/bin/bash

# Install the OpenFHE-Python library
rm -r "/workspaces/ReBoot/openfhe-python"
cp -r "/openfhe-python" "/workspaces/ReBoot"
python3 -m pip install -e openfhe-python

# Install additional requirements
python3 -m pip install -U setuptools pip
python3 -m pip install -r requirements.txt
python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Compile the C++ library into a Python module (Release, for running experiments)
cd "/workspaces/ReBoot/cpp"
mkdir build
cd build
cmake ..
make -j$(nproc)

# Compile a Debug variant (unoptimized, with symbols) for stepping through with gdb
cd "/workspaces/ReBoot/cpp"
mkdir build-debug
cd build-debug
cmake -DCMAKE_BUILD_TYPE=Debug ..
make -j$(nproc)

# Install tools
apt update
apt install -y htop btop tree wget unzip zip tmux gdb
