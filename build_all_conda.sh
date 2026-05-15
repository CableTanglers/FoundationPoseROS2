#!/bin/bash

PROJ_ROOT=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Install dependencies
pip install torchvision==0.16.0+cu121 torchaudio==2.1.0 torch==2.1.0+cu121 --index-url https://download.pytorch.org/whl/cu121
# AIC PATCH (HUNK 11 — build isolation guard): pytorch3d's setup.py imports
# torch at build time. PEP 517 build isolation (pip 23+) creates a fresh
# build env without torch, causing ModuleNotFoundError. --no-build-isolation
# reuses the conda env's torch. Same guard applied to mycuda below.
pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable"
python -m pip install -r requirements.txt

# Clone source repository of FoundationPose
git clone https://github.com/NVlabs/FoundationPose.git

# HUNK 10: README L81-82 instructs a manual C++14 -> C++17 edit at
# FoundationPose/bundlesdf/mycuda/setup.py L18-19 (nvcc_flags + c_flags).
# Required on Ada/Hopper GPUs (sm_89/sm_90+) where the C++14 ABI rejects
# newer torch + nvcc combos. We sed it inline so the Docker build is
# unattended.
_MYCUDA_SETUP=FoundationPose/bundlesdf/mycuda/setup.py
if [ -f "${_MYCUDA_SETUP}" ]; then
    sed -i 's/c++14/c++17/g; s/std=c++14/std=c++17/g' "${_MYCUDA_SETUP}"
    echo "HUNK 10: rewrote C++14 -> C++17 in ${_MYCUDA_SETUP}"
fi

# Create the weights directory and download the pretrained weights from FoundationPose
gdown --folder https://drive.google.com/drive/folders/1BEQLZH69UO5EOfah-K9bfI3JyP9Hf7wC -O FoundationPose/weights/2023-10-28-18-33-37 
gdown --folder https://drive.google.com/drive/folders/12Te_3TELLes5cim1d7F7EBTwUSe7iRBj -O FoundationPose/weights/2024-01-11-20-02-45

# Install pybind11
cd ${PROJ_ROOT}/FoundationPose && git clone https://github.com/pybind/pybind11 && \
    cd pybind11 && git checkout v2.10.0 && \
    mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DPYBIND11_INSTALL=ON -DPYBIND11_TEST=OFF && \
    sudo make -j6 && sudo make install

# Install Eigen
cd ${PROJ_ROOT}/FoundationPose && wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz && \
    tar xvzf ./eigen-3.4.0.tar.gz && rm ./eigen-3.4.0.tar.gz && \
    cd eigen-3.4.0 && \
    mkdir build && \
    cd build && \
    cmake .. && \
    sudo make install

# Clone and install nvdiffrast
# AIC PATCH (HUNK 11): upstream `cd /nvdiffrast` is an absolute-path bug —
# nvdiffrast was just git-cloned into ${PROJ_ROOT}/FoundationPose/nvdiffrast,
# not /nvdiffrast. Use the correct relative path.
cd ${PROJ_ROOT}/FoundationPose && git clone https://github.com/NVlabs/nvdiffrast && \
    cd nvdiffrast && pip install --no-build-isolation .

# Install mycpp
cd ${PROJ_ROOT}/FoundationPose/mycpp/ && \
rm -rf build && mkdir -p build && cd build && \
cmake .. && \
sudo make -j$(nproc)

# Install mycuda
# AIC PATCH (HUNK 11): same --no-build-isolation guard as pytorch3d above.
# mycuda's setup.py imports torch + nvcc helpers at build time; PEP 517 build
# isolation kills it.
cd ${PROJ_ROOT}/FoundationPose/bundlesdf/mycuda && \
rm -rf build *egg* *.so && \
python3 -m pip install --no-build-isolation -e .

cd ${PROJ_ROOT}
