#!/usr/bin/env bash
set -e

echo "======================================="
echo " robotoc_quad_mini Local Setup"
echo " Ubuntu 22.04 required"
echo "======================================="

# ---------- System deps (minimal) ----------
sudo apt update
sudo apt install -y \
    build-essential \
    cmake \
    git \
    curl \
    python3-venv \
    python3-dev \
    libeigen3-dev \
    robotpkg-pinocchio

# ---------- Create Python venv ----------
python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
# IMPORTANT: lock numpy to <2
pip install "numpy<2" pybullet

# ---------- Clone Robotoc locally ----------
mkdir -p external
cd external

if [ ! -d "robotoc" ]; then
  git clone https://github.com/mayataka/robotoc.git
  git checkout c0ebe305af90a75d258ee728782615ac0367c725
fi

cd robotoc
mkdir -p build
cd build

# ---------- Local install prefix ----------
INSTALL_DIR=$(pwd)/../../install

export CMAKE_PREFIX_PATH=/opt/openrobots
export PKG_CONFIG_PATH=/opt/openrobots/lib/pkgconfig

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=../../install \
    -DCMAKE_PREFIX_PATH=/opt/openrobots

make -j4
make install

cd ../../../

echo ""
echo "======================================="
echo " Setup Complete."
echo ""
echo " To run:"
echo " source activate_env.sh"
echo " ./run.sh"
echo "======================================="
