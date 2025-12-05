#!/bin/bash
mkdir depends && cd depends
git init
git submodule add https://github.com/scipr-lab/libsnark.git libsnark
git submodule update --init --recursive
sudo apt-get install build-essential cmake git libgmp3-dev libprocps4-dev python-markdown libboost-all-dev libssl-dev pkg-config


cd ..
mkdir build && cd build
cmake ..
make -j$(nproc)


