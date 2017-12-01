#!/usr/bin/env bash

alias sudo=clear

wget https://raw.githubusercontent.com/torch/ezinstall/master/install-deps -O ./install-deps
chmod +x ./install-deps
./install-deps

apt-get install swig

pip3 install --upgrade pip

# clone repo
git clone https://github.com/facebookresearch/faiss.git /faiss
cd /faiss

git checkout e652a6648f52d20d95426d1814737e0e8601f348

# use example makefile
cp ./example_makefiles/makefile.inc.Linux ./makefile.inc

# change stuff in example makefile, for ubuntu 16.04
# openblas dir
sed -i "s/lib64\/libopenblas\.so\.0/lib\/libopenblas\.so/g" ./makefile.inc
# nvcc compiler options
sed -i "s/std c++11 \-lineinfo/std c++11 \-lineinfo \-Xcompiler \-D__CORRECT_ISO_CPP11_MATH_H_PROTO/g" ./makefile.inc
# cuda installation root
sed -i "s/CUDAROOT=\/usr\/local\/cuda-8.0\//CUDAROOT=\/usr\/local\/cuda/g" ./makefile.inc
# python include directories
sed -i "s/PYTHONCFLAGS=\-I\/usr\/include\/python2.7\/ \-I\/usr\/lib64\/python2.7\/site\-packages\/numpy\/core\/include\//PYTHONCFLAGS=\-I \/usr\/include\/python3\.5 \-I \/usr\/local\/lib\/python3\.5\/dist\-packages\/numpy\/core\/include/g" ./makefile.inc

# build
cd /faiss
make
cd gpu
make
cd ..
make py
cd gpu
make py
cd ..

mkdir /tmp/faiss
find -name "*.so" -exec cp {} /tmp/faiss \;
find -name "*.a" -exec cp {} /tmp/faiss \;
find -name "*.py" -exec cp {} /tmp/faiss \;
mv /tmp/faiss .
cd faiss

# convert to python3
2to3 -w ./*.py
rm -rf *.bak

# Fix relative imports
for i in *.py; do
  filename=`echo $i | cut -d "." -f 1`
  echo $filename
  find -name "*.py" -exec sed -i "s/import $filename/import \.$filename/g" {} \;
  find -name "*.py" -exec sed -i "s/from $filename import/from \.$filename import/g" {} \;
done

cd ..

git clone https://github.com/ixaxaar/pytorch-dnc
rm -rf pytorch-dnc/faiss
mv faiss pytorch-dnc
cd pytorch-dnc
pip3 install -e .

