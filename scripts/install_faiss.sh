#!/usr/bin/env bash

install_openblas() {
    # Get and build OpenBlas (Torch is much better with a decent Blas)
    cd /tmp/
    rm -rf OpenBLAS
    git clone https://github.com/xianyi/OpenBLAS.git
    cd OpenBLAS
    if [ $(getconf _NPROCESSORS_ONLN) == 1 ]; then
        make NO_AFFINITY=1 USE_OPENMP=0 USE_THREAD=0
    else
        make NO_AFFINITY=1 USE_OPENMP=1
    fi
    RET=$?;
    if [ $RET -ne 0 ]; then
        echo "Error. OpenBLAS could not be compiled";
        exit $RET;
    fi
    sudo make install
    RET=$?;
    if [ $RET -ne 0 ]; then
        echo "Error. OpenBLAS could not be installed";
        exit $RET;
    fi
}


# pre-requisites
if [[ -r /usr/bin/pacman ]]; then
  pacman -Syy
  pacman -S --noconfirm git  wget python-pip

elif [[ -r /usr/bin/apt-get ]]; then
  apt-get update
  apt-get install -y git wget  python3-examples python3-pip

elif [[ -r /usr/bin/yum ]]; then
  # cause install-deps supports only fedora v21 and v22
  yum install -y wget cmake curl readline-devel ncurses-devel \
                  gcc-c++ gcc-gfortran git gnuplot unzip \
                  nodejs npm libjpeg-turbo-devel libpng-devel \
                  ImageMagick GraphicsMagick-devel fftw-devel \
                  sox-devel sox qt-devel qtwebkit-devel \
                  python-ipython czmq czmq-devel python3-tools findutils which

  install_openblas

else
  echo "Does not support your distribution, top kek"
  exit 1
fi

wget https://raw.githubusercontent.com/torch/ezinstall/master/install-deps -O ./install-deps
chmod +x ./install-deps
./install-deps

pip3 install --upgrade pip

# clone repo
git clone https://github.com/facebookresearch/faiss.git /faiss
cd /faiss

git checkout e652a6648f52d20d95426d1814737e0e8601f348

# use example makefile
cp ./example_makefiles/makefile.inc.Linux ./makefile.inc

# change stuff in example makefile, for ubuntu 16.04
# arch
if [[ -r /usr/bin/pacman ]]; then
  echo "Arch Linux found"
  pacman -S --noconfirm swig

  # openblas dir
  sed -i "s/lib64\/libopenblas\.so\.0/lib\/libopenblas\.so/g" ./makefile.inc
  # nvcc compiler options
  sed -i "s/std c++11 \-lineinfo/std c++11 \-lineinfo \-Xcompiler \-D__CORRECT_ISO_CPP11_MATH_H_PROTO/g" ./makefile.inc
  # cuda installation root
  sed -i "s/CUDAROOT=\/usr\/local\/cuda-8.0\//CUDAROOT=\/usr\/local\/cuda/g" ./makefile.inc
  # python include directories
  sed -i "s/PYTHONCFLAGS=\-I\/usr\/include\/python2.7\/ \-I\/usr\/lib64\/python2.7\/site\-packages\/numpy\/core\/include\//PYTHONCFLAGS=\-I \/usr\/include\/python3\.6m \-I \/usr\/lib\/python3\.6\/site\-packages\/numpy\/core\/include/g" ./makefile.inc

# ubuntu
elif [[ -r /usr/bin/apt-get ]]; then
  echo "Ubuntu found"
  apt-get -qq update
  apt-get install -y swig

  # openblas dir
  sed -i "s/lib64\/libopenblas\.so\.0/lib\/libopenblas\.so/g" ./makefile.inc
  # nvcc compiler options
  sed -i "s/std c++11 \-lineinfo/std c++11 \-lineinfo \-Xcompiler \-D__CORRECT_ISO_CPP11_MATH_H_PROTO/g" ./makefile.inc
  # cuda installation root
  sed -i "s/CUDAROOT=\/usr\/local\/cuda-8.0\//CUDAROOT=\/usr\/local\/cuda/g" ./makefile.inc
  # python include directories
  sed -i "s/PYTHONCFLAGS=\-I\/usr\/include\/python2.7\/ \-I\/usr\/lib64\/python2.7\/site\-packages\/numpy\/core\/include\//PYTHONCFLAGS=\-I \/usr\/include\/python3\.5 \-I \/usr\/local\/lib\/python3\.5\/dist\-packages\/numpy\/core\/include/g" ./makefile.inc

# fedora
elif [[ -r /usr/bin/yum ]]; then
  echo "Fedora found"
  yum install -y swig

  # openblas dir
  sed -i "s/lib64\/libopenblas\.so\.0/lib\/libopenblas\.so/g" ./makefile.inc
  cp /tmp/OpenBLAS/libopenblas.so /usr/lib/
  # nvcc compiler options
  sed -i "s/std c++11 \-lineinfo/std c++11 \-lineinfo \-Xcompiler \-D__CORRECT_ISO_CPP11_MATH_H_PROTO/g" ./makefile.inc
  # cuda installation root
  sed -i "s/CUDAROOT=\/usr\/local\/cuda-8.0\//CUDAROOT=\/usr\/local\/cuda/g" ./makefile.inc
  # python include directories
  sed -i "s/PYTHONCFLAGS=\-I\/usr\/include\/python2.7\/ \-I\/usr\/lib64\/python2.7\/site\-packages\/numpy\/core\/include\//PYTHONCFLAGS=\-I \/usr\/include\/python3\.6m \-I \/usr\/local\/lib64\/python3\.6\/site\-packages\/numpy\/core\/include/g" ./makefile.inc

# fucked
else
  echo "Does not support your distribution, top kek"
  exit -1
fi

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

