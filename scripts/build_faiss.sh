#!/bin/bash
set -e

# Script to compile FAISS with CUDA and cuBLAS support directly into current virtual environment
# This script will use the currently active Python environment

# Check if a virtual environment is active
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Error: No active Python virtual environment detected."
    echo "Please activate a virtual environment before running this script."
    exit 1
fi

echo "Using Python virtual environment at: $VIRTUAL_ENV"

# Configuration
FAISS_VERSION="1.10.0" # Latest stable version
BUILD_DIR="$(pwd)/faiss_build"
INSTALL_DIR="$VIRTUAL_ENV" # Install directly to virtual environment
CUDA_ARCHS="75"            # For compute capability 7.5, use comma-separated list for multiple archs
PYTHON_BINDINGS="ON"       # Set to OFF to disable Python bindings
USE_MKL="OFF"              # Set to ON to use Intel MKL (recommended for performance)
USE_CUVS="OFF"             # Set to ON to enable NVIDIA cuVS implementations
OPT_LEVEL="avx2"           # Options: generic, avx2, avx512, avx512_spr (x86-64) or generic, sve (aarch64)
ENABLE_GPU="ON"            # Set to OFF for CPU-only build
BUILD_TYPE="Release"       # Options: Release, Debug
ENABLE_TESTING="OFF"       # Set to ON to build tests
BUILD_SHARED="ON"          # Set to ON for shared libraries, OFF for static
PARALLEL_JOBS=$(nproc)     # Number of parallel build jobs

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Print configuration
echo -e "${GREEN}FAISS Build Configuration:${NC}"
echo -e "  FAISS version: ${YELLOW}${FAISS_VERSION}${NC}"
echo -e "  Build directory: ${YELLOW}${BUILD_DIR}${NC}"
echo -e "  Install directory: ${YELLOW}${INSTALL_DIR}${NC}"
echo -e "  CUDA architectures: ${YELLOW}${CUDA_ARCHS}${NC}"
echo -e "  Python bindings: ${YELLOW}${PYTHON_BINDINGS}${NC}"
echo -e "  Use Intel MKL: ${YELLOW}${USE_MKL}${NC}"
echo -e "  Use NVIDIA cuVS: ${YELLOW}${USE_CUVS}${NC}"
echo -e "  Optimization level: ${YELLOW}${OPT_LEVEL}${NC}"
echo -e "  Enable GPU: ${YELLOW}${ENABLE_GPU}${NC}"
echo -e "  Build type: ${YELLOW}${BUILD_TYPE}${NC}"
echo -e "  Enable testing: ${YELLOW}${ENABLE_TESTING}${NC}"
echo -e "  Build shared libraries: ${YELLOW}${BUILD_SHARED}${NC}"
echo -e "  Parallel jobs: ${YELLOW}${PARALLEL_JOBS}${NC}"

# Function to check for required tools
check_requirements() {
    echo -e "${GREEN}Checking for required tools...${NC}"
    local missing_tools=()

    # Check for C++ compiler
    if ! command -v g++ &>/dev/null && ! command -v clang++ &>/dev/null; then
        missing_tools+=("C++17 compiler (g++ or clang++)")
    else
        echo -e "  C++ compiler: ${YELLOW}$(command -v g++ 2>/dev/null || command -v clang++)${NC}"
    fi

    # Check for CMake
    if ! command -v cmake &>/dev/null; then
        missing_tools+=("CMake")
    else
        echo -e "  CMake: ${YELLOW}$(cmake --version | head -n1)${NC}"
    fi

    # Check for Git
    if ! command -v git &>/dev/null; then
        missing_tools+=("Git")
    else
        echo -e "  Git: ${YELLOW}$(git --version)${NC}"
    fi

    # Check for CUDA if GPU is enabled
    if [ "$ENABLE_GPU" = "ON" ]; then
        if ! command -v nvcc &>/dev/null; then
            missing_tools+=("CUDA toolkit (nvcc)")
        else
            echo -e "  CUDA: ${YELLOW}$(nvcc --version | grep release)${NC}"
        fi
    fi

    # Check for Python and NumPy if Python bindings are enabled
    if [ "$PYTHON_BINDINGS" = "ON" ]; then
        # Check for NumPy
        if ! python -c "import numpy" &>/dev/null; then
            missing_tools+=("NumPy (Python package)")
        else
            echo -e "  NumPy: ${YELLOW}$(python -c "import numpy; print(numpy.__version__)")${NC}"
        fi

        # Check for SWIG
        if ! command -v swig &>/dev/null; then
            missing_tools+=("SWIG")
        else
            echo -e "  SWIG: ${YELLOW}$(swig -version | head -n2 | tail -n1)${NC}"
        fi
    fi

    # Report missing tools
    if [ ${#missing_tools[@]} -gt 0 ]; then
        echo -e "${RED}Missing required tools:${NC}"
        for tool in "${missing_tools[@]}"; do
            echo -e "  - ${RED}${tool}${NC}"
        done
        echo -e "${RED}Please install the missing tools and try again.${NC}"
        exit 1
    fi

    echo -e "${GREEN}All required tools are installed.${NC}"
}

# Clone or update FAISS repository
clone_or_update_faiss() {
    echo -e "${GREEN}Cloning or updating FAISS repository...${NC}"

    mkdir -p "$BUILD_DIR"

    if [ ! -d "$BUILD_DIR/faiss" ]; then
        cd "$BUILD_DIR"
        echo -e "${GREEN}Cloning FAISS repository...${NC}"
        git clone https://github.com/facebookresearch/faiss.git
        cd faiss
        if [ -n "$FAISS_VERSION" ]; then
            echo -e "${GREEN}Checking out version ${FAISS_VERSION}...${NC}"
            git checkout "v${FAISS_VERSION}" || git checkout "${FAISS_VERSION}"
        fi
    else
        echo -e "${GREEN}FAISS repository already exists, updating...${NC}"
        cd "$BUILD_DIR/faiss"
        git fetch
        if [ -n "$FAISS_VERSION" ]; then
            echo -e "${GREEN}Checking out version ${FAISS_VERSION}...${NC}"
            git checkout "v${FAISS_VERSION}" || git checkout "${FAISS_VERSION}"
        fi
    fi
}

# Configure build with CMake
configure_build() {
    echo -e "${GREEN}Configuring build with CMake...${NC}"

    mkdir -p "$BUILD_DIR/faiss/build"
    cd "$BUILD_DIR/faiss/build"

    # Get current Python executable and site-packages directory
    PYTHON_EXECUTABLE=$(which python)
    PYTHON_SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")

    echo -e "${GREEN}Using Python: ${YELLOW}${PYTHON_EXECUTABLE}${NC}"
    echo -e "${GREEN}Python site-packages: ${YELLOW}${PYTHON_SITE_PACKAGES}${NC}"

    # Prepare CMake arguments
    CMAKE_ARGS=(
        "-DCMAKE_BUILD_TYPE=${BUILD_TYPE}"
        "-DFAISS_ENABLE_GPU=${ENABLE_GPU}"
        "-DFAISS_ENABLE_PYTHON=${PYTHON_BINDINGS}"
        "-DBUILD_TESTING=${ENABLE_TESTING}"
        "-DBUILD_SHARED_LIBS=${BUILD_SHARED}"
        "-DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}"
        "-DFAISS_OPT_LEVEL=${OPT_LEVEL}"
        "-DPython_EXECUTABLE=${PYTHON_EXECUTABLE}"
    )

    # Add CUDA architectures if GPU is enabled
    if [ "$ENABLE_GPU" = "ON" ]; then
        CMAKE_ARGS+=("-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHS}")
    fi

    # Add cuVS support if enabled
    if [ "$USE_CUVS" = "ON" ]; then
        CMAKE_ARGS+=("-DFAISS_ENABLE_CUVS=ON")
    fi

    # Add MKL support if enabled
    if [ "$USE_MKL" = "ON" ]; then
        CMAKE_ARGS+=("-DBLA_VENDOR=Intel10_64_dyn")
    fi

    # Run CMake
    cmake "${CMAKE_ARGS[@]}" ..
}

# Build FAISS
build_faiss() {
    echo -e "${GREEN}Building FAISS...${NC}"
    cd "$BUILD_DIR/faiss/build"

    # Build C++ library
    echo -e "${GREEN}Building C++ library...${NC}"
    make -j"${PARALLEL_JOBS}" faiss

    # Build optimized versions if needed
    if [ "$OPT_LEVEL" = "avx2" ]; then
        echo -e "${GREEN}Building AVX2 optimized version...${NC}"
        make -j"${PARALLEL_JOBS}" faiss_avx2
    elif [ "$OPT_LEVEL" = "avx512" ]; then
        echo -e "${GREEN}Building AVX512 optimized version...${NC}"
        make -j"${PARALLEL_JOBS}" faiss_avx512
    elif [ "$OPT_LEVEL" = "avx512_spr" ]; then
        echo -e "${GREEN}Building AVX512 Sapphire Rapids optimized version...${NC}"
        make -j"${PARALLEL_JOBS}" faiss_avx512_spr
    fi

    # Build Python bindings if enabled
    if [ "$PYTHON_BINDINGS" = "ON" ]; then
        echo -e "${GREEN}Building Python bindings...${NC}"
        make -j"${PARALLEL_JOBS}" swigfaiss
    fi
}

# Install FAISS
install_faiss() {
    echo -e "${GREEN}Installing FAISS...${NC}"
    cd "$BUILD_DIR/faiss/build"

    # Install C++ library and headers
    echo -e "${GREEN}Installing C++ library and headers...${NC}"
    if [ "$BUILD_SHARED" = "ON" ]; then
        # First copy shared libraries to the virtual environment's lib directory
        mkdir -p "$VIRTUAL_ENV/lib"
        find . -name "*.so*" -not -path "*python*" -type f -exec cp -v {} "$VIRTUAL_ENV/lib/" \;
    fi

    # Install Python bindings if enabled
    if [ "$PYTHON_BINDINGS" = "ON" ]; then
        echo -e "${GREEN}Installing Python bindings...${NC}"
        cd "$BUILD_DIR/faiss/build/faiss/python"
        python setup.py install --prefix="$VIRTUAL_ENV"
    fi
}

# Run tests if enabled
run_tests() {
    if [ "$ENABLE_TESTING" = "ON" ]; then
        echo -e "${GREEN}Running tests...${NC}"
        cd "$BUILD_DIR/faiss/build"
        make test
    fi
}

# Create a test script
create_test_script() {
    echo -e "${GREEN}Creating test script...${NC}"

    mkdir -p "$BUILD_DIR"

    # Create Python test script
    cat >"$BUILD_DIR/test_faiss.py" <<EOF
import numpy as np
import faiss

print("FAISS version:", faiss.__version__)

# Create a small dataset
d = 64                           # dimension
nb = 10000                       # database size
nq = 1000                        # queries size
np.random.seed(1234)             # make reproducible
xb = np.random.random((nb, d)).astype('float32')
xq = np.random.random((nq, d)).astype('float32')

# Create an index
index = faiss.IndexFlatL2(d)     # build the index
print(f"Index is trained: {index.is_trained}")
index.add(xb)                    # add vectors to the index
print(f"Index contains {index.ntotal} vectors")

# Search
k = 4                            # we want 4 nearest neighbors
D, I = index.search(xq[:5], k)   # search 5 vectors
print(f"Search results distances: \n{D}")
print(f"Search results indices: \n{I}")

# Test GPU if available
try:
    print("\nTesting GPU support...")
    res = faiss.StandardGpuResources()  # use a single GPU
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
    D_gpu, I_gpu = gpu_index.search(xq[:5], k)
    print(f"GPU search results distances: \n{D_gpu}")
    print(f"GPU search results indices: \n{I_gpu}")
    print("GPU support is working")
except Exception as e:
    print(f"GPU support not available or not working: {e}")
EOF

    echo -e "${GREEN}Test script created at: ${YELLOW}${BUILD_DIR}/test_faiss.py${NC}"
    echo -e "${GREEN}You can run it with: ${YELLOW}python ${BUILD_DIR}/test_faiss.py${NC}"
}

# Main execution
check_requirements
clone_or_update_faiss
configure_build
build_faiss
install_faiss
run_tests
create_test_script

echo -e "${GREEN}FAISS build completed successfully!${NC}"
echo -e "${GREEN}FAISS installed directly to your virtual environment: ${YELLOW}${VIRTUAL_ENV}${NC}"

if [ "$PYTHON_BINDINGS" = "ON" ]; then
    echo -e "${GREEN}Python bindings have been installed to your virtual environment.${NC}"
    echo -e "${GREEN}To test your installation, run: ${YELLOW}python ${BUILD_DIR}/test_faiss.py${NC}"
fi
