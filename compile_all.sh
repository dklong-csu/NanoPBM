#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Valid: Debug, Release, RelWithDebInfo
BUILD_TYPE="Release"
DO_BUILD=0 # True
N_BUILD_JOBS=1
INSTALL_DEPENDENCIES=1 # False
SUNDIALS_LOGGING_LEVEL=2
while getopts ":dwrn:ih" opt; do
    case $opt in 
        d)
        BUILD_TYPE="Debug"
        SUNDIALS_LOGGING_LEVEL=2
        ;;
        w)
        BUILD_TYPE="RelWithDebInfo"
        SUNDIALS_LOGGING_LEVEL=2
        ;;
        r)
        BUILD_TYPE="Release"
        SUNDIALS_LOGGING_LEVEL=0
        ;;
        n)
        N_BUILD_JOBS=$OPTARG 
        ;;
        i)
        INSTALL_DEPENDENCIES=0 # True
        ;;
        h)
        DO_BUILD=1 # False
        echo "---- Instructions: compile_all.sh ----"
        echo "Installs NanoPBM and required dependencies."
        echo 
        echo "  Usage: compile_all.sh [-d] [-w] [-r] [-h]"
        echo 
        echo "  Options:"
        echo "    -d Builds in debug mode"
        echo "    -w Builds in release with debug information mode"
        echo "    -r Builds in release mode"
        echo "    -h Provides usage instructions"
        ;;
        \?)
        echo "Invalid option: -$OPTARG"
        ${SCRIPT_DIR}/compile_all.sh -h
        ;;
    esac
done


if [ $DO_BUILD -eq 0 ]; then

echo "Building in mode: ${BUILD_TYPE}"
echo "Building with ${N_BUILD_JOBS} jobs"


# ---- Install Sundials ----
mkdir -p dependencies
mkdir -p dependencies/sundials
mkdir -p dependencies/install/sundials

SUNDIALS_SRC_DIR=${SCRIPT_DIR}/sundials
SUNDIALS_BUILD_DIR=${SCRIPT_DIR}/dependencies/sundials
SUNDIALS_INSTALL_DIR=${SCRIPT_DIR}/dependencies/install/sundials

if [ $INSTALL_DEPENDENCIES -eq 0 ]; then 

echo "---- Installing SUNDIALS ---- "

cmake -S ${SUNDIALS_SRC_DIR} \
-B ${SUNDIALS_BUILD_DIR} \
-D CMAKE_INSTALL_PREFIX=${SUNDIALS_INSTALL_DIR} \
-D CMAKE_BUILD_TYPE=${BUILD_TYPE} \
-D ENABLE_LAPACK=ON \
-D ENABLE_OPENMP=ON \
-D BUILD_ARKODE=OFF \
-D BUILD_CVODES=OFF \
-D BUILD_IDA=OFF \
-D BUILD_IDAS=OFF \
-D EXAMPLES_ENABLE_CXX=ON \
-D SUNDIALS_LOGGING_LEVEL=${SUNDIALS_LOGGING_LEVEL} \
-D SUNDIALS_BUILD_WITH_MONITORING=ON \
-D CMAKE_CXX_STANDARD=20 
cd ${SUNDIALS_BUILD_DIR} 
make -j ${N_BUILD_JOBS}
make install

fi

# ---- Install NanoPBM ----
cd ${SCRIPT_DIR}
mkdir -p build
NANOPBM_BUILD_DIR=${SCRIPT_DIR}/build
cd ${NANOPBM_BUILD_DIR}
cmake -S ${SCRIPT_DIR} \
-B ${NANOPBM_BUILD_DIR} \
-D CMAKE_BUILD_TYPE=${BUILD_TYPE} \
-D SUNDIALS_DIR=${SUNDIALS_INSTALL_DIR}/lib/cmake/sundials

make -j ${N_BUILD_JOBS}

fi
