#! /bin/sh


#compilation
~/llvm-EPI-development-toolchain-cross/bin/clang --target=riscv64-unknown-linux-gnu  -mepi  -v -fno-vectorize -fno-slp-vectorize -c convolutional_inference-debug-longvls.c -o lib.o

#create static library
ar rcs lib.a lib.o

