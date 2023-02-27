
CC=~/llvm-EPI-development-toolchain-cross/bin/clang
CFLAGS=--target=riscv64-unknown-linux-gnu  -mepi  -v -fno-vectorize -fno-slp-vectorize
#compilation and create static library
lib.o:
	${CC} ${CFLAGS} -c convolutional_inference.c -o lib.o 
	ar rcs lib.a lib.o

clean:
	rm -rf lib.o lib.a
