obj:
	cd safpa; gcc -Wall -Ofast -c -fPIC -o permanent_fault_lib.o permanent_fault_lib.c -lm

lib:
	cd safpa; gcc -shared -fPIC permanent_fault_lib.o -o permanent_fault_lib.so -lm

all: obj lib
	cd safpa; rm -f permanent_fault_lib.o