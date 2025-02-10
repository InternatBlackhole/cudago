CC = nvcc
CFLAGS = -O2 -g
INC_DIRS = -I$(CURDIR)/headers/ -I$(CURDIR)/headers/gen/
LIBS = -lstdc++ -lm

KERNEL_NUM = 100
GEN_HEADERS_DIR = headers/gen

$(shell mkdir -p build)
$(shell mkdir -p headers/ $(GEN_HEADERS_DIR))


#OBJS = $(patsubst %.cu, build/%.o, $(wildcard *.cu))
OBJS = build/vector.o build/kernels.o build/callbacks.o build/funcs.o build/generator.o
NON_GEN_HEADERS = $(wildcard headers/*.hpp) $(wildcard headers/*.h) $(wildcard headers/*.cuh)
 
test: $(OBJS) $(NON_GEN_HEADERS)
	$(CC) $(CFLAGS) $(INC_DIRS) -o test $(OBJS) $(LIBS)

all: test

debug: CFLAGS = -O2 -g
debug: $(OBJS) $(NON_GEN_HEADERS)
	$(CC) $(CFLAGS) $(INC_DIRS) -o debug $^ $(LIBS)

build/%.o: %.cu 
	$(CC) $(CFLAGS) $(INC_DIRS) -o $@ -c $< $(LIBS)

build/generator.o: generator.cpp headers/generator.hpp
	$(CC) $(CFLAGS) $(INC_DIRS) -o $@ -c $< $(LIBS)


build/vector.o: build/vector.cu headers/common.hpp $(GEN_HEADERS_DIR)/callbacks.cuh
	$(CC) $(CFLAGS) $(INC_DIRS) -o $@ -c $< $(LIBS)

build/vector.cu: code-gen/vector_init.sh
	./code-gen/vector_init.sh $(KERNEL_NUM) $@


build/kernels.o: build/kernels.cu $(GEN_HEADERS_DIR)/kernels.cuh
	$(CC) $(CFLAGS) $(INC_DIRS) -o $@ -c $< $(LIBS)

$(GEN_HEADERS_DIR)/kernels.cuh build/kernels.cu &: code-gen/kernel_generator.sh
	./code-gen/kernel_generator.sh $(KERNEL_NUM) $(GEN_HEADERS_DIR)/kernels.cuh build/kernels.cu


build/callbacks.o: build/callbacks.cu $(GEN_HEADERS_DIR)/callbacks.cuh
	$(CC) $(CFLAGS) $(INC_DIRS) -o $@ -c $< $(LIBS)

build/callbacks.cu $(GEN_HEADERS_DIR)/callbacks.cuh &: code-gen/callback_generator.sh $(GEN_HEADERS_DIR)/kernels.cuh
	./code-gen/callback_generator.sh $(KERNEL_NUM) $(GEN_HEADERS_DIR)/callbacks.cuh build/callbacks.cu
	
@PHONY: clean
clean:
	rm -rf build test debug headers/gen/*