CC = nvcc
CFLAGS = -O2 -g
INC_DIRS = -I$(CURDIR)/headers/
LIBS = -lstdc++ -lm

$(shell mkdir -p build)

#OBJS = $(patsubst %.cu, build/%.o, $(wildcard *.cu))
OBJS = build/main.o build/generator.o
NON_GEN_HEADERS = $(wildcard headers/*.hpp) $(wildcard headers/*.h) $(wildcard headers/*.cuh)
 
test: $(OBJS) $(NON_GEN_HEADERS)
	$(CC) $(CFLAGS) $(INC_DIRS) -o test $(OBJS) $(LIBS)

all: test

debug: CFLAGS = -O2 -g
debug: $(OBJS) $(NON_GEN_HEADERS)
	$(CC) $(CFLAGS) $(INC_DIRS) -o debug $^ $(LIBS)

build/%.o: %.cu 
	$(CC) $(CFLAGS) $(INC_DIRS) -o $@ -c $< $(LIBS)

build/%.o: %.cpp
	$(CC) $(CFLAGS) $(INC_DIRS) -o $@ -c $< $(LIBS)
	
@PHONY: clean
clean:
	rm -rf build test debug