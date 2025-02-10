CC = nvcc
CFLAGS = -O2 -g
INC_DIRS = -Istb/
LIBS = -lstdc++ -lm

$(shell mkdir -p build)

OBJS:= $(patsubst %.cu,build/%.o,$(wildcard *.cu))

edge_recognition: $(OBJS)
	$(CC) $(CFLAGS) $(INC_DIRS) -o edge_recognition $^ $(LIBS)

build/%.o: %.cu
	$(CC) $(CFLAGS) $(INC_DIRS) -c $< -o $@

all: edge_recognition

printVars:
	@echo OBJS: $(OBJS)

@PHONY: printVars
