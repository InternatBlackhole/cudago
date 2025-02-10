export CGO_ENABLED=1
export CGO_CFLAGS=$(shell pkg-config --cflags cudart)
export CGO_LDFLAGS=$(shell pkg-config --libs cudart)

test: main.go ints/ints.go
	go mod tidy
	go build -o test .

ints/ints.go ints/utilities.go &: ints.cu
	~/go/bin/CudaGo -package ints -precompile -- ints.cu

.PHONY: clean
clean:
	rm -f test ints
	rmdir cu 2>/dev/null || true

all: test

.PHONY: printVars
printVars:
	@echo CGO_ENABLED=$(CGO_ENABLED)
	@echo CGO_CFLAGS=$(CGO_CFLAGS)
	@echo CGO_LDFLAGS=$(CGO_LDFLAGS)
	@echo PICS=$(PICS)
	@echo OUTDIR=$(OUTDIR)

