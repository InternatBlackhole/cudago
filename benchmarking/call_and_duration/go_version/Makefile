export CGO_ENABLED=1
export CGO_CFLAGS=$(shell pkg-config --cflags cudart)
export CGO_LDFLAGS=$(shell pkg-config --libs cudart)

edge_recognition: main.go edge/edges.go sort/bitonic-sort.go ints/ints.go increase.go borders.go sort.go
	go mod tidy
	go build -o edge_recognition .

edge/edges.go edge/utilities.go &: edges.cu
	~/go/bin/CudaGo -package edge -precompile -- $^

sort/bitonic-sort.go sort/utilities.go &: bitonic-sort.cu
	~/go/bin/CudaGo -package sort -precompile -- $^

ints/ints.go ints/utilities.go &: ints.cu
	~/go/bin/CudaGo -package ints -precompile -- $^


.PHONY: clean
clean:
	rm -f edge_recognition edge sort ints
	rmdir cu 2>/dev/null || true

all: edge_recognition

.PHONY: printVars
printVars:
	@echo CGO_ENABLED=$(CGO_ENABLED)
	@echo CGO_CFLAGS=$(CGO_CFLAGS)
	@echo CGO_LDFLAGS=$(CGO_LDFLAGS)
	@echo PICS=$(PICS)
	@echo OUTDIR=$(OUTDIR)

