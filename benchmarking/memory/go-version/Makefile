export CGO_ENABLED=1
export CGO_CFLAGS=$(shell pkg-config --cflags cudart)
export CGO_LDFLAGS=$(shell pkg-config --libs cudart)

#CUDAGO_TARGETS:="cu/kernels.go cu/utilities.go"

MODULE:=$(shell go list)
PKGNAME:=cu

test: main.go
	go mod tidy
	go build -o test .

.PHONY: clean
clean:
	rm -rf test cu/

all: test

.PHONY: printVars
printVars:
	@echo CGO_ENABLED=$(CGO_ENABLED)
	@echo CGO_CFLAGS=$(CGO_CFLAGS)
	@echo CGO_LDFLAGS=$(CGO_LDFLAGS)

benchmark: test
	./test 2>benchmark.log >measurements.csv

