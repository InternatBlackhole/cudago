export CGO_ENABLED=1
export CGO_CFLAGS=$(shell pkg-config --cflags cudart)
export CGO_LDFLAGS=$(shell pkg-config --libs cudart)

CUDAGO_TARGETS:="cu/kernels.go cu/utilities.go"

MODULE:=$(shell go list)
PKGNAME:=cu

KERNEL_NUM=100

test: main.go ${CUDAGO_TARGETS} ignore__callbacks.go ignore__calls.go
	go mod tidy
	go build -o test .

${CUDAGO_TARGETS} &: ignore__kernels.cu
	~/go/bin/CudaGo -package $(PKGNAME) -precompile -- $<

ignore__kernels.cu: ./code-gen/kernel_generator.sh
	./code-gen/kernel_generator.sh ${KERNEL_NUM} > $@

ignore__callbacks.go: ./code-gen/call_generator.sh
	./code-gen/call_generator.sh ${KERNEL_NUM} $(MODULE)/$(PKGNAME) > $@

ignore__calls.go: ./code-gen/init.sh
	./code-gen/init.sh ${KERNEL_NUM} > $@

.PHONY: clean
clean:
	rm -rf test cu/ ignore__*

all: test

.PHONY: printVars
printVars:
	@echo CGO_ENABLED=$(CGO_ENABLED)
	@echo CGO_CFLAGS=$(CGO_CFLAGS)
	@echo CGO_LDFLAGS=$(CGO_LDFLAGS)

benchmark: test
	./test 2>benchmark.log >measurements.csv

