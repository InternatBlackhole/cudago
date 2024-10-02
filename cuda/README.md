# CUDA Driver API Go wrapper

This package provides Go bindings for the CUDA Driver API.

## How to use

Every program that uses this package must first run the following inititialization code:

```go
dev, err := cuda.Init(0) // 0 is the device number
if err != nil {
    panic(err) // or handle the error
}
defer dev.Close() // close the library when done. REQUIRED, see below
```

The `Init` function initializes the CUDA driver API and returns a `Device` object that represents the device. The `Close` method must be called when the program is done using the library. This is required because the `Init` function calls `runtime.LockOSThread` to ensure that the CUDA driver API is called from the same OS thread. This means that only that goroutine can call the CUDA driver API functions (and use the OS thread).
`Close` will unlock the OS thread.
