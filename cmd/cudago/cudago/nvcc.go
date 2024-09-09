package main

import "syscall"

func mkFifo(name string) {
	err := syscall.Mkfifo(name, 0666)
	if err != nil {
		panic(err)
	}
}
