package cuda

//#include <cuda.h>
import "C"

func Init() {
	C.cuInit(0)
}
