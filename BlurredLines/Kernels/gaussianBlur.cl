/*
* A kernel that applies a gaussian blur to pixels.
*/
__kernel void gaussianBlur(
    __global const uchar3 *pixels,
    __global uchar3 *pixelsOut,
	__local uchar3 *pixelsLocal,
	__global const float *gaussianKernel,
	int kernelSize,
	int width)
{
    size_t globalId = get_global_id(0);
    size_t localId = get_local_id(0);
	
	pixelsLocal[localId] = pixels[globalId];
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
//	for(int i = 0; i < kernelSize; ++i) {
//	    float gaussianKernelValue = gaussianKernel[i];
//	    
//	}
	
	uchar3 inversePixel = pixelsLocal[localId];
	inversePixel.s0 = 255 - inversePixel.s0;
	inversePixel.s1 = 255 - inversePixel.s1;
	inversePixel.s2 = 255 - inversePixel.s2;
	
	pixelsOut[globalId] = inversePixel;
}
