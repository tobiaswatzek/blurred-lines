/*
* A kernel that applies a gaussian blur to pixels.
*/
__kernel void gaussianBlur(
    __global const unsigned char *red,
	__global const unsigned char *green,
	__global const unsigned char *blue,
    __global unsigned char *redOut,
	__global unsigned char *greenOut,
	__global unsigned char *blueOut,
	__local unsigned char *redLocal,
    __local unsigned char *greenLocal,
    __local unsigned char *blueLocal,
	__global const float *gaussianKernel,
	int kernelSize,
	int width)
{
    size_t globalIdRed = get_global_id(0);
    size_t globalIdGreen = get_global_id(1);
    size_t globalIdBlue = get_global_id(2);

    size_t localIdRed = get_local_id(0);
    size_t localIdGreen = get_local_id(1);
    size_t localIdBlue = get_local_id(2);
	
	redLocal[localIdRed] = red[globalIdRed];
	greenLocal[localIdGreen] = green[globalIdGreen];
	blueLocal[localIdBlue] = blue[globalIdBlue];
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	
	redOut[globalIdRed] = redLocal[localIdRed] + 5;
	greenOut[globalIdGreen] = greenLocal[localIdGreen] + 5;
	blueOut[globalIdBlue] = blueLocal[localIdBlue] + 5;
	    
}
