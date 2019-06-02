/*
* A kernel that applies a gaussian blur to pixels.
*/
__kernel void gaussianBlur(
    __global const unsigned char *redPixels,
    __global const unsigned char *greenPixels,
    __global const unsigned char *bluePixels,
    __global unsigned char *redPixelsOut,
    __global unsigned char *greenPixelsOut,
    __global unsigned char *bluePixelsOut,
	__local unsigned char *redPixelsLocal,
	__local unsigned char *greenPixelsLocal,
	__local unsigned char *bluePixelsLocal,
	__constant  const float *gaussianKernel,
	int kernelSize,
	int width)
{
    size_t globalId = get_global_id(0);
    size_t localId = get_local_id(0);
    size_t localSize = get_local_size(0);

    redPixelsLocal[localId] = redPixels[globalId];
    greenPixelsLocal[localId] = greenPixels[globalId];
    bluePixelsLocal[localId] = bluePixels[globalId];

	barrier(CLK_LOCAL_MEM_FENCE);

    float3 blurredPixel = (float3)(0);
    int radius = kernelSize >> 1;
    int rightPadding = width - radius;
    size_t x = globalId % width;
    size_t y = globalId / width;
        
    if (x >= radius && x < rightPadding) {
        for(int i = 0; i < kernelSize; ++i) {
            float gaussianKernelValue = gaussianKernel[i];
            int pixelId = localId - radius + i;

            uchar3 pixel = (uchar3)(0);

            if (pixelId < 0 || pixelId >= localSize) {
                pixel.s0 = redPixels[globalId - radius + i];
                pixel.s1 = greenPixels[globalId - radius + i];
                pixel.s2 = bluePixels[globalId - radius + i];
            } else {
                pixel.s0 = redPixelsLocal[pixelId];
                pixel.s1 = greenPixelsLocal[pixelId];
                pixel.s2 = bluePixelsLocal[pixelId];
            }

            blurredPixel.s0 += pixel.s0 * gaussianKernelValue;
            blurredPixel.s1 += pixel.s1 * gaussianKernelValue;
            blurredPixel.s2 += pixel.s2 * gaussianKernelValue;
        }

        blurredPixel.s0 = blurredPixel.s0 < 255 ? blurredPixel.s0 : 255;
        blurredPixel.s1 = blurredPixel.s1 < 255 ? blurredPixel.s1 : 255;
        blurredPixel.s2 = blurredPixel.s2 < 255 ? blurredPixel.s2 : 255;
    }  else {
        blurredPixel.s0 = redPixels[globalId];
        blurredPixel.s1 = greenPixels[globalId];
        blurredPixel.s2 = bluePixels[globalId];
    }
    // ignore border pixels for now

//    if(globalId < 10) {
//        printf("Id: %d; Original: <%d,%d,%d>; Blurred <%f,%f,%f>;\n", 
//            globalId,
//            pixelsLocal[localId].s0,
//            pixelsLocal[localId].s1,
//            pixelsLocal[localId].s2,
//            blurredPixel.s0,
//            blurredPixel.s1,
//            blurredPixel.s2);
//    }

	redPixelsOut[globalId] =  (unsigned char) blurredPixel.s0;
	greenPixelsOut[globalId] =  (unsigned char) blurredPixel.s1;
	bluePixelsOut[globalId] =  (unsigned char) blurredPixel.s2;
}
