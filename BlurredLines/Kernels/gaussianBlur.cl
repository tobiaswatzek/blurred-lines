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
	__local float3 *pixelsLocal,
	__constant  const float *gaussianKernel)
{
    size_t globalId = get_global_id(0);
    size_t localId = get_local_id(0);
    size_t localSize = get_local_size(0);

    pixelsLocal[localId] = (float3) (redPixels[globalId], greenPixels[globalId], bluePixels[globalId]);

	barrier(CLK_LOCAL_MEM_FENCE);

    float3 blurredPixel = (float3)(0);
    int radius = GAUSS_KERNEL_SIZE >> 1;
    int rightPadding = IMAGE_WIDTH - radius;
    size_t x = globalId % IMAGE_WIDTH;
    size_t y = globalId / IMAGE_WIDTH;
        
    for(int i = 0; i < GAUSS_KERNEL_SIZE; ++i) {
        float gaussianKernelValue = gaussianKernel[i];
        float3 pixel = (float3)(0);
        int globalPixelId = globalId - radius + i;
        int localPixelId = localId - radius + i;

        // Border pixels use their own value
        if ((x < radius || x >= rightPadding) && (globalPixelId < 0 || globalPixelId >= IMAGE_WIDTH)) {
            pixel = pixelsLocal[localId];
        } else if (localPixelId < 0 || localPixelId >= localSize) {
            pixel.s0 = redPixels[globalPixelId];
            pixel.s1 = greenPixels[globalPixelId];
            pixel.s2 = bluePixels[globalPixelId];
        } else {
            pixel = pixelsLocal[localPixelId];
        }

        blurredPixel += pixel * gaussianKernelValue;
    }

    blurredPixel.s0 = blurredPixel.s0 < 255 ? blurredPixel.s0 : 255;
    blurredPixel.s1 = blurredPixel.s1 < 255 ? blurredPixel.s1 : 255;
    blurredPixel.s2 = blurredPixel.s2 < 255 ? blurredPixel.s2 : 255;
   
    // ignore border pixels for now

	redPixelsOut[globalId] =  (unsigned char) blurredPixel.s0;
	greenPixelsOut[globalId] =  (unsigned char) blurredPixel.s1;
	bluePixelsOut[globalId] =  (unsigned char) blurredPixel.s2;
}
