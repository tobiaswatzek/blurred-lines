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
    __constant const float *gaussianKernel,
	__local float3 *pixelsLocal)
{
    int radius = GAUSS_KERNEL_SIZE >> 1;
    // == x in image
    size_t globalIdX = get_global_id(0);
    // == y in image
    size_t globalIdY = get_global_id(1);

    size_t localIdX = get_local_id(0);
    size_t localIdY = get_local_id(1);
    size_t localSizeX = get_local_size(0);
    size_t localSizeY = get_local_size(1);


    if (globalIdX > IMAGE_WIDTH || globalIdY > IMAGE_HEIGHT) {
        return;
    }

    // map 2d to 1d
    size_t globalId = IMAGE_WIDTH * globalIdX + globalIdY;

    pixelsLocal[localId] = (float3) (redPixels[globalId], greenPixels[globalId], bluePixels[globalId]);

    if (localIdX == 0) {

    }


	barrier(CLK_LOCAL_MEM_FENCE);


    int rightPadding = IMAGE_WIDTH - radius;
    int bottomPadding = IMAGE_HEIGHT - radius;

    float3 blurredPixel = (float3)(0);

    for(int i = 0; i < GAUSS_KERNEL_SIZE; ++i) {
        float gaussianKernelValue = gaussianKernel[i];
        float3 pixel = (float3)(0);
        int globalPixelId = globalId - radius + i;
        int localPixelId = localId - radius + i;

        // Border pixels use their own value
        if ((x < radius || x >= rightPadding) && (globalPixelId < 0 || globalPixelId >= IMAGE_WIDTH)) {
            pixel = pixelsLocal[localId];
        } else if (localPixelId < 0 || localPixelId >= localSize) {
            // some pixels are outside of the loaded local pixels
            pixel.s0 = redPixels[globalPixelId];
            pixel.s1 = greenPixels[globalPixelId];
            pixel.s2 = bluePixels[globalPixelId];
        } else {
            pixel = pixelsLocal[localPixelId];
        }
        // use the power of built in vector types
        blurredPixel += pixel * gaussianKernelValue;
    }

    // the _sat modifier changes values that over or underflow the target type to the nearest
    // representable value so 259f would be set to 255
	redPixelsOut[globalId] = convert_uchar_sat(blurredPixel.s0);
	greenPixelsOut[globalId] = convert_uchar_sat(blurredPixel.s1);
	bluePixelsOut[globalId] = convert_uchar_sat(blurredPixel.s2);
}
