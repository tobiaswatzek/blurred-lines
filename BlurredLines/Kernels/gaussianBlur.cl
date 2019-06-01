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
    size_t localSize = get_local_size(0);

    pixelsLocal[localId] = pixels[globalId];

	barrier(CLK_LOCAL_MEM_FENCE);

    float3 blurredPixel = (float3)(0, 0, 0);
    int radius = kernelSize >> 1;
    int rightPadding = width - radius;
    size_t rowId = globalId % width;
       
    if (rowId >= radius && rowId < rightPadding) {
        for(int i = 0; i < kernelSize; ++i) {
            float gaussianKernelValue = gaussianKernel[i];
            int pixelId = localId - radius + i;

            uchar3 pixel;

            if (pixelId < 0 || pixelId > localSize) {
                pixel = pixels[globalId - radius + i];
            } else {
                pixel = pixelsLocal[pixelId];
            }
            
            blurredPixel.s0 += pixel.s0 * gaussianKernelValue;
            blurredPixel.s1 += pixel.s1 * gaussianKernelValue;
            blurredPixel.s2 += pixel.s2 * gaussianKernelValue;
                
            if (globalId == 100) {
                printf("i: %d, radius: %d; idx: %d; gauss: %f; blur: <%f; %f; %f;>; orig: <%d; %d; %d>;", i, 
                    radius, 
                    globalId - radius + i,
                    gaussianKernelValue, 
                    blurredPixel.s0, 
                    blurredPixel.s1, 
                    blurredPixel.s2,
                    pixel.s0, 
                    pixel.s1, 
                    pixel.s2);
            }
            
            
            
        }
        
        blurredPixel.s0 = blurredPixel.s0 < 255 ? blurredPixel.s0 : 255;
        blurredPixel.s1 = blurredPixel.s1 < 255 ? blurredPixel.s1 : 255;
        blurredPixel.s2 = blurredPixel.s2 < 255 ? blurredPixel.s2 : 255;
    } 
    // ignore border pixels for now
	barrier(CLK_LOCAL_MEM_FENCE);

	pixelsOut[globalId] = (uchar3) (blurredPixel.s0, blurredPixel.s1, blurredPixel.s2);
}
