/*
 * A kernel that applies a gaussian blur to pixels.
 */
__kernel void gaussianBlur(__global const unsigned char *redPixels,
                           __global const unsigned char *greenPixels,
                           __global const unsigned char *bluePixels,
                           __global unsigned char *redPixelsOut,
                           __global unsigned char *greenPixelsOut,
                           __global unsigned char *bluePixelsOut,
                           __constant const float *gaussianKernel,
                           __local float3 *pixelsLocal) {
  // == x in image
  size_t globalIdX = get_global_id(0);
  // == y in image
  size_t globalIdY = get_global_id(1);

  size_t globalSizeX = get_global_size(0);
  size_t globalSizeY = get_global_size(1);

  size_t localIdX = get_local_id(0) + GAUSS_RADIUS_SIZE;
  size_t localIdY = get_local_id(1) + GAUSS_RADIUS_SIZE;
  size_t localSizeX = get_local_size(0) + GAUSS_KERNEL_SIZE - 1;
  size_t localSizeY = get_local_size(1) + GAUSS_KERNEL_SIZE - 1;

  if (globalIdX >= IMAGE_WIDTH && globalIdY >= IMAGE_HEIGHT) {
    return;
  }

  // map 2d to 1d
  size_t globalId = IMAGE_WIDTH * globalIdY + globalIdX;
  size_t localId = localSizeX * localIdY + localIdX;

  pixelsLocal[localId] = (float3)(redPixels[globalId], greenPixels[globalId],
                                  bluePixels[globalId]);

  if ((localIdX - GAUSS_RADIUS_SIZE) == 0) {
    // pixel is on the left side
    for (int i = 1; i <= GAUSS_RADIUS_SIZE; ++i) {
      size_t offsetGlobalId;
      if (globalIdX >= i) {
        offsetGlobalId = IMAGE_WIDTH * globalIdY + (globalIdX - i);
      } else {
        // if we would go outside of the image use our own pixel value
        offsetGlobalId = globalId;
      }
      size_t offsetLocalId = localSizeX * localIdY + (localIdX - i);
      pixelsLocal[offsetLocalId] =
          (float3)(redPixels[offsetGlobalId], greenPixels[offsetGlobalId],
                   bluePixels[offsetGlobalId]);
    }
  } else if ((localIdX - GAUSS_RADIUS_SIZE) ==
                 (localSizeX - GAUSS_KERNEL_SIZE + 1) ||
             globalIdX == IMAGE_WIDTH) {
    // pixel is on the right side
    for (int i = 1; i <= GAUSS_RADIUS_SIZE; ++i) {
      size_t offsetGlobalId;
      if ((globalIdX + i) <= IMAGE_WIDTH) {
        offsetGlobalId = IMAGE_WIDTH * globalIdY + (globalIdX + i);
      } else {
        // if we would go outside of the image use our own pixel value
        offsetGlobalId = globalId;
      }
      size_t offsetLocalId = localSizeX * localIdY + (localIdX + i);
      pixelsLocal[offsetLocalId] =
          (float3)(redPixels[offsetGlobalId], greenPixels[offsetGlobalId],
                   bluePixels[offsetGlobalId]);
    }
  }

  if ((localIdY - GAUSS_RADIUS_SIZE) == 0) {
    // pixel is on the top
    for (int i = 1; i <= GAUSS_RADIUS_SIZE; ++i) {
      size_t offsetGlobalId;
      if (globalIdY >= i) {
        offsetGlobalId = IMAGE_WIDTH * (globalIdY - i) + globalIdX;
      } else {
        // if we would go outside of the image use our own pixel value
        offsetGlobalId = globalId;
      }
      size_t offsetLocalId = localSizeX * (localIdY - i) + localIdX;
      pixelsLocal[offsetLocalId] =
          (float3)(redPixels[offsetGlobalId], greenPixels[offsetGlobalId],
                   bluePixels[offsetGlobalId]);
    }
  } else if ((localIdY - GAUSS_RADIUS_SIZE) ==
                 (localSizeY - GAUSS_KERNEL_SIZE + 1) ||
             globalIdY == IMAGE_HEIGHT) {
    // pixel is on the bottom
    for (int i = 1; i <= GAUSS_RADIUS_SIZE; ++i) {
      size_t offsetGlobalId;
      if ((globalIdY + i) <= IMAGE_HEIGHT) {
        offsetGlobalId = IMAGE_WIDTH * (globalIdY + i) + globalIdX;
      } else {
        // if we would go outside of the image use our own pixel value
        offsetGlobalId = globalId;
      }
      size_t offsetLocalId = localSizeX * (localIdY + i) + localIdX;
      pixelsLocal[offsetLocalId] =
          (float3)(redPixels[offsetGlobalId], greenPixels[offsetGlobalId],
                   bluePixels[offsetGlobalId]);
    }
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  float3 blurredPixel = (float3)(0);

  for (int x = -GAUSS_RADIUS_SIZE; x <= GAUSS_RADIUS_SIZE; ++x) {
    size_t offsetLocalId = localSizeX * localIdY + (localIdX + x);
    if(localIdX == GAUSS_RADIUS_SIZE && globalId < 256) {
        printf("GID: %d; x %d; off %d;", globalId, x, offsetLocalId);
    }
    float gaussianKernelValue = gaussianKernel[x + GAUSS_RADIUS_SIZE];
    float3 pixel = pixelsLocal[offsetLocalId];

    // use the power of built in vector types
    blurredPixel += pixel * gaussianKernelValue;
  }

  for (int y = -GAUSS_RADIUS_SIZE; y <= GAUSS_RADIUS_SIZE; ++y) {
      size_t offsetLocalId = localSizeX * (localIdY + y) + localIdX;

      float gaussianKernelValue = gaussianKernel[y + GAUSS_RADIUS_SIZE];
      float3 pixel = pixelsLocal[offsetLocalId];
      
      // use the power of built in vector types
      blurredPixel += pixel * gaussianKernelValue;
  }
  
  blurredPixel /= 2;

  // the _sat modifier changes values that over or underflow the target type to
  // the nearest representable value so 259f would be set to 255
  redPixelsOut[globalId] = convert_uchar_sat(blurredPixel.s0);
  greenPixelsOut[globalId] = convert_uchar_sat(blurredPixel.s1);
  bluePixelsOut[globalId] = convert_uchar_sat(blurredPixel.s2);
}

