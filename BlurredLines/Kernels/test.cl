/*
* a kernel that add the elements of two vectors pairwise
*/
__kernel void vector_add(
	__global const int *A,
	__global const int *B,
	__global int *C,
	int kernelSize)
{
    int radius = kernelSize >> 1;
	size_t i = get_global_id(0);
	printf("%d-%d-%d\n", i, kernelSize, radius);
	C[i] = A[i] + B[i];
}
