/* #ifndef KERNELS_H
#define KERNELS_H

#include "structs.h"
__global __void apply_gray_filter_kernel(pixel * p, int width, int height);
void apply_gray_filter(animated_gif *image);
__global __ void blur_image_kernel(pixel* p_cuda, pixel* new_p,int width,int height,int size);
 void test_blur_image_kernel(pixel * p_cuda, pixel* new_p,int* end, int width, int height, int threshold);
 void apply_sobel_filter_kernel(pixel* p_cuda,pixel* sobel, int width,int height);
void apply_blur_filter( animated_gif * image, int size, int threshold );
void apply_sobel_filter(animated_gif* image)
#endif // KERNELS_H */