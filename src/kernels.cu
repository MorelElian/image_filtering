#define MEMORY_PER_THREAD 10
#define N_IMAGES_PER_ROUND 2
#include <cuda.h>
#include <cuda_runtime.h>
#include "gif_lib.h"
__global__ void apply_gray_filter_kernel(pixel * p, int width, int height)
{
    int j;
    j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= width * height) return;

    int moy;
    moy = (p[j].r + p[j].g + p[j].b) / 3;
    if (moy < 0) moy = 0;
    if (moy > 255) moy = 255;

    p[j].r = moy;
    p[j].g = moy;
    p[j].b = moy;
}
extern "C" void apply_gray_filter_cuda(animated_gif *image,int nb_threads)
{
    pixel **p;
    p = image->p;
    int x;
    int n_images = image->n_images;
    int width;
    int height;
    
    int nb_blocks;
    size_t size_pixels;
    int n_pixels;
    for (int i = 0; i < n_images; i++) {
        pixel* d_p;
        n_pixels = image->width[i] * image->height[i];
        size_pixels = n_pixels * sizeof(pixel);
        nb_threads = 256;
        nb_blocks = (size_pixels / nb_threads)+1;
        width = image->width[i];
        height = image->height[i];

        cudaMalloc(&d_p, size_pixels);
        cudaMemcpy(d_p, p[i], size_pixels, cudaMemcpyHostToDevice);
        
        apply_gray_filter_kernel<<<nb_blocks, nb_threads>>>(d_p, width, height);
        cudaDeviceSynchronize();
        
        cudaMemcpy(p[i], d_p, size_pixels, cudaMemcpyDeviceToHost);
        cudaFree(d_p);
        
    }
}
__global__ void blur_image_kernel(pixel* p_cuda, pixel* new_p,int width,int height,int size)
{
    //indice du point
    
    int i  = blockIdx.x * blockDim.x +threadIdx.x;
    int j = i / width; // peut valoir en 0 et height-1 equivalent de j dans la version originale
    int k = i % width; // peut valoir en 0 et width-1 equivalent de k
    if(i >= width * height) return;
    else
    {
        //There are 3 cases : top/middle/bottom of the image
        if(j >= size && j < height/10 -size && k>=size && k <width-size)
        {
            int stencil_j, stencil_k ;
            int t_r = 0 ;
            int t_g = 0 ;
            int t_b = 0 ;
            for ( stencil_j = -size ; stencil_j <= size ; stencil_j++ )
                    {
                        for ( stencil_k = -size ; stencil_k <= size ; stencil_k++ )
                        {
                            t_r += p_cuda[(j+stencil_j) * width + k].r ;
                            t_g += p_cuda[(j+stencil_j)*width + k+stencil_k].g ;
                            t_b += p_cuda[(j+stencil_j)*width + k+stencil_k].b ;
                        }
                    }
            new_p[j*width +k].r = t_r / ( (2*size+1)*(2*size+1) ) ;
            new_p[j*width +k].g = t_g / ( (2*size+1)*(2*size+1) ) ;
            new_p[j*width +k].b = t_b / ( (2*size+1)*(2*size+1) ) ;
        }
        else if(j>=height/10 -size && j < height * 0.9+size && k >= size && k < width - size)
        {
            new_p[width*j + k].r = p_cuda[width*j + k].r ; 
            new_p[width*j + k].g = p_cuda[width*j + k].g ; 
            new_p[width*j + k].b = p_cuda[width*j + k].b ;
        }
        else if(j>=0.9*height + size && j < height-size && k>=size && k < width-size)
        {
            int stencil_j, stencil_k ;
            int t_r = 0 ;
            int t_g = 0 ;
            int t_b = 0 ;
            for ( stencil_j = -size ; stencil_j <= size ; stencil_j++ )
                    {
                        for ( stencil_k = -size ; stencil_k <= size ; stencil_k++ )
                        {
                            t_r += p_cuda[(j+stencil_j)*width + k+stencil_k].r ;
                            t_g += p_cuda[(j+stencil_j)*width + k+stencil_k].g ;
                            t_b += p_cuda[(j+stencil_j)*width + k+stencil_k].b ;
                        }
                    }
            new_p[j*width +k].r = t_r / ( (2*size+1)*(2*size+1) ) ;
            new_p[j*width +k].g = t_g / ( (2*size+1)*(2*size+1) ) ;
            new_p[j*width +k].b = t_b / ( (2*size+1)*(2*size+1) ) ;
        }
        else
        {
            new_p[width*j + k].r = p_cuda[width*j + k].r ; 
            new_p[width*j + k].g = p_cuda[width*j + k].g ; 
            new_p[width*j + k].b = p_cuda[width*j + k].b ;
        }
    }
}
__global__ void test_blur_image_kernel(pixel * p_cuda, pixel* new_p,int* end, int width, int height, int threshold)
{
    int i  = blockIdx.x * blockDim.x +threadIdx.x;
    int j = i / width; // peut valoir en 0 et height-1 equivalent de j dans la version originale
    int k = i %width; // peut valoir en 0 et width-1 equivalent de k
    
    if(i >= width * height) return;
    float diff_r ;
    float diff_g ;
    float diff_b ;
    if(j >= 0 && k >=0)
    {
        diff_r = (new_p[j*width +k].r - p_cuda[j*width +k].r) ;
        diff_g = (new_p[j*width +k].g - p_cuda[j*width +k].g) ;
        diff_b = (new_p[j*width +k].b - p_cuda[j*width +k].b);
       
        if ( diff_r > threshold || -diff_r > threshold 
                ||
                    diff_g > threshold || -diff_g > threshold
                    ||
                    diff_b > threshold || -diff_b > threshold
            ) {
            
            *(end) = 0 ;
        }
    }
    p_cuda[width*j + k].r = new_p[width*j + k].r ;
    p_cuda[width*j + k].g = new_p[width*j + k].g ;
    p_cuda[width*j + k].b = new_p[width*j + k].b ;
}
extern "C" void
apply_blur_filter_cuda( animated_gif * image, int size, int threshold ,int nb_threads)
{
   
    int width, height,nb_blocks ;
    int x;
    x = 1;
    int end;
    end = 0;
    
    size_t size_pixels;
    pixel ** p ;
    
    int i;
    
    
    /* Get the pixels of all images */
    p = image->p ;


    /* Process all images */
    for ( i = 0 ; i < image->n_images ; i++ )
    {
        
        width = image->width[i] ;
        height = image->height[i] ;
        
        
        size_pixels = width * height * sizeof(pixel);
        nb_blocks = width*height / nb_threads +1;
        
        /* Allocate array of newa pixels */
        pixel * new_p ;
        pixel* p_cuda;
        cudaMalloc(&new_p,size_pixels);
        cudaMalloc(&p_cuda,size_pixels);
        cudaMemcpy(p_cuda, p[i], size_pixels, cudaMemcpyHostToDevice);
        do
        {
            cudaDeviceSynchronize();
            blur_image_kernel<<<nb_blocks,nb_threads>>>(p_cuda,new_p,width,height,size);
            cudaDeviceSynchronize();
            end = 1;
          
            int *end_cuda;
            cudaMalloc(&end_cuda,sizeof(int));
            cudaMemcpy(end_cuda,&end,sizeof(int),cudaMemcpyHostToDevice);
            test_blur_image_kernel<<<nb_blocks,nb_threads>>>(p_cuda,new_p,end_cuda,width,height,threshold);
            cudaDeviceSynchronize();
            cudaMemcpy(&end,end_cuda,sizeof(int),cudaMemcpyDeviceToHost);
            cudaFree(end_cuda);
            
        } while (threshold >0 && !end);
        cudaMemcpy(p[i],p_cuda,size_pixels,cudaMemcpyDeviceToHost);
        

        cudaFree(p_cuda);
        cudaFree(new_p);
    }
    
}
__global__ void apply_sobel_filter_kernel(pixel* p_cuda,pixel* sobel, int width,int height)
{
    int i,j,k;
    i = blockIdx.x* blockDim.x + threadIdx.x;
    j = i /width;
    k = i%width;
    
    //int pixel_blue_no, pixel_blue_n, pixel_blue_ne;
   

    float deltaX_blue ;
    float deltaY_blue ;
    float val_blue;

    //pixel_blue_no = p_cuda[(j-1)*width + k-1].b ;
    //pixel_blue_n  = p_cuda[width * (j-1) +k  ].b ;
    //pixel_blue_ne = p_cuda[width * (j-1) +k+1].b ;
    
    
    deltaX_blue = -p_cuda[(j-1)*width + k-1].b  + p_cuda[width * (j-1) +k+1].b - 2*p_cuda[width * (j  ) +k-1].b  + 2*p_cuda[width * (j  ) +k+1].b - p_cuda[width * (j+1) +k-1].b + p_cuda[width * (j+1) +k+1].b ;          

    deltaY_blue = p_cuda[width * (j+1) +k+1].b  + 2*p_cuda[width * (j+1) +k  ].b  + p_cuda[width * (j+1) +k-1].b - p_cuda[width * (j-1) +k+1].b - 2*p_cuda[width * (j-1) +k  ].b -p_cuda[(j-1)*width + k-1].b ;

    val_blue = sqrt(deltaX_blue * deltaX_blue + deltaY_blue * deltaY_blue)/4; 
   
    //printf("%f \n",val_blue);
    if ( val_blue > 50 ) 
    {
        //printf("aaa\n");
        sobel[width*j + k].r = 254 ;
        sobel[width*j + k].g = 254 ;
        sobel[width*j + k].b = 254 ;
    } else
    {

        sobel[width*j + k].r = 0 ;
        sobel[width*j + k].g = 0 ;
        sobel[width*j + k].b = 0 ;
    }
    
}
extern "C" void apply_sobel_filter_cuda(animated_gif* image,int nb_threads)
{
    int i,width,height;

    int nb_blocks;

    pixel**p;
     p = image->p ;
    
    for(i = 0 ;i < image->n_images; i++)
    {
        
        width = image->width[i] ;
        height = image->height[i] ;
        
        nb_blocks = width*height /nb_threads +1;
        pixel* sobel;
        pixel* p_cuda;
        cudaMalloc(&sobel,width*height*sizeof(pixel));
        cudaMalloc(&p_cuda,width*height*sizeof(pixel));
        
        cudaMemcpy(p_cuda,p[i],width*height*sizeof(pixel),cudaMemcpyHostToDevice);
        apply_sobel_filter_kernel<<<nb_blocks,nb_threads>>>(p_cuda,sobel,width,height);
        cudaDeviceSynchronize();
        
    
        cudaFree(p_cuda);
        cudaMemcpy(p[i],sobel,width*height*sizeof(pixel),cudaMemcpyDeviceToHost);
        cudaFree(sobel);
        
    }
}
bool test_gpu_available(int width,int height)
// This function tests two things : first is there a gpu available on the node second is there enough memory one the gpu to do the computation
{
     int deviceCount;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
    if (error_id != cudaSuccess || deviceCount == 0) {
        return false;
    }
    else
    {
        int device ; // ID du périphérique CUDA à utiliser (dans ce cas, le premier périphérique)
        device = 0;
        size_t free_byte, total_byte;
        cudaSetDevice(device);
        cudaMemGetInfo(&free_byte, &total_byte);
        double free_db = (double)free_byte;
        free_db /= 1024.0;
        free_db /= 1024.0;
        double needed_memory;
        needed_memory = width/1024.0    * height /1024.0 *sizeof(pixel) *MEMORY_PER_THREAD * N_IMAGES_PER_ROUND  *sizeof(int);
        if(needed_memory > free_db * 0.8)
        {
            return false;
        }
        return true;
     
    }
}