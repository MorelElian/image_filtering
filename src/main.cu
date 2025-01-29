/*
 * INF560
 *
 * Image Filtering Project
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <sys/time.h>
#include <mpi.h>
#include "gif_lib.h"
#include "structs.h"
#include "vanilla_function.c"
#include "kernels.cu"

//#include "kernels.h"


/* Set this macro to 1 to enable debugging information */
#define SOBELF_DEBUG 0
#define USE_GPU 1
#define THREADS 256
/*
 * Load a GIF image from a file and return a
 * structure of type animated_gif.
 */
animated_gif * load_pixels( char * filename ) 
{
    GifFileType * g ;
    ColorMapObject * colmap ;
    int error ;
    int n_images ;
    int * width ;
    int * height ;
    pixel ** p ;
    int i ;
    animated_gif * image ;

    /* Open the GIF image (read mode) */
    g = DGifOpenFileName( filename, &error ) ;
    if ( g == NULL ) 
    {
        fprintf( stderr, "Error DGifOpenFileName %s\n", filename ) ;
        return NULL ;
    }

    /* Read the GIF image */
    error = DGifSlurp( g ) ;
    if ( error != GIF_OK )
    {
        fprintf( stderr, 
                "Error DGifSlurp: %d <%s>\n", error, GifErrorString(g->Error) ) ;
        return NULL ;
    }

    /* Grab the number of images and the size of each image */
    n_images = g->ImageCount ;

    width = (int *)malloc( n_images * sizeof( int ) ) ;
    if ( width == NULL )
    {
        fprintf( stderr, "Unable to allocate width of size %d\n",
                n_images ) ;
        return 0 ;
    }

    height = (int *)malloc( n_images * sizeof( int ) ) ;
    if ( height == NULL )
    {
        fprintf( stderr, "Unable to allocate height of size %d\n",
                n_images ) ;
        return 0 ;
    }

    /* Fill the width and height */
    for ( i = 0 ; i < n_images ; i++ ) 
    {
        width[i] = g->SavedImages[i].ImageDesc.Width ;
        height[i] = g->SavedImages[i].ImageDesc.Height ;

#if SOBELF_DEBUG
        printf( "Image %d: l:%d t:%d w:%d h:%d interlace:%d localCM:%p\n",
                i, 
                g->SavedImages[i].ImageDesc.Left,
                g->SavedImages[i].ImageDesc.Top,
                g->SavedImages[i].ImageDesc.Width,
                g->SavedImages[i].ImageDesc.Height,
                g->SavedImages[i].ImageDesc.Interlace,
                g->SavedImages[i].ImageDesc.ColorMap
                ) ;
#endif
    }


    /* Get the global colormap */
    colmap = g->SColorMap ;
    if ( colmap == NULL ) 
    {
        fprintf( stderr, "Error global colormap is NULL\n" ) ;
        return NULL ;
    }

#if SOBELF_DEBUG
    printf( "Global color map: count:%d bpp:%d sort:%d\n",
            g->SColorMap->ColorCount,
            g->SColorMap->BitsPerPixel,
            g->SColorMap->SortFlag
            ) ;
#endif

    /* Allocate the array of pixels to be returned */
    p = (pixel **)malloc( n_images * sizeof( pixel * ) ) ;
    if ( p == NULL )
    {
        fprintf( stderr, "Unable to allocate array of %d images\n",
                n_images ) ;
        return NULL ;
    }

    for ( i = 0 ; i < n_images ; i++ ) 
    {
        p[i] = (pixel *)malloc( width[i] * height[i] * sizeof( pixel ) ) ;
        if ( p[i] == NULL )
        {
        fprintf( stderr, "Unable to allocate %d-th array of %d pixels\n",
                i, width[i] * height[i] ) ;
        return NULL ;
        }
    }
    
    /* Fill pixels */

    /* For each image */
    for ( i = 0 ; i < n_images ; i++ )
    {
        int j ;

        /* Get the local colormap if needed */
        if ( g->SavedImages[i].ImageDesc.ColorMap )
        {

            /* TODO No support for local color map */
            fprintf( stderr, "Error: application does not support local colormap\n" ) ;
            return NULL ;

            colmap = g->SavedImages[i].ImageDesc.ColorMap ;
        }

        /* Traverse the image and fill pixels */
        for ( j = 0 ; j < width[i] * height[i] ; j++ ) 
        {
            int c ;

            c = g->SavedImages[i].RasterBits[j] ;

            p[i][j].r = colmap->Colors[c].Red ;
            p[i][j].g = colmap->Colors[c].Green ;
            p[i][j].b = colmap->Colors[c].Blue ;
        }
    }

    /* Allocate image info */
    image = (animated_gif *)malloc( sizeof(animated_gif) ) ;
    if ( image == NULL ) 
    {
        fprintf( stderr, "Unable to allocate memory for animated_gif\n" ) ;
        return NULL ;
    }

    /* Fill image fields */
    image->n_images = n_images ;
    image->width = width ;
    image->height = height ;
    image->p = p ;
    image->g = g ;

#if SOBELF_DEBUG
    printf( "-> GIF w/ %d image(s) with first image of size %d x %d\n",
            image->n_images, image->width[0], image->height[0] ) ;
#endif

    return image ;
}

int 
output_modified_read_gif( char * filename, GifFileType * g ) 
{
    GifFileType * g2 ;
    int error2 ;

#if SOBELF_DEBUG
    printf( "Starting output to file %s\n", filename ) ;
#endif

    g2 = EGifOpenFileName( filename, false, &error2 ) ;
    if ( g2 == NULL )
    {
        fprintf( stderr, "Error EGifOpenFileName %s\n",
                filename ) ;
        return 0 ;
    }

    g2->SWidth = g->SWidth ;
    g2->SHeight = g->SHeight ;
    g2->SColorResolution = g->SColorResolution ;
    g2->SBackGroundColor = g->SBackGroundColor ;
    g2->AspectByte = g->AspectByte ;
    g2->SColorMap = g->SColorMap ;
    g2->ImageCount = g->ImageCount ;
    g2->SavedImages = g->SavedImages ;
    g2->ExtensionBlockCount = g->ExtensionBlockCount ;
    g2->ExtensionBlocks = g->ExtensionBlocks ;

    error2 = EGifSpew( g2 ) ;
    if ( error2 != GIF_OK ) 
    {
        fprintf( stderr, "Error after writing g2: %d <%s>\n", 
                error2, GifErrorString(g2->Error) ) ;
        return 0 ;
    }

    return 1 ;
}


int
store_pixels( char * filename, animated_gif * image )
{
    int n_colors = 0 ;
    pixel ** p ;
    int i, j, k ;
    GifColorType * colormap ;

    /* Initialize the new_p set of colors */
    colormap = (GifColorType *)malloc( 256 * sizeof( GifColorType ) ) ;
    if ( colormap == NULL ) 
    {
        fprintf( stderr,
                "Unable to allocate 256 colors\n" ) ;
        return 0 ;
    }

    /* Everything is white by default */
    for ( i = 0 ; i < 256 ; i++ ) 
    {
        colormap[i].Red = 255 ;
        colormap[i].Green = 255 ;
        colormap[i].Blue = 255 ;
    }

    /* Change the background color and store it */
    int moy ;
    moy = (
            image->g->SColorMap->Colors[ image->g->SBackGroundColor ].Red
            +
            image->g->SColorMap->Colors[ image->g->SBackGroundColor ].Green
            +
            image->g->SColorMap->Colors[ image->g->SBackGroundColor ].Blue
            )/3 ;
    if ( moy < 0 ) moy = 0 ;
    if ( moy > 255 ) moy = 255 ;

#if SOBELF_DEBUG
    printf( "[DEBUG] Background color (%d,%d,%d) -> (%d,%d,%d)\n",
            image->g->SColorMap->Colors[ image->g->SBackGroundColor ].Red,
            image->g->SColorMap->Colors[ image->g->SBackGroundColor ].Green,
            image->g->SColorMap->Colors[ image->g->SBackGroundColor ].Blue,
            moy, moy, moy ) ;
#endif

    colormap[0].Red = moy ;
    colormap[0].Green = moy ;
    colormap[0].Blue = moy ;

    image->g->SBackGroundColor = 0 ;

    n_colors++ ;

    /* Process extension blocks in main structure */
    for ( j = 0 ; j < image->g->ExtensionBlockCount ; j++ )
    {
        int f ;

        f = image->g->ExtensionBlocks[j].Function ;
        if ( f == GRAPHICS_EXT_FUNC_CODE )
        {
            int tr_color = image->g->ExtensionBlocks[j].Bytes[3] ;

            if ( tr_color >= 0 &&
                    tr_color < 255 )
            {

                int found = -1 ;

                moy = 
                    (
                     image->g->SColorMap->Colors[ tr_color ].Red
                     +
                     image->g->SColorMap->Colors[ tr_color ].Green
                     +
                     image->g->SColorMap->Colors[ tr_color ].Blue
                    ) / 3 ;
                if ( moy < 0 ) moy = 0 ;
                if ( moy > 255 ) moy = 255 ;

#if SOBELF_DEBUG
                printf( "[DEBUG] Transparency color image %d (%d,%d,%d) -> (%d,%d,%d)\n",
                        i,
                        image->g->SColorMap->Colors[ tr_color ].Red,
                        image->g->SColorMap->Colors[ tr_color ].Green,
                        image->g->SColorMap->Colors[ tr_color ].Blue,
                        moy, moy, moy ) ;
#endif

                for ( k = 0 ; k < n_colors ; k++ )
                {
                    if ( 
                            moy == colormap[k].Red
                            &&
                            moy == colormap[k].Green
                            &&
                            moy == colormap[k].Blue
                       )
                    {
                        found = k ;
                    }
                }
                if ( found == -1  ) 
                {
                    if ( n_colors >= 256 ) 
                    {
                        fprintf( stderr, 
                                "Error: Found too many colors inside the image\n"
                               ) ;
                        return 0 ;
                    }

#if SOBELF_DEBUG
                    printf( "[DEBUG]\tnewa color %d\n",
                            n_colors ) ;
#endif

                    colormap[n_colors].Red = moy ;
                    colormap[n_colors].Green = moy ;
                    colormap[n_colors].Blue = moy ;


                    image->g->ExtensionBlocks[j].Bytes[3] = n_colors ;

                    n_colors++ ;
                } else
                {
#if SOBELF_DEBUG
                    printf( "[DEBUG]\tFound existing color %d\n",
                            found ) ;
#endif
                    image->g->ExtensionBlocks[j].Bytes[3] = found ;
                }
            }
        }
    }

    for ( i = 0 ; i < image->n_images ; i++ ) // for every image
    {
        for ( j = 0 ; j < image->g->SavedImages[i].ExtensionBlockCount ; j++ )
        {
            int f ;

            f = image->g->SavedImages[i].ExtensionBlocks[j].Function ;
            if ( f == GRAPHICS_EXT_FUNC_CODE )
            {
                int tr_color = image->g->SavedImages[i].ExtensionBlocks[j].Bytes[3] ;

                if ( tr_color >= 0 &&
                        tr_color < 255 )
                {

                    int found = -1 ;

                    moy = 
                        (
                         image->g->SColorMap->Colors[ tr_color ].Red
                         +
                         image->g->SColorMap->Colors[ tr_color ].Green
                         +
                         image->g->SColorMap->Colors[ tr_color ].Blue
                        ) / 3 ;
                    if ( moy < 0 ) moy = 0 ;
                    if ( moy > 255 ) moy = 255 ;

#if SOBELF_DEBUG
                    printf( "[DEBUG] Transparency color image %d (%d,%d,%d) -> (%d,%d,%d)\n",
                            i,
                            image->g->SColorMap->Colors[ tr_color ].Red,
                            image->g->SColorMap->Colors[ tr_color ].Green,
                            image->g->SColorMap->Colors[ tr_color ].Blue,
                            moy, moy, moy ) ;
#endif

                    for ( k = 0 ; k < n_colors ; k++ )
                    {
                        if ( 
                                moy == colormap[k].Red
                                &&
                                moy == colormap[k].Green
                                &&
                                moy == colormap[k].Blue
                           )
                        {
                            found = k ;
                        }
                    }
                    if ( found == -1  ) 
                    {
                        if ( n_colors >= 256 ) 
                        {
                            fprintf( stderr, 
                                    "Error: Found too many colors inside the image\n"
                                   ) ;
                            return 0 ;
                        }

#if SOBELF_DEBUG
                        printf( "[DEBUG]\tnewa color %d\n",
                                n_colors ) ;
#endif

                        colormap[n_colors].Red = moy ;
                        colormap[n_colors].Green = moy ;
                        colormap[n_colors].Blue = moy ;


                        image->g->SavedImages[i].ExtensionBlocks[j].Bytes[3] = n_colors ;

                        n_colors++ ;
                    } else
                    {
#if SOBELF_DEBUG
                        printf( "[DEBUG]\tFound existing color %d\n",
                                found ) ;
#endif
                        image->g->SavedImages[i].ExtensionBlocks[j].Bytes[3] = found ;
                    }
                }
            }
        }
    }

#if SOBELF_DEBUG
    printf( "[DEBUG] Number of colors after background and transparency: %d\n",
            n_colors ) ;
#endif

    p = image->p ;

    /* Find the number of colors inside the image */
    for ( i = 0 ; i < image->n_images ; i++ )
    {

#if SOBELF_DEBUG
        printf( "OUTPUT: Processing image %d (total of %d images) -> %d x %d\n",
                i, image->n_images, image->width[i], image->height[i] ) ;
#endif

        for ( j = 0 ; j < image->width[i] * image->height[i] ; j++ ) 
        {
            int found = 0 ;
            for ( k = 0 ; k < n_colors ; k++ )
            {
                if ( p[i][j].r == colormap[k].Red &&
                        p[i][j].g == colormap[k].Green &&
                        p[i][j].b == colormap[k].Blue )
                {
                    found = 1 ;
                }
            }

            if ( found == 0 ) 
            {
                if ( n_colors >= 256 ) 
                {
                    fprintf( stderr, 
                            "Error: Found too many colors inside the image\n"
                           ) ;
                    return 0 ;
                }

#if SOBELF_DEBUG
                printf( "[DEBUG] Found new_p %d color (%d,%d,%d)\n",
                        n_colors, p[i][j].r, p[i][j].g, p[i][j].b ) ;
#endif

                colormap[n_colors].Red = p[i][j].r ;
                colormap[n_colors].Green = p[i][j].g ;
                colormap[n_colors].Blue = p[i][j].b ;
                n_colors++ ;
            }
        }
    }

#if SOBELF_DEBUG
    printf( "OUTPUT: found %d color(s)\n", n_colors ) ;
#endif


    /* Round up to a power of 2 */
    if ( n_colors != (1 << GifBitSize(n_colors) ) )
    {
        n_colors = (1 << GifBitSize(n_colors) ) ;
    }

#if SOBELF_DEBUG
    printf( "OUTPUT: Rounding up to %d color(s)\n", n_colors ) ;
#endif

    /* Change the color map inside the animated gif */
    ColorMapObject * cmo ;

    cmo = GifMakeMapObject( n_colors, colormap ) ;
    if ( cmo == NULL )
    {
        fprintf( stderr, "Error while creating a ColorMapObject w/ %d color(s)\n",
                n_colors ) ;
        return 0 ;
    }

    image->g->SColorMap = cmo ;

    /* Update the raster bits according to color map */
    for ( i = 0 ; i < image->n_images ; i++ )
    {
        for ( j = 0 ; j < image->width[i] * image->height[i] ; j++ ) 
        {
            int found_index = -1 ;
            for ( k = 0 ; k < n_colors ; k++ ) 
            {
                if ( p[i][j].r == image->g->SColorMap->Colors[k].Red &&
                        p[i][j].g == image->g->SColorMap->Colors[k].Green &&
                        p[i][j].b == image->g->SColorMap->Colors[k].Blue )
                {
                    found_index = k ;
                }
            }

            if ( found_index == -1 ) 
            {
                fprintf( stderr,
                        "Error: Unable to find a pixel in the color map\n" ) ;
                return 0 ;
            }

            image->g->SavedImages[i].RasterBits[j] = found_index ;
        }
    }


    /* Write the final image */
    if ( !output_modified_read_gif( filename, image->g ) ) { return 0 ; }

    return 1 ;
}


#define CONV(l,c,nb_c) \
    (l)*(nb_c)+(c)
void test (animated_gif * image)
{
    
}

/*
 * Main entry point
 */
int 
main( int argc, char ** argv )
{
   MPI_Init(NULL,NULL);
    
    char * input_filename ; 
    char * output_filename ;
    char * file_to_save;
    char * N;
    animated_gif * image ;
    struct timeval t1, t2;
    double loading_time,subgroup_time,sobel_time,gathering_time,export_time,full_time;
    double duration ;
    int rank,size,chunk_size,remainder,nb_threads;
    int true_chunk_size;
    animated_gif* subgroup;
    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int sendcounts;
    int pos_to_affect;
    int on_gpu;
    nb_threads  = THREADS;
    on_gpu = USE_GPU;
    file_to_save = "collecting_cuda.csv";
    // we need to define a new_p MPI_Datatype MPI_PIXEL
    MPI_Datatype MPI_PIXEL;
    MPI_Type_contiguous(3, MPI_INT, &MPI_PIXEL);
    MPI_Type_commit(&MPI_PIXEL);
    //printf("%d %d \n",rank,size);
    /* Check command-line arguments */
    if ( argc < 4 )
    {
        fprintf( stderr, "Usage: %s input.gif output.gif \n", argv[0] ) ;
        return 1 ;
    }
     input_filename = argv[1] ;
    output_filename = argv[2] ;
    N = argv[3];
    int opt;
    while ((opt = getopt(argc, argv, "t:g:f:")) != -1) {
        
        switch (opt) {
        case 't':
            nb_threads = atoi(optarg);
            break;
        case 'g':
            on_gpu = atoi(optarg);
            break;
        case 'f':
            file_to_save = optarg;
            break;
        }
    }


   

    /* IMPORT Timer start */
    gettimeofday(&t1, NULL);

    /* Load file and store the pixels in array */
    //every process load the image
    image = load_pixels( input_filename ) ;
    // On traite le cas d'un nombre d'images non divisible par size
    gettimeofday(&t2, NULL);

    loading_time = (t2.tv_sec -t1.tv_sec)+((t2.tv_usec-t1.tv_usec)/1e6);
    on_gpu =  on_gpu && test_gpu_available(image->width[0],image->height[0]);
    int images_restantes;
    if ( image == NULL ) { return 1 ; }
    if(on_gpu)
    {
       
        // we have a new_p time : the subgroup allocation time
        gettimeofday(&t1, NULL);
        int chunk_size_0;
        chunk_size_0 = 2 * image->n_images / (size +1);
        images_restantes = image->n_images - 2 * image->n_images / (size +1);
        if(rank == 0)
        {
            
            chunk_size = chunk_size_0;
            true_chunk_size = chunk_size;
            pos_to_affect = 0;
        }
        else
        {
            chunk_size = images_restantes / (size-1);
            remainder = images_restantes % (size-1);
        //Each one will determine how many images it will have to filters, and which ones
            int rank_b;
            rank_b =rank-1;
            pos_to_affect = 2 * image->n_images / (size +1) + chunk_size * rank_b + (rank_b < remainder ? rank_b : remainder);
            true_chunk_size = chunk_size + (rank_b < remainder ? 1 : 0);
        }
    }
    else
    {
        chunk_size = image->n_images / size;
        remainder = image->n_images % size;
    // Each one will determine how many images it will have to filters, and which ones
        pos_to_affect = chunk_size * rank + (rank < remainder ? rank : remainder);
        true_chunk_size = chunk_size + (rank < remainder ? 1 : 0);
    }
    //Allocation
    subgroup = (animated_gif*)malloc(sizeof(animated_gif));
    subgroup->width = (int*)malloc(true_chunk_size * sizeof(int));
    subgroup->height = (int*)malloc(true_chunk_size * sizeof(int));
    subgroup->n_images = true_chunk_size;
    subgroup->p = (pixel**) malloc(true_chunk_size  * sizeof(pixel *));

     
    // We can't use scatterv with image->p because there are memory problems : les pointeurs sont différents dans chaque process à cause de l'adressage virtuel
   
    
    /* IMPORT Timer stop */
    // Each one treat the good images
    int i;
    for(i = 0 ; i<true_chunk_size;i++)
    {
        subgroup->height[i] = image->height[pos_to_affect +i];
        subgroup->width[i] = image->width[pos_to_affect +i];
        subgroup->p[i] = image->p[pos_to_affect +i];
    }

    gettimeofday(&t2,NULL);
    subgroup_time = (t2.tv_sec -t1.tv_sec)+((t2.tv_usec-t1.tv_usec)/1e6); 
    
    
    //printf("Nombre maximum de blocs pour %d threads: %d\n", total_threads, max_blocks);

    
    
    /* FILTER Timer start */
    //printf("before time t1\n");
    gettimeofday(&t1, NULL);
    if(rank == 0 && on_gpu )
    {
        on_gpu = 1;
        apply_gray_filter_cuda(subgroup,nb_threads) ;
   
        apply_blur_filter_cuda(subgroup,5,20,nb_threads);

        apply_sobel_filter_cuda(subgroup,nb_threads) ;

    }
    else
    {   
        on_gpu = 0;
        apply_gray_filter_v(subgroup) ;

        apply_blur_filter_v(subgroup,5,20);
    
        apply_sobel_filter_v(subgroup) ;
        
    }
    
    gettimeofday(&t2, NULL);
    
    sobel_time = (t2.tv_sec -t1.tv_sec)+((t2.tv_usec-t1.tv_usec)/1e6);


    //printf("%lf \n",export_time);

    /* EXPORT Timer start */
    

    /* Store file from array of pixels to GIF file */
    
     if(rank == 0)
    {
        
    // rank 0 doit recevoir les données de tout le monde ça c'est galère
    //gathering Time
        int chunk_size_b,remainder_b;
        gettimeofday(&t1,NULL);
        int * pos_current_rank;
        pos_current_rank = (int*) malloc(size * sizeof(int));
        for(i = 0; i < size; i ++)
        {
            pos_current_rank[i] =0;
        }
        // On va recevoir n_images communications 
        // attention comme on ne sait pas quelle est la taille de l'image recu cela ne marche que si toutes les images on a la même taille

        MPI_Status status;
        if(on_gpu)
        {
            for(i = 0; i < image->n_images - true_chunk_size; i++)
            {
                MPI_Status status;
                
                //printf("We are waiting for someting \n");
                pixel* tmp = (pixel*)malloc(image->width[0] * image->height[0] * sizeof(pixel));
                MPI_Recv(tmp,image->width[0] * image->height[0],MPI_PIXEL,MPI_ANY_SOURCE,0,MPI_COMM_WORLD,&status);
                //printf("We received something from %d \n",status.MPI_SOURCE);
                chunk_size_b = images_restantes / (size-1);
                remainder_b = images_restantes % (size-1);
            //Each one will determine how many images it will have to filters, and which ones
                int rank_b;
                rank_b = status.MPI_SOURCE-1;
                pos_to_affect = 2 * image->n_images / (size +1) + chunk_size_b * rank_b + (rank_b < remainder_b ? rank_b : remainder_b);
                image->p[pos_to_affect +pos_current_rank[status.MPI_SOURCE]] = tmp;
                pos_current_rank[status.MPI_SOURCE]++;
            }
        }
        else
        {
            for(i = 0; i < image->n_images - true_chunk_size; i++)
            {
                MPI_Status status;
                //printf("We are waiting for someting \n");
                pixel* tmp = (pixel*)  malloc(image->width[0] * image->height[0] * sizeof(pixel));
                
                MPI_Recv(tmp,image->width[0] * image->height[0],MPI_PIXEL,MPI_ANY_SOURCE,0,MPI_COMM_WORLD,&status);
                //printf("We received something from %d \n",status.MPI_SOURCE);
                image->p[status.MPI_SOURCE * chunk_size + (status.MPI_SOURCE < remainder ? i : remainder) +pos_current_rank[status.MPI_SOURCE]] = tmp;
                pos_current_rank[status.MPI_SOURCE]++;
            }
        }
        gettimeofday(&t2,NULL);
        gathering_time =(t2.tv_sec -t1.tv_sec)+((t2.tv_usec-t1.tv_usec)/1e6);
        
        gettimeofday(&t1, NULL);
        
        

    }
    else
    {
        int j;
        for(j = 0;j< true_chunk_size;j++)
        {
            MPI_Send(subgroup->p[j],subgroup->width[j] * subgroup->height[j],MPI_PIXEL,0,0,MPI_COMM_WORLD);
        }
    }
    /* EXPORT Timer stop */
    

    //freeing memory 
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank ==0)
    {
    if ( !store_pixels( output_filename, image ) ) { return 1 ; }
        printf("file done \n");
        gettimeofday(&t2, NULL);
        
        export_time = (t2.tv_sec -t1.tv_sec)+((t2.tv_usec-t1.tv_usec)/1e6);
         FILE *f = fopen(file_to_save,"a");
        full_time = loading_time  + sobel_time + export_time;
        fprintf(f,"%s;%d;%d;%d;%d;%d;%lf;%lf;%lf;%lf \n",input_filename,image->n_images,image->width[0] * image->height[0],on_gpu,nb_threads,size,loading_time,sobel_time,export_time,full_time);
        fclose(f);
    }
    
    free(subgroup->height);
    free(subgroup->width);
    
    free(subgroup->p);
        //printf("Done for someone \n");
   
    
    MPI_Finalize();
    return 0 ;
}

