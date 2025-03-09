#include <omp.h>
#include <stdio.h>

#define WIDTH 2048
#define HEIGHT 2048
#define KERNEL_SIZE 5
#define SIGMA 1.0

void gaussian_blur_parallel(float input[HEIGHT][WIDTH],
                            float output[HEIGHT][WIDTH]) {
    float kernel[KERNEL_SIZE][KERNEL_SIZE] = { {0.003, 0.013, 0.022, 0.013, 0.003},
    {0.013, 0.059, 0.097, 0.059, 0.013},
    {0.022, 0.097, 0.159, 0.097, 0.022},
    {0.013, 0.059, 0.097, 0.059, 0.013},
    {0.003, 0.013, 0.022, 0.013, 0.003}};

    #pragma omp parallel for collapse(2) schedule(dynamic) \
        default(none) shared(input, output, kernel) 
    for(int i = 2; i < HEIGHT-2; i++) {
        for(int j = 2; j < WIDTH-2; j++) {
            float sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for(int ki = -2; ki <= 2; ki++) {
                for(int kj = -2; kj <= 2; kj++) {
                    sum += input[i+ki][j+kj] * 
                           kernel[ki+2][kj+2];
                }
            }
            #pragma omp critical
            output[i][j] = sum;
        }
    }
}

int main() {
    static float input[HEIGHT][WIDTH];
    static float output[HEIGHT][WIDTH];
    double start = omp_get_wtime();
    gaussian_blur_parallel(input, output);
    double end = omp_get_wtime();
    
    printf("Parallel time (%d threads): %.4f seconds\n", 
           omp_get_max_threads(), end - start);
    
    return 0;
}