#include <omp.h>
#include <stdio.h>
#include <stdlib.h>   // for rand(), srand()
#include <time.h>     // for time(NULL)

#define WIDTH 2048
#define HEIGHT 2048
#define KERNEL_SIZE 5

void gaussian_blur_parallel(float input[HEIGHT][WIDTH],
                            float output[HEIGHT][WIDTH]) {
    float kernel[KERNEL_SIZE][KERNEL_SIZE] = {
        {0.003f, 0.013f, 0.022f, 0.013f, 0.003f},
        {0.013f, 0.059f, 0.097f, 0.059f, 0.013f},
        {0.022f, 0.097f, 0.159f, 0.097f, 0.022f},
        {0.013f, 0.059f, 0.097f, 0.059f, 0.013f},
        {0.003f, 0.013f, 0.022f, 0.013f, 0.003f}
    };

    // Parallel region with a collapsed 2D loop
    #pragma omp parallel for collapse(2) schedule(static) \
        default(none) shared(input, output, kernel)
    for (int i = 2; i < HEIGHT - 2; i++) {
        for (int j = 2; j < WIDTH - 2; j++) {
            float sum = 0.0f;

            // SIMD vectorization for the inner kernel loop
            #pragma omp simd reduction(+:sum)
            for (int ki = -2; ki <= 2; ki++) {
                for (int kj = -2; kj <= 2; kj++) {
                    sum += input[i + ki][j + kj] * 
                           kernel[ki + 2][kj + 2];
                }
            }
            // No critical needed since each thread writes a unique (i, j)
            output[i][j] = sum;
        }
    }
}

int main() {
    static float input[HEIGHT][WIDTH];
    static float output[HEIGHT][WIDTH];

    // Initialize input with random values
    srand((unsigned)time(NULL));
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            input[i][j] = (float)rand() / RAND_MAX * 255.0f;
        }
    }

    // Measure wall-clock time with OpenMP's timer
    double start = omp_get_wtime();
    gaussian_blur_parallel(input, output);
    double end = omp_get_wtime();

    printf("Parallel time (%d threads): %.4f seconds\n", 
           omp_get_max_threads(), end - start);

    return 0;
}
