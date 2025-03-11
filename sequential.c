#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define WIDTH 2048
#define HEIGHT 2048
#define KERNEL_SIZE 5
#define SIGMA 1.0

void gaussian_blur(float input[HEIGHT][WIDTH], 
                   float output[HEIGHT][WIDTH]) {
    float kernel[KERNEL_SIZE][KERNEL_SIZE] = {
        {0.003, 0.013, 0.022, 0.013, 0.003},
        {0.013, 0.059, 0.097, 0.059, 0.013},
        {0.022, 0.097, 0.159, 0.097, 0.022},
        {0.013, 0.059, 0.097, 0.059, 0.013},
        {0.003, 0.013, 0.022, 0.013, 0.003}
    };

    for(int i = 2; i < HEIGHT-2; i++) {
        for(int j = 2; j < WIDTH-2; j++) {
            float sum = 0.0;
            for(int ki = -2; ki <= 2; ki++) {
                for(int kj = -2; kj <= 2; kj++) {
                    sum += input[i+ki][j+kj] * 
                           kernel[ki+2][kj+2];
                }
            }
            output[i][j] = sum;
        }
    }
}

int main() {
    static float input[HEIGHT][WIDTH];
    static float output[HEIGHT][WIDTH];
    
    // Initialize with random values (0-255)
    for(int i = 0; i < HEIGHT; i++) {
        for(int j = 0; j < WIDTH; j++) {
            input[i][j] = (float)rand() / RAND_MAX * 255;
        }
    }

    clock_t start = clock();
    gaussian_blur(input, output);
    clock_t end = clock();
    
    double time = ((double)(end - start)) / CLOCKS_PER_SEC;
    sleep(1);
    printf("Sequential time: %.4f seconds\n", time);
    
    return 0;
}