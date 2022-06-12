#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define SPHERES 20

#define rnd( x ) (x * rand() / RAND_MAX)
#define INF 2e10f
#define DIM 2048

struct Sphere {
   float   r, b, g;
   float   radius;
   float   x, y, z;
   __device__ float hit(float ox, float oy, float* n) {
      float dx = ox - x;
      float dy = oy - y;
      if (dx * dx + dy * dy < radius * radius) {
         float dz = sqrtf(radius * radius - dx * dx - dy * dy);
         *n = dz / sqrtf(radius * radius);
         return dz + z;
      }
      return -INF;
   }
};

__global__ void kernel(Sphere* s, unsigned char* ptr)
{
   int x = threadIdx.x + blockIdx.x * blockDim.x;
   int y = threadIdx.y + blockIdx.y * blockDim.y;
   int offset = x + y * blockDim.x * gridDim.x;
   float ox = (x - DIM / 2);
   float oy = (y - DIM / 2);

   float r = 0, g = 0, b = 0;
   float maxz = -INF;

   for (int i = 0; i < SPHERES; i++) {
      float   n;
      float   t = s[i].hit(ox, oy, &n);

      if (t > maxz) {
         float fscale = n;
         r = s[i].r * fscale;
         g = s[i].g * fscale;
         b = s[i].b * fscale;
         maxz = t;
      }
   }

   ptr[offset * 4 + 0] = (int)(r * 255);
   ptr[offset * 4 + 1] = (int)(g * 255);
   ptr[offset * 4 + 2] = (int)(b * 255);
   ptr[offset * 4 + 3] = 255;
}

void ppm_write(unsigned char* bitmap, int xdim, int ydim, FILE* fp)
{
   int i, x, y;
   fprintf(fp, "P3\n");
   fprintf(fp, "%d %d\n", xdim, ydim);
   fprintf(fp, "255\n");
   for (y = 0; y < ydim; y++) {
      for (x = 0; x < xdim; x++) {
         i = x + y * xdim;
         fprintf(fp, "%d %d %d ", bitmap[4 * i], bitmap[4 * i + 1], bitmap[4 * i + 2]);
      }
      fprintf(fp, "\n");
   }
}

int main(int argc, char* argv[])
{
  unsigned char* bitmap;
  Sphere* d_s;
  unsigned char* d_bitmap;

  cudaEvent_t start, stop;
  float timeDiff;
  
  FILE* fp = fopen("result_cuda.ppm","w");

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  srand(time(NULL));

  // host malloc
  Sphere* temp_s = (Sphere*)malloc(sizeof(Sphere) * SPHERES);
  for (int i = 0; i < SPHERES; i++) {
    temp_s[i].r = rnd(1.0f);
    temp_s[i].g = rnd(1.0f);
    temp_s[i].b = rnd(1.0f);
    temp_s[i].x = rnd(2000.0f) - 1000;
    temp_s[i].y = rnd(2000.0f) - 1000;
    temp_s[i].z = rnd(2000.0f) - 1000;
    temp_s[i].radius = rnd(200.0f) + 40;
  }
  bitmap = (unsigned char*)malloc(sizeof(unsigned char) * DIM * DIM * 4);

  // device malloc
  cudaMalloc((void**)&d_s, sizeof(Sphere) * SPHERES);
  cudaMalloc((void**)&d_bitmap, sizeof(unsigned char) * DIM * DIM * 4);

  // copy to device from host
  cudaMemcpy(d_s, temp_s, sizeof(Sphere) * SPHERES, cudaMemcpyHostToDevice);

  // start
  cudaEventRecord(start, 0);

  // max 1024 threads in one block and made enough block
  dim3 dimBlock(32, 32);
  dim3 dimGrid(DIM / dimBlock.x, DIM / dimBlock.y);

  // run
  kernel <<<dimGrid, dimBlock>>> (d_s, d_bitmap);
  cudaDeviceSynchronize();

  // copy the result in device to host
   cudaMemcpy(bitmap, d_bitmap, sizeof(unsigned char) * DIM * DIM * 4, cudaMemcpyDeviceToHost);

  // stop
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&timeDiff, start, stop);
  printf("CUDA ray tracing: %f sec \n", timeDiff / 1000);

  ppm_write(bitmap, DIM, DIM, fp);
  printf("[result_cuda.ppm]was generated.\n");

  fclose(fp);
  free(bitmap);
  free(temp_s);

  cudaFree(d_s);
  cudaFree(d_bitmap);

  return 0;
}
