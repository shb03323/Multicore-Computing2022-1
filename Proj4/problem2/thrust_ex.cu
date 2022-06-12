#include <thrust/transform_reduce.h>
#include <thrust/iterator/counting_iterator.h>

long num_steps = 1000000000; 
double step = 1.0/(double) num_steps;

template<typename T>
struct calculate_pi {
  double step;
  // parameter
  calculate_pi(double step) : step(step){}
  // operation
  __host__ __device__ 
  T operator()(const T &i) const 
  {
    double x = (i+0.5)*step;
    return 4.0/(1.0+x*x);
  }
};

int main ()
{ 
  // start point
  auto begin = thrust::counting_iterator<double>(0);
  // end point
  auto end = thrust::counting_iterator<double>(num_steps);

  step = 1.0/(double) num_steps;

  calculate_pi<double> unary_op(step);
  thrust::plus<double> binary_op;

  cudaEvent_t start, stop;
  float timeDiff;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  srand(time(NULL));
  // start
  cudaEventRecord(start, 0);

  // transform reudce
  double sum = thrust::transform_reduce(begin, end, unary_op, 0.0, binary_op);

  // stop
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&timeDiff, start, stop);

  double pi = step * sum;
  printf("pi=%.8lf\n",pi);
  printf("execute time : %f sec \n", timeDiff / 1000);
}
