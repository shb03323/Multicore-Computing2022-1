#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

long num_steps = 10000000;
double step;

void main(int argc, char* argv[])
{
	long i; double x, pi, sum = 0.0;
	double start_time, end_time;

	int type, chunk_size, thread_num;
	if (argc == 4) {
		// input type & thread number
		type = atoi(argv[1]);
		chunk_size = atoi(argv[2]);
		thread_num = atoi(argv[3]);
	}
	else {
		printf("wrong input\n");
		exit(0);
	}

	// set the thread number
	omp_set_num_threads(thread_num);

	start_time = omp_get_wtime();
	step = 1.0 / (double)num_steps;

	// type 1 : static
	if (type == 1) {
		#pragma omp parallel shared(sum, num_steps, step) private(i, x)
		{
			#pragma omp for reduction(+:sum) schedule(static, chunk_size)
			for (i = 0;i < num_steps; i++) {
				x = (i + 0.5) * step;
				sum = sum + 4.0 / (1.0 + x * x);
			}
		}
	}

	// type 2 : dynamic
	if (type == 2) {
#pragma omp parallel for reduction(+:sum) private(i, x) schedule(dynamic, chunk_size)
		for (i = 0;i < num_steps; i++) {
			x = (i + 0.5) * step;
			sum = sum + 4.0 / (1.0 + x * x);
		}
	}

	// type 3 : guided
	if (type == 3) {
#pragma omp parallel for reduction(+:sum) private(i, x) schedule(guided, chunk_size)
		for (i = 0;i < num_steps; i++) {
			x = (i + 0.5) * step;
			sum = sum + 4.0 / (1.0 + x * x);
		}
	}
	pi = step * sum;

	end_time = omp_get_wtime();
	double timeDiff = end_time - start_time;
	printf("Execution Time : %lfms\n", timeDiff);

	printf("pi=%.24lf\n", pi);
}