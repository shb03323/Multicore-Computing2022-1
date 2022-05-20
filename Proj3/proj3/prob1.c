#include <stdio.h>
#include <omp.h>
#include <stdbool.h>
#include <stdlib.h>

#define END_NUM 200000

// function for checking prime
bool isPrime(int src) {
	for (int i = 2; src > i; i++) {
		if (0 == src % i)
			return false;
	}
	return true;
}

int main(int argc, char* argv[]) {
	int type, thread_num;
	if (argc == 3) {
		// input type & thread number
		type = atoi(argv[1]);
		thread_num = atoi(argv[2]);
	}
	else {
		printf("wrong input\n");
		exit(0);
	}
	
	// set the thread number
	omp_set_num_threads(thread_num);

	// counting prime number
	int result = 0;
	// set time
	double start_time, end_time;
	start_time = omp_get_wtime();
	int i;

	// type 1 : static, default chunk
	if (type == 1) {
#pragma omp parallel for reduction(+:result) schedule(static)
		for (i = 2; i < END_NUM; i++) {
			if (isPrime(i))
				result++;
		}
	}

	// type 2 : dynamic, default chunk
	else if (type == 2) {
#pragma omp parallel for reduction(+:result) schedule(dynamic)
		for (int i = 2; i < END_NUM; i++) {
			if (isPrime(i))
				result++;
		}
	}

	// type 3 : static, 10 chunk
	else if (type == 3) {
#pragma omp parallel for reduction(+:result) schedule(static, 10)
		for (int i = 2; i < END_NUM; i++) {
			if (isPrime(i))
				result++;
		}
	}

	// type 4 : dynamic, 10 chunk
	else if (type == 4) {
#pragma omp parallel for reduction(+:result) schedule(dynamic, 10)
		for (int i = 2; i < END_NUM; i++) {
			if (isPrime(i))
				result++;
		}
	}

	else {
		printf("Type error : input only type 1~4\n");
		exit(0);
	}

	end_time = omp_get_wtime();
	double timeDiff = end_time - start_time;

	if (type == 1) printf("STATIC DEFAULT CHUNK\n");
	else if (type == 2) printf("DYNAMIC DEFAULT CHUNK\n");
	else if (type == 3) printf("STATIC 10 CHUNK\n");
	else if (type == 4) printf("DYNAMIC 10 CHUNK\n");

	printf("Number of threads : %d\n", thread_num);
	printf("Execution time : %lfs\n", timeDiff);
	printf("Result : %d\n", result);
}