package com.company.problem1;

public class pc_static_block {
    private static int NUM_END = 200000; // default input
    private static int NUM_THREADS = 8; // default number of threads
    public static void main(String[] args) {
        if (args.length == 2) {
            NUM_THREADS = Integer.parseInt(args[0]);
            NUM_END = Integer.parseInt(args[1]);
        }

        System.out.println("Number of Threads : " + NUM_THREADS);

        int counter = 0;
        FindingThread1[] ft = new FindingThread1[NUM_THREADS];
        long programStartTime = System.currentTimeMillis();

        for (int i = 0; i < NUM_THREADS; i++) {
            if (i == NUM_THREADS - 1)
                ft[i] = new FindingThread1((i * NUM_END) / NUM_THREADS, NUM_THREADS);
            else
                ft[i] = new FindingThread1((i * NUM_END) / NUM_THREADS, ((i + 1) * NUM_END) / NUM_THREADS);
            ft[i].start();
        }

        try {
            for (int i = 0; i < NUM_THREADS; i++) {
                ft[i].join();
                counter += ft[i].counter;
            }
        } catch (InterruptedException e) { }

        long programEndTime = System.currentTimeMillis();
        long timeDiff = programEndTime - programStartTime;

        for (int i = 0; i < NUM_THREADS; i++) {
            System.out.println("Thread #" + i + "'s execution time : " + (ft[i].executionTime) + "ms");
        }

        System.out.println("Program Execution Time : " + timeDiff + "ms");
        System.out.println("1..." + (NUM_END - 1) + " prime# counter = " + counter);
    }
}

class FindingThread1 extends Thread {
    int counter = 0;
    long executionTime;
    int low, high;
    FindingThread1 (int low, int high) {
        this.low = low;
        this.high = high;
    }

    public void run() {
        long startTime = System.currentTimeMillis();
        for (int i = low; i < high; i++) {
            if (isPrime(i)) {
                counter += 1;
            }
        }
        long endTime = System.currentTimeMillis();
        executionTime = endTime - startTime;
    }

    private static boolean isPrime(int x) {
        if (x <= 1)
            return false;
        for (int i = 2; i < x; i++) {
            if (x % i == 0)
                return false;
        }
        return true;
    }
}
