package com.company.problem1;

public class pc_static_cyclic {
    private static int NUM_END = 200000; // default input
    private static int NUM_THREADS = 8; // default number of threads
    public static void main(String[] args) {
        if (args.length == 2) {
            NUM_THREADS = Integer.parseInt(args[0]);
            NUM_END = Integer.parseInt(args[1]);
        }

        System.out.println("Number of Threads : " + NUM_THREADS);

        int counter = 0;
        FindingThread2[] ft = new FindingThread2[NUM_THREADS];
        long programStartTime = System.currentTimeMillis();

        for (int i = 0; i < NUM_THREADS; i++) {
            ft[i] = new FindingThread2(i, NUM_END, NUM_THREADS);
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

class FindingThread2 extends Thread {
    int counter = 0;
    long executionTime;
    int remainder, NUM_END, NUM_THREADS;
    FindingThread2 (int remainder, int NUM_END, int NUM_THREADS) {
        this.remainder = remainder;
        this.NUM_END = NUM_END;
        this.NUM_THREADS = NUM_THREADS;
    }

    public void run() {
        long startTime = System.currentTimeMillis();
        for (int i = remainder; i < NUM_END; i += NUM_THREADS) {
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

