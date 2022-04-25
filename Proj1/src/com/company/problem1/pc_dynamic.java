package com.company.problem1;

import java.util.concurrent.atomic.AtomicInteger;

public class pc_dynamic {
    private static int NUM_END = 200000; // default input
    private static int NUM_THREADS = 8; // default number of threads
    public static void main(String[] args) {
        if (args.length == 2) {
            NUM_THREADS = Integer.parseInt(args[0]);
            NUM_END = Integer.parseInt(args[1]);
        }

        System.out.println("Number of Threads : " + NUM_THREADS);

        int counter = 0;
        AtomicInteger index = new AtomicInteger(0); // to count index

        FindingThread3[] ft = new FindingThread3[NUM_THREADS];
        long programStartTime = System.currentTimeMillis();

        for (int i = 0; i < NUM_THREADS; i++) {
            ft[i] = new FindingThread3(index, NUM_END);
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

class FindingThread3 extends Thread {
    AtomicInteger index;
    int counter = 0;
    long executionTime;
    int NUM_END;
    FindingThread3 (AtomicInteger index, int NUM_END) {
        this.index = index;
        this.NUM_END = NUM_END;
    }

    public void run() {
        long startTime = System.currentTimeMillis();
        while (index.get() < NUM_END) {
            if (isPrime(index.getAndAdd(1))) {
                counter++;
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