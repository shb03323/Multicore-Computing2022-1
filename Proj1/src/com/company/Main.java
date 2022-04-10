package com.company;

public class Main {
    private static int NUM_END = 10000;
    private static int NUM_THREAD = 4; // assume NUM_END is divisible by NUM_THREAD

    public static void main(String[] args) {
        if (args.length == 2) {
            NUM_THREAD = Integer.parseInt(args[0]);
            NUM_END = Integer.parseInt(args[1]);
        }
        System.out.println("number of threads:"+NUM_THREAD);
        System.out.println("sum from 1 to "+NUM_END+"=");

        int[] int_arr = new int [NUM_END];
        int i,s;
        for (i = 0; i < NUM_END; i++)
            int_arr[i] = i + 1;

        s = sum(int_arr);
        System.out.println("sum=" + s);
    }

    static int sum(int[] arr) {
        // insert your code here
        int len = arr.length;
        int ans = 0;
        SumThread[] sumThread = new SumThread[NUM_THREAD];
        for(int i = 0; i < NUM_THREAD; i++) {
            sumThread[i] = new SumThread(arr, (i * len) / NUM_THREAD, ((i + 1) * len) / NUM_THREAD);
            sumThread[i].start();
        }

        try {
            for (int i = 0; i < NUM_THREAD; i++) {
                sumThread[i].join();
                System.out.println("sum of sumThread[" + i + "]:" + sumThread[i].ans);
                ans += sumThread[i].ans;
            }
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        return ans;
    }
}

class SumThread extends Thread {
    int lo, hi; // fields for communicating inputs
    int[] arr;
    int ans = 0; // for communicating result
    SumThread(int[] a, int l, int h) {
        lo = l;
        hi = h;
        arr = a;
    }
    public void run() {
        // insert your code here
        for (int i = lo; i < hi; i++) {
            ans += arr[i];
        }
    }
}
