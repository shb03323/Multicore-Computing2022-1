package com.company.problem2;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.*;
import java.lang.*;

// command-line execution example) java MatmultD 6 < mat500.txt
// 6 means the number of threads to use
// < mat500.txt means the file that contains two matrices is given as standard input
//
// In eclipse, set the argument value and file input by using the menu [Run]->[Run Configurations]->{[Arguments], [Common->Input File]}.

// Original JAVA source code: http://stackoverflow.com/questions/21547462/how-to-multiply-2-dimensional-arrays-matrix-multiplication
public class MatmultD {
    private static Scanner sc = new Scanner(System.in);
    private static long[] threadTime;
    static int thread_no = 0;
    public static void main(String [] args) {
        File file = new File(System.getProperty("user.dir") + "/" + args[1]);
        try {
            sc = new Scanner(file);

            if (args.length == 2)
                thread_no = Integer.parseInt(args[0]);
            else thread_no = 1;

            int[][] a = readMatrix();
            int[][] b = readMatrix();

            long startTime = System.currentTimeMillis();
            int[][] c = multMatrix(a,b);
            long endTime = System.currentTimeMillis();

//            printMatrix(a);
//            printMatrix(b);
            //printMatrix(c);

            System.out.printf("thread_no: %d\n" , thread_no);
            System.out.printf("Calculation Time: %d ms\n" , endTime-startTime);

            for (int i = 0; i < thread_no; i++) {
                System.out.printf("[thread_no]:%2d , [Time]:%4d ms\n", i, threadTime[i]);
            }
        } catch(FileNotFoundException ignored) {

        }
    }

    public static int[][] readMatrix() {
        int rows = sc.nextInt();
        int cols = sc.nextInt();
        int[][] result = new int[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = sc.nextInt();
            }
        }
        return result;
    }

    public static void printMatrix(int[][] mat) {
        System.out.println("Matrix["+mat.length+"]["+mat[0].length+"]");
        int rows = mat.length;
        int columns = mat[0].length;
        int sum = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                System.out.printf("%4d " , mat[i][j]);
                sum+=mat[i][j];
            }
            System.out.println();
        }
        System.out.println();
        System.out.println("Matrix Sum = " + sum + "\n");
    }

    public static int[][] multMatrix(int a[][], int b[][]){//a[m][n], b[n][p]
        if(a.length == 0) return new int[0][0];
        if(a[0].length != b.length) return null; //invalid dims

        int n = a[0].length;
        int m = a.length;
        int p = b[0].length;
        int ans[][] = new int[m][p];

        MultThread[] mt = new MultThread[thread_no];
        threadTime = new long[thread_no];
        for (int i = 0; i < thread_no; i++) {
            int start = i * (a.length / thread_no);
            int end;
            if (i == thread_no - 1)
                end = a.length;
            else
                end = (i + 1) * (a.length / thread_no);
            mt[i] = new MultThread(i, start, end, a, b, ans);
            long startTime = System.currentTimeMillis();
            threadTime[i] = startTime;
            mt[i].start();
        }

        try {
            for (int i = 0; i < thread_no; i++) {
                mt[i].join();
                long endTime = System.currentTimeMillis();
                threadTime[i] = endTime - threadTime[i];
            }
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        for(int i = 0;i < m;i++){
            for(int j = 0;j < p;j++){
                for(int k = 0;k < n;k++){
                    ans[i][j] += a[i][k] * b[k][j];
                }
            }
        }
        return ans;
    }
}

class MultThread extends Thread {
    int num;
    int start;
    int end;
    int[][] a;
    int[][] b;
    int[][] ans;

    MultThread(int num, int start, int end, int[][] a, int[][] b, int[][] ans) {
        this.num = num;
        this.start = start;
        this.end = end;
        this.a = a;
        this.b = b;
        this.ans = ans;
    }

    public void run() {
        int n = a[0].length;
        int p = b[0].length;

        for (int i = start; i < end; i++) {
            for (int j = 0; j < p; j++) {
                for (int k = 0; k < n; k++) {
                    ans[i][j] += a[i][k] * b[k][j];
                }
            }
        }
    }
}