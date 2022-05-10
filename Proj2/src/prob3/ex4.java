package prob3;

import java.util.concurrent.BrokenBarrierException;
import java.util.concurrent.CyclicBarrier;

public class ex4 {
    public static void main(String[] args) {
        CyclicBarrier cyclicBarrier = new CyclicBarrier(5);
        for (int i = 0; i < 10; ++i) {
            CyclicThread ct = new CyclicThread("Thread " + i, cyclicBarrier);
            ct.start();
        }
    }
}

class CyclicThread extends Thread {
    private final CyclicBarrier cyclicBarrier;

    public CyclicThread(String name, CyclicBarrier cyclicBarrier) {
        super(name);
        this.cyclicBarrier = cyclicBarrier;
    }

    @Override
    public void run() {
        System.out.println(getName()+": Started");
        try {
            sleep((int)(Math.random() * 10000));
            System.out.println(getName()+": await");
            cyclicBarrier.await();
        } catch (InterruptedException | BrokenBarrierException e) {
            throw new RuntimeException(e);
        }
        System.out.println(getName()+": Ended");
    }
}
