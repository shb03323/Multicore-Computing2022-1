package prob3;

import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;

public class ex1 {
    public static void main(String[] args) {
        BlockingQueue<String> blockingQueue = new ArrayBlockingQueue<>(10);

        for (int i = 1; i <= 5; i ++) {
            Producer p = new Producer("producer " + i, blockingQueue);
            Consumer c = new Consumer("consumer " + i, blockingQueue);

            p.start();
            c.start();
        }
    }
}

class Producer extends Thread {
    BlockingQueue<String> blockingQueue;

    public Producer(String name, BlockingQueue<String> blockingQueue){
        super(name);
        this.blockingQueue = blockingQueue;
    }

    @Override
    public void run() {
        while (true){
            try {
                sleep((int)(Math.random() * 10000));
                System.out.println(getName()+": trying to produce");
                blockingQueue.put(getName());
                System.out.println(getName()+": just produced");
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
        }
    }
}

class Consumer extends Thread {
    BlockingQueue<String> blockingQueue;

    public Consumer(String name, BlockingQueue<String> blockingQueue){
        super(name);
        this.blockingQueue = blockingQueue;
    }

    @Override
    public void run() {
        while (true) {
            try {
                sleep((int)(Math.random() * 20000));
                System.out.println(getName()+":                                     about to consume");
                blockingQueue.take();
                System.out.println(getName()+":                                     have been consumed");
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }
}