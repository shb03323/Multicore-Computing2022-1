package prob3;

import java.util.concurrent.atomic.AtomicInteger;

public class ex3 {
    public static void main(String[] args) {
        AtomicInteger atomicInteger = new AtomicInteger();
        atomicInteger.set(0);
        for (int i = 1; i <= 10; i ++) {
            PlusAndMinus pm = new PlusAndMinus("Thread "+i, atomicInteger);
            pm.start();
        }
    }
}

class PlusAndMinus extends Thread {
    private final AtomicInteger atomicInteger;

    public PlusAndMinus(String name, AtomicInteger atomicInteger) {
        super(name);
        this.atomicInteger = atomicInteger;
    }

    @Override
    public void run() {
        while (true) {
            try {
                sleep((int)(Math.random() * 10000));
            } catch (InterruptedException e) {}
            System.out.println("Get value : " + atomicInteger.get());
            System.out.println(getName()+": trying to add");
            System.out.println("Added value : " + atomicInteger.addAndGet(1));
            System.out.println(getName()+": just added");
            try {
                sleep((int)(Math.random() * 20000));
            } catch (InterruptedException e) {}
            System.out.println(getName()+": trying to subtract");
            System.out.println("Get and subtract this value : " + atomicInteger.getAndAdd(-1));
            System.out.println(getName()+": just subtracted");
        }
    }
}