package prob3;

import java.util.concurrent.locks.ReentrantReadWriteLock;

public class ex2 {
    public static void main(String[] args) {
        ReentrantReadWriteLock lock = new ReentrantReadWriteLock();
        for (int i = 1; i <= 5; i ++) {
            Reader r = new Reader("reader " + i, lock);
            Writer w = new Writer("writer " + i, lock);

            r.start();
            w.start();
        }
    }
}

class Reader extends Thread {
    private final ReentrantReadWriteLock lock;

    public Reader(String name, ReentrantReadWriteLock lock) {
        super(name);
        this.lock = lock;
    }

    @Override
    public void run() {
        lock.readLock().lock();
        System.out.println(getName()+": locked");

        try {
            sleep((int)(Math.random() * 1000));
        } catch (InterruptedException e) {
            e.printStackTrace();
        } finally {
            System.out.println(getName()+": unlocking");
            lock.readLock().unlock();
        }
    }
}

class Writer extends Thread {
    private final ReentrantReadWriteLock lock;

    public Writer(String name, ReentrantReadWriteLock lock) {
        super(name);
        this.lock = lock;
    }

    @Override
    public void run() {
        lock.writeLock().lock();
        System.out.println(getName()+":                                     locked");
        try {
            sleep((int)(Math.random() * 2000));
        } catch (InterruptedException e) {
            e.printStackTrace();
        } finally {
            System.out.println(getName()+":                                     unlocking");
            lock.writeLock().unlock();
        }
    }
}
