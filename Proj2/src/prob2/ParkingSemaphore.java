package prob2;

import java.util.concurrent.Semaphore;

public class ParkingSemaphore {
    public static void main(String[] args) {
        ParkingGarage2 parkingGarage = new ParkingGarage2(7);
        for (int i=1; i<= 10; i++) {
            Car c = new Car("Car "+i, parkingGarage);
            c.start();
        }
    }
}

class ParkingGarage2 {
    private final Semaphore places;
    public ParkingGarage2(int capacity) {
        if (capacity < 0)
            capacity = 0;
        places = new Semaphore(capacity);
    }
    public void enter() { // enter parking garage
        try {
            places.acquire();
        } catch(InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }
    public void leave() { // leave parking garage
        places.release();
    }
}

class Car extends Thread {
    private final ParkingGarage2 parkingGarage;
    public Car(String name, ParkingGarage2 p) {
        super(name);
        this.parkingGarage = p;
    }

    private void tryingEnter()
    {
        System.out.println(getName()+": trying to enter");
    }


    private void justEntered()
    {
        System.out.println(getName()+": just entered");

    }

    private void aboutToLeave()
    {
        System.out.println(getName()+":                                     about to leave");
    }

    private void Left()
    {
        System.out.println(getName()+":                                     have been left");
    }

    public void run() {
        while (true) {
            try {
                sleep((int)(Math.random() * 10000)); // drive before parking
            } catch (InterruptedException e) {}
            tryingEnter();
            parkingGarage.enter();
            justEntered();
            try {
                sleep((int)(Math.random() * 20000)); // stay within the parking garage
            } catch (InterruptedException e) {}
            aboutToLeave();
            parkingGarage.leave();
            Left();
        }
    }
}
