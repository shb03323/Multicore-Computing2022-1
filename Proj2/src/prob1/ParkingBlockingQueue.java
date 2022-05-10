package prob1;

import java.util.concurrent.ArrayBlockingQueue;

public class ParkingBlockingQueue {
    public static void main(String[] args) {
        ParkingGarage parkingGarage = new ParkingGarage(7);
        for (int i=1; i<= 10; i++) {
            Car c = new Car("Car "+i, parkingGarage);
            c.start();
        }
    }
}

class ParkingGarage {
    private final ArrayBlockingQueue<String> places;
    public ParkingGarage(int capacity) {
        if (capacity < 0)
            capacity = 0;
        this.places = new ArrayBlockingQueue<>(capacity);
    }
    public void enter(String name) { // enter parking garage
        try {
            places.put(name);
        } catch(InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }
    public void leave() { // leave parking garage
        try {
            places.take();
        } catch(InterruptedException e) {
            e.printStackTrace();
        }
    }
}

class Car extends Thread {
    private final ParkingGarage parkingGarage;
    public Car(String name, ParkingGarage p) {
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
            parkingGarage.enter(getName());
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
