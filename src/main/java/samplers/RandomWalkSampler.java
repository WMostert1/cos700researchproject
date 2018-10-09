package samplers;

import mutators.BitMutator;

public class RandomWalkSampler implements SolutionSampler<boolean []> {

    private boolean [] currentPoint = null;
    private BitMutator bitMutator;
    private int numberOfAttributes;
    private long stepsAllowed;
    private long stepsTaken = 0;
    private int percent = 1;

    public RandomWalkSampler(BitMutator bitMutator, int numAttributes, int stepMultiplier) {
        this.bitMutator = bitMutator;
        this.numberOfAttributes = numAttributes;
        this.stepsAllowed = stepMultiplier*numAttributes;
        System.out.println("Doing step count: "+ stepsAllowed);
    }

    @Override
    public boolean[] getSample() {
         stepsTaken++;
         if (currentPoint == null){
             currentPoint = bitMutator.getRandomPoint(numberOfAttributes);
             return currentPoint;
         }

        currentPoint = bitMutator.mutate(currentPoint);
        return currentPoint;
    }

    @Override
    public boolean isDone() {
        return stepsTaken >= stepsAllowed;
    }

    @Override
    public void showProgress() {
        if((stepsTaken+ 1)/((double)stepsAllowed) > percent/100.0){
            percent++;
            System.out.println(percent+"%");
        }
    }

    @Override
    public void reset() {
        this.currentPoint = null;
        this.stepsTaken = 0;
        this.percent = 1;
    }
}
