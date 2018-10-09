package samplers;

import java.util.ArrayList;
import java.util.List;

public class ExhaustiveSampler implements SolutionSampler<boolean []> {
    private int counter = 0;
    private int percent = 1;

    List<boolean []> solutions = new ArrayList<>();

    public ExhaustiveSampler(int numAttributes){
        System.out.println("Initializing exhaustive binary sample set");
        for(int i = 0; i < Math.pow(2.0, numAttributes); i++){
            solutions.add(binaryStringToBinaryArray(Integer.toBinaryString(i), numAttributes));
        }
        System.out.println("Done initializing exhaustive binary sample set");
    }

    private boolean [] binaryStringToBinaryArray(String binaryString, int numAttributes){
        boolean [] arr = new boolean[numAttributes];
        for(int i = binaryString.length() - 1; i >= 0; i--){
            arr[numAttributes - i - 1] = binaryString.charAt(binaryString.length() - i -1) == '1';
        }
        return arr;
    }

    @Override
    public boolean[] getSample() {
        return solutions.get(counter++);
    }

    @Override
    public boolean isDone() {
        return counter >= solutions.size();
    }

    @Override
    public void showProgress() {
        if((counter+ 1)/((double)solutions.size()) > percent/100.0){
            percent++;
            System.out.println(percent+"%");
        }
    }

    @Override
    public void reset() {
        this.counter = 0;
        this.percent = 1;
    }
}
