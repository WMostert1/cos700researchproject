package samplers;

import com.google.common.base.Stopwatch;
import com.google.common.collect.Lists;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;

public class ExhaustiveSampler implements SolutionSampler<boolean []> {
    private int counter = 0;
    private int percent = 1;
    private  long prevMillis = 0;
    private  Stopwatch stopwatch;

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

    public synchronized List<boolean[]> getBatchSample(int batchSize){
        List<boolean[]> batch = Lists.newArrayList();
        for(int i = 0 ; i < batchSize && !isDone(); i++){
            batch.add(getSample());
        }
        return batch;
    }

    @Override
    public boolean[] getSample() {
        if(stopwatch == null){
            stopwatch = Stopwatch.createStarted();
        }
        return solutions.get(counter++);
    }

    @Override
    public boolean isDone() {
        boolean done = counter >= solutions.size();
        if(done){
            stopwatch.stop();
        }
        return done;
    }

    @Override
    public void showProgress() {

        long millis = stopwatch.elapsed(TimeUnit.MILLISECONDS);
        prevMillis = millis - prevMillis;

        long timeLeftInSeconds = (prevMillis * (solutions.size() -counter))/1000 ;

        if((counter+ 1)/((double)solutions.size()) > percent/100.0){
            percent++;

            String timeLeft = timeLeftInSeconds / 60.0 >= 1 ? Double.valueOf(timeLeftInSeconds / 60.0).toString() + " minutes " : timeLeftInSeconds+" seconds";

            System.out.println(percent+"% - estimated time left : "+timeLeft);
        }
    }

    @Override
    public void reset() {
        this.counter = 0;
        this.percent = 1;
        this.stopwatch = null;
        this.prevMillis = 0;
    }
}
