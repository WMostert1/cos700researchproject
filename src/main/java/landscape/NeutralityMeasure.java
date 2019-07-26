package landscape;

import fitness.FitnessEvaluator;
import mutators.UniformBitMutator;
import org.apache.commons.math3.util.Pair;
import samplers.RandomWalkSampler;
import utils.MathUtils;
import weka.core.Instances;
import weka.core.PartitionGenerator;

import java.math.BigDecimal;
import java.util.Queue;
import java.util.concurrent.ArrayBlockingQueue;

//Abrie's measure
public class NeutralityMeasure implements IMeasure {
    private RandomWalkSampler randomWalkSampler;
    private static final int THREE_POINT_STRUCTURE_STEP = 3;
    public Pair<BigDecimal, BigDecimal> get(FitnessEvaluator fitnessEvaluator, Instances data) throws Exception {
        randomWalkSampler = new RandomWalkSampler(new UniformBitMutator(1),data.numAttributes()-1, 20);

        ArrayBlockingQueue<BigDecimal> threeLastStepFitness = new ArrayBlockingQueue<>(THREE_POINT_STRUCTURE_STEP);

        int sneutral = 0;

        int largestSequence = 0;
        int currentSequence = 0;


        do {
            addAndEvict(threeLastStepFitness,MathUtils.doubleToBigDecimal(fitnessEvaluator.getQuality(randomWalkSampler.getSample(), data)));
            if(threeLastStepFitness.size() >= THREE_POINT_STRUCTURE_STEP){
                if(isQueueEqual(threeLastStepFitness)){
                    sneutral++;
                    currentSequence++;
                }else{
                    if(currentSequence > largestSequence){
                        largestSequence = currentSequence;
                    }
                    currentSequence = 0;
                }
            }
        }while (!randomWalkSampler.isDone());

        largestSequence = Math.max(currentSequence, largestSequence);


        BigDecimal m1 = MathUtils.doubleToBigDecimal((double)sneutral/randomWalkSampler.getStepsTaken());
        BigDecimal m2 = MathUtils.doubleToBigDecimal((double)largestSequence/randomWalkSampler.getStepsTaken());

        return new Pair<>(m1,m2);
    }

    private void addAndEvict(ArrayBlockingQueue<BigDecimal> queue, BigDecimal value){
        if(queue.size() >= THREE_POINT_STRUCTURE_STEP){
            queue.remove();
        }
        queue.add(value);
    }

    private boolean isQueueEqual(ArrayBlockingQueue<BigDecimal> queue){
        if(queue.isEmpty()){
            throw new IllegalStateException("Queue does not contain " + THREE_POINT_STRUCTURE_STEP + " elements.");
        }
        BigDecimal [] arr = queue.toArray(new BigDecimal[queue.size()]);
        for(int i = 1 ; i < THREE_POINT_STRUCTURE_STEP; i++){
            if(arr[i].compareTo(arr[i-1]) != 0){
                return false;
            }
        }
        return true;
    }
}
