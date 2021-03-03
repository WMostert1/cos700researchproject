package fs;

import com.google.common.collect.Lists;
import fitness.FitnessEvaluator;
import lons.examples.ConcreteBinarySolution;
import utils.MathUtils;
import weka.core.Instances;

import java.math.BigDecimal;
import java.security.SecureRandom;
import java.util.ArrayList;
import java.util.List;
import java.util.TreeMap;

public class RandomFeatureSelection extends StochasticFeatureSelection implements FeatureSelectionAlgorithm {

    private final int NO_RUNS = 30;
    private final SecureRandom random = new SecureRandom();
    private ArrayList<BigDecimal> iterationFitnessValues = Lists.newArrayList();

    @Override
    public FeatureSelectionResult apply(Instances data, FitnessEvaluator fitnessEvaluator) throws Exception {

        iterationFitnessValues.clear();
        Double bestFitness = null;
        boolean [] bestSolution = null;
        BigDecimal averageAccuracy = BigDecimal.ZERO;

        int numAttributes = data.numAttributes() -1;
        for(int run = 0; run < NO_RUNS; run++){
            boolean [] solution = new boolean[numAttributes];
            for(int i = 0; i < numAttributes; i++){
                solution[i] = random.nextBoolean();
            }

            Double fitness = fitnessEvaluator.getQuality(solution, data);
            if(bestFitness == null || fitness > bestFitness){
                bestFitness = fitness;
                bestSolution = solution;
            }

            BigDecimal bdFitness = MathUtils.doubleToBigDecimal(fitness);


            averageAccuracy = averageAccuracy.add(bdFitness);
            iterationFitnessValues.add(bdFitness);

        }


         averageAccuracy = averageAccuracy.divide(BigDecimal.valueOf(NO_RUNS), MathUtils.ROUNDING_MODE);
        return new FeatureSelectionResult(data, averageAccuracy.doubleValue(), ConcreteBinarySolution.constructBinarySolution(bestSolution));
    }

    @Override
    public String getAlgorithmName() {
        return "Random";
    }

    @Override
    public ArrayList<BigDecimal> getIterationFitnessValues() {
        return iterationFitnessValues;
    }
}
