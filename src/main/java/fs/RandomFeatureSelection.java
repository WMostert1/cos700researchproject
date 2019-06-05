package fs;

import fitness.FitnessEvaluator;
import lons.examples.ConcreteBinarySolution;
import utils.MathUtils;
import weka.core.Instances;

import java.math.BigDecimal;
import java.security.SecureRandom;
import java.util.TreeMap;

public class RandomFeatureSelection implements FeatureSelectionAlgorithm {

    private final int NO_RUNS = 30;
    private final SecureRandom random = new SecureRandom();

    @Override
    public FeatureSelectionResult apply(Instances data, FitnessEvaluator fitnessEvaluator) throws Exception {
        Double bestFitness = null;
        boolean [] bestSolution = null;
        BigDecimal averageAccuracy = BigDecimal.ZERO;

        int numAttributes = data.numAttributes() -1;
        for(int run = 1; run < NO_RUNS; run++){
            boolean [] solution = new boolean[numAttributes];
            for(int i = 0; i < numAttributes; i++){
                solution[i] = random.nextBoolean();
            }

            Double fitness = fitnessEvaluator.getQuality(solution, data);
            if(bestFitness == null || fitness > bestFitness){
                bestFitness = fitness;
                bestSolution = solution;
            }

            averageAccuracy = averageAccuracy.add(MathUtils.doubleToBigDecimal(fitness));
        }

         averageAccuracy = averageAccuracy.divide(BigDecimal.valueOf(NO_RUNS), MathUtils.ROUNDING_MODE);
        return new FeatureSelectionResult(data, averageAccuracy.doubleValue(), ConcreteBinarySolution.constructBinarySolution(bestSolution));
    }

    @Override
    public String getAlgorithmName() {
        return "Random Feature Selection";
    }
}
