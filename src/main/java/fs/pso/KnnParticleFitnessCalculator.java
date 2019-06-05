package fs.pso;

import fitness.FitnessEvaluator;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;
import utils.MathUtils;
import weka.core.Instances;

import java.math.BigDecimal;

import static utils.MathUtils.doubleToBigDecimal;

public class KnnParticleFitnessCalculator implements ParticleFitnessCalculator{

    private FitnessEvaluator fitnessEvaluator;
    private Instances data;
    private static final BigDecimal THRESHOLD = doubleToBigDecimal(0.6);

    public KnnParticleFitnessCalculator(FitnessEvaluator fitnessEvaluator, Instances data) {
        this.fitnessEvaluator = fitnessEvaluator;
        this.data = data;
    }

    @Override
    public BigDecimal calculateFitness(Particle p) {
        boolean [] solution = dimensionsToBooleanArray(p);
        try {
            return doubleToBigDecimal(fitnessEvaluator.getQuality(solution, data));
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }




    public static boolean [] dimensionsToBooleanArray(Particle p){
        boolean [] solution = new boolean [p.getEnabledDimensions()];
        int k = 0;
        for(int i = 0; i < p.getPosition().size(); i++){
            if(p.isDisabledDimension(i)){
                continue;
            }
            solution[k++] = p.getPosition(i).compareTo(THRESHOLD) >= 0;
        }
        return solution;
    }
}
