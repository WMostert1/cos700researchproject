package fs;

import com.google.common.collect.Lists;
import fitness.FitnessEvaluator;
import fs.pso.BoothFunction;
import fs.pso.GbestPSO;
import fs.pso.KnnParticleFitnessCalculator;
import fs.pso.Particle;
import lons.examples.BinarySolution;
import lons.examples.ConcreteBinarySolution;
import utils.MathUtils;
import weka.core.Instances;

import java.math.BigDecimal;
import java.util.List;

import static utils.MathUtils.doubleToBigDecimal;

public class FullPSOSearchFeatureSelection implements FeatureSelectionAlgorithm {
    private final int PSO_RUNS = 30;
    @Override
    public FeatureSelectionResult apply(Instances data, FitnessEvaluator fitnessEvaluator) throws Exception {
        List<Particle> bestParticles = Lists.newArrayList();
        for(int i = 0; i < 30; i++) {
            GbestPSO pso = new GbestPSO(
                    new KnnParticleFitnessCalculator(fitnessEvaluator, data),
                    data.numAttributes() - 1,
                    5,
                    doubleToBigDecimal(1.49445),
                    doubleToBigDecimal(1.49445),
                    doubleToBigDecimal(0.0),
                    doubleToBigDecimal(1.0),
                    100
            );
            bestParticles.add(pso.optimize());
        }

        BigDecimal averageAccuracy = BigDecimal.ZERO;
        Particle bestParticle = null;
        for(Particle p : bestParticles){
            if(bestParticle == null){
                bestParticle = p;
            }else if (p.getFitness().compareTo(bestParticle.getFitness()) > 0){
                bestParticle = p;
            }
            averageAccuracy = averageAccuracy.add(p.getFitness());
        }

        averageAccuracy = averageAccuracy.divide(BigDecimal.valueOf(PSO_RUNS), MathUtils.ROUNDING_MODE);



        FeatureSelectionResult result = new FeatureSelectionResult(
                data, averageAccuracy.doubleValue(), ConcreteBinarySolution.constructBinarySolution(KnnParticleFitnessCalculator.dimensionsToBooleanArray(bestParticle))
        );
        return result;
    }

    @Override
    public String getAlgorithmName() {
        return "Full GBest PSO";
    }
}
