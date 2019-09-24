package landscape;

import com.google.common.collect.Maps;
import fitness.FitnessEvaluator;
import mutators.UniformBitMutator;
import utils.OutputFormatter;
import weka.core.Instances;

import java.util.HashMap;
import java.util.Map;

public class FitnessDistributionMeasure implements IMeasure{
    private int numberOfIsoLevels;
    private double lowerBound;
    private double upperBound;
    private Map<Integer, Map<boolean[],Double>> sampleBins;

    private UniformBitMutator uniformBitMutator = new UniformBitMutator();
    int scalingConstant;

    public FitnessDistributionMeasure(int numberOfIsoLevels, double lowerBound, double upperBound, int scalingConstant) {
        this.numberOfIsoLevels = numberOfIsoLevels;
        this.lowerBound = lowerBound;
        this.upperBound = upperBound;
        sampleBins = new HashMap<>();
        this.scalingConstant = scalingConstant;
    }

    public Map<Integer, Map<boolean[],Double>> get(Instances data, FitnessEvaluator fitnessEvaluator) throws Exception {
        double increment = (upperBound - lowerBound) / numberOfIsoLevels;
        //increment = bin sizes
        for(int i = 0; i < numberOfIsoLevels;i++){
            sampleBins.put(i,new HashMap<>());
        }

        int numberOfSteps = scalingConstant * data.numAttributes()-1;
        Map<boolean[], Double> fitnessMap = Maps.newHashMap();
        for(int i = 0; i < numberOfSteps; i++){
            boolean [] samplePoint = uniformBitMutator.getRandomPoint(data.numAttributes()-1);
            if(fitnessMap.get(samplePoint) == null){
                fitnessMap.put(samplePoint, fitnessEvaluator.getQuality(samplePoint, data));
            }else{
                i--;
            }
        }

        fitnessMap.forEach((boolean [] key, Double fitness) -> {
            for (int i = 0; i < numberOfIsoLevels; i++) {
                double isoLow = lowerBound + (increment * i);
                double isoHigh = lowerBound + (increment * (i + 1));
                if (fitness >= isoLow && fitness < isoHigh) {
                    sampleBins.get(i).put(key, fitness);
                    break;
                }
            }
        });

        return sampleBins;
    }
}
