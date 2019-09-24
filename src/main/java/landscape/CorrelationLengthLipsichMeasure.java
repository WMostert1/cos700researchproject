package landscape;

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import fitness.FitnessEvaluator;
import fs.FeatureSelectionResult;
import mutators.BitMutator;
import mutators.UniformBitMutator;
import org.apache.commons.math3.stat.correlation.PearsonsCorrelation;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class CorrelationLengthLipsichMeasure implements IMeasure {
    private UniformBitMutator uniformBitMutator = new UniformBitMutator(); //default to 1 bit
    Integer apply(Instances data, FitnessEvaluator fitnessEvaluator) throws Exception{
        Map<Integer, boolean []> points = Maps.newHashMap();



        for(int initial = 0; initial < 600; initial++){
           points.put(initial, uniformBitMutator.getRandomPoint(data.numAttributes()-1));
        }

        for(int i = 1; i <= 30; i++){
            List<Double> miFitnessList = Lists.newArrayList();
        }


        //double miFit
        boolean [] minitial = uniformBitMutator.getRandomPoint(data.numAttributes()-1);
        for(int i = 0 ; i < 30; i++){
            boolean [] mi = uniformBitMutator.mutate(minitial);
            double miFitness = fitnessEvaluator.getQuality(mi, data);
        }
        return null;
    }
}
