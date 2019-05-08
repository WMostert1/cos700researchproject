package fs;

import fitness.FitnessEvaluator;
import weka.core.Instances;

public interface FeatureSelectionAlgorithm {
    FeatureSelectionResult apply(Instances data, FitnessEvaluator fitnessEvaluator) throws Exception;

    String getAlgorithmName();
}
