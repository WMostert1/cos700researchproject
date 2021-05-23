package fs;

import fitness.FitnessEvaluator;
import utils.DataSetInstanceSplitter;
import weka.attributeSelection.GreedyStepwise;
import weka.core.Instances;

import static utils.GlobalConstants.TRAINING_PERCENTAGE;

public class SequentialForwardSelection extends SequentialSelection implements FeatureSelectionAlgorithm {
    @Override
    public FeatureSelectionResult apply(Instances data, FitnessEvaluator fitnessEvaluator) throws Exception {
        GreedyStepwise greedyStepwise = new GreedyStepwise();
        greedyStepwise.setOptions(weka.core.Utils.splitOptions("weka.attributeSelection.GreedyStepwise -N -1 -num-slots 1 "));

        DataSetInstanceSplitter splitter = new DataSetInstanceSplitter(data, TRAINING_PERCENTAGE);
        return seqSelect(greedyStepwise, fitnessEvaluator, data, splitter);
    }

    @Override
    public String getAlgorithmName() {
        return "SFS";
    }
}
