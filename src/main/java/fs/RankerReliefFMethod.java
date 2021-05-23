package fs;

import fitness.FitnessEvaluator;
import utils.DataSetInstanceSplitter;
import weka.attributeSelection.ReliefFAttributeEval;
import weka.core.Instances;

import static utils.GlobalConstants.TRAINING_PERCENTAGE;

public class RankerReliefFMethod extends GenericRankerFeatureSelection implements FeatureSelectionAlgorithm {

    @Override
    public FeatureSelectionResult apply(final Instances data, final FitnessEvaluator fitnessEvaluator) throws Exception {
        DataSetInstanceSplitter splitter = new DataSetInstanceSplitter(data, TRAINING_PERCENTAGE);
        ReliefFAttributeEval reliefEval = new ReliefFAttributeEval();
        reliefEval.buildEvaluator(splitter.getTrainingSet());
        return rank(reliefEval, fitnessEvaluator, splitter, data);
    }

    @Override
    public String getAlgorithmName() {
        return "REL";
    }
}
