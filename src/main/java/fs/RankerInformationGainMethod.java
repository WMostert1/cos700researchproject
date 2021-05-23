package fs;

import fitness.FitnessEvaluator;
import utils.DataSetInstanceSplitter;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.core.Instances;

import static utils.GlobalConstants.TRAINING_PERCENTAGE;

public class RankerInformationGainMethod extends GenericRankerFeatureSelection implements FeatureSelectionAlgorithm {

    @Override
    public FeatureSelectionResult apply(Instances data, FitnessEvaluator fitnessEvaluator) throws Exception {
        DataSetInstanceSplitter splitter = new DataSetInstanceSplitter(data, TRAINING_PERCENTAGE);
        InfoGainAttributeEval infoGainAttributeEval = buildInfoGainEvaluator(splitter.getTrainingSet());
        return rank(infoGainAttributeEval, fitnessEvaluator, splitter, data);
    }

    @Override
    public String getAlgorithmName() {
        return "IGFS";
    }

    public static InfoGainAttributeEval buildInfoGainEvaluator(Instances data) throws Exception {
        InfoGainAttributeEval infoGainAttributeEval = new InfoGainAttributeEval();
        infoGainAttributeEval.setMissingMerge(true);
        infoGainAttributeEval.buildEvaluator(data);
        return infoGainAttributeEval;
    }
}
