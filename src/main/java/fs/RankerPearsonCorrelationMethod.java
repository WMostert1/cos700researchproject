package fs;

import fitness.FitnessEvaluator;
import utils.DataSetInstanceSplitter;
import weka.attributeSelection.CorrelationAttributeEval;
import weka.core.Instances;

import static utils.GlobalConstants.TRAINING_PERCENTAGE;

public class RankerPearsonCorrelationMethod extends GenericRankerFeatureSelection implements FeatureSelectionAlgorithm {
    @Override
    public FeatureSelectionResult apply(Instances data, FitnessEvaluator fitnessEvaluator) throws Exception {
        DataSetInstanceSplitter splitter = new DataSetInstanceSplitter(data, TRAINING_PERCENTAGE);
        CorrelationAttributeEval correlationEval = new CorrelationAttributeEval();
        //principalComponents.setOptions(weka.core.Utils.splitOptions("weka.attributeSelection.PrincipalComponents -R 0.95 -A 5"));
        correlationEval.buildEvaluator(splitter.getTrainingSet());
        return rank(correlationEval, fitnessEvaluator, splitter, data);
    }

    @Override
    public String getAlgorithmName() {
        return "PCR";
    }
}
