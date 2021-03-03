package fs;

import fitness.FitnessEvaluator;
import utils.DataSetInstanceSplitter;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.PrincipalComponents;
import weka.core.Instances;

import static utils.GlobalConstants.PERCENTAGE_SPLIT;

public class RankerPrincipalComponentsMethod extends GenericRankerFeatureSelection implements FeatureSelectionAlgorithm {
    @Override
    public FeatureSelectionResult apply(Instances data, FitnessEvaluator fitnessEvaluator) throws Exception {
        DataSetInstanceSplitter splitter = new DataSetInstanceSplitter(data, PERCENTAGE_SPLIT);
        PrincipalComponents principalComponents = new PrincipalComponents();
        principalComponents.setOptions(weka.core.Utils.splitOptions("weka.attributeSelection.PrincipalComponents -R 0.95 -A 5"));
        principalComponents.buildEvaluator(splitter.getTrainingSet());
        return rank(principalComponents, fitnessEvaluator, splitter, data);
    }

    @Override
    public String getAlgorithmName() {
        return "Principal Components";
    }
}
