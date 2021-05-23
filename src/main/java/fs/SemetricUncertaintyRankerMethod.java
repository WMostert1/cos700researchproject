package fs;

import fitness.FitnessEvaluator;


import weka.core.Instances;

public class SemetricUncertaintyRankerMethod extends GenericRankerFeatureSelection implements FeatureSelectionAlgorithm {
    @Override
    public FeatureSelectionResult apply(Instances data, FitnessEvaluator fitnessEvaluator) throws Exception {
        throw new RuntimeException("Not implemented");
//        DataSetInstanceSplitter splitter = new DataSetInstanceSplitter(data, PERCENTAGE_SPLIT);
//        CorrelationAttributeEval correlationEval = new CorrelationAttributeEval();
//        //principalComponents.setOptions(weka.core.Utils.splitOptions("weka.attributeSelection.PrincipalComponents -R 0.95 -A 5"));
//        correlationEval.buildEvaluator(splitter.getTrainingSet());
//        return rank(correlationEval, fitnessEvaluator, splitter, data);
    }

    @Override
    public String getAlgorithmName() {
        return "Pearson Correlation Ranker";
    }
}
