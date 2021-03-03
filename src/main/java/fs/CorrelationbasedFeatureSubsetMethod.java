package fs;

import fitness.FitnessEvaluator;
import lons.examples.ConcreteBinarySolution;
import utils.DataSetInstanceSplitter;
import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.core.Instances;

import static fs.FeatureSelectorUtils.convertAttributeIndexArrayToBinarySolutionFormat;
import static utils.GlobalConstants.PERCENTAGE_SPLIT;

public class CorrelationbasedFeatureSubsetMethod implements FeatureSelectionAlgorithm {
    @Override
    public FeatureSelectionResult apply(Instances data, FitnessEvaluator fitnessEvaluator) throws Exception {
        DataSetInstanceSplitter splitter = new DataSetInstanceSplitter(data, PERCENTAGE_SPLIT);
        CfsSubsetEval cfsSubsetEval = new CfsSubsetEval();
        cfsSubsetEval.setOptions(weka.core.Utils.splitOptions("weka.attributeSelection.CfsSubsetEval -P 1 -E 1"));
        cfsSubsetEval.buildEvaluator(data);

        int numberOfAttributes = data.numAttributes()-1;

        BestFirst bestFirst = new BestFirst();
        bestFirst.setOptions(weka.core.Utils.splitOptions("weka.attributeSelection.BestFirst -D 1 -N 5"));

        int [] subAttributes = bestFirst.search(cfsSubsetEval, splitter.getTrainingSet());

        ConcreteBinarySolution subSolution = (ConcreteBinarySolution) ConcreteBinarySolution.constructBinarySolution(convertAttributeIndexArrayToBinarySolutionFormat(subAttributes, numberOfAttributes));

        Double accuracy = fitnessEvaluator.getQuality(subSolution.getDesignVector(), data);

        if(accuracy == null){
            throw new RuntimeException("Unknown solution!");
        }

        return new FeatureSelectionResult(FeatureSelectorUtils.getInstancesFromAttributeInclusionIndicesArr(splitter.getTestingSet(), subAttributes), accuracy, subSolution);
    }

    @Override
    public String getAlgorithmName() {
        return "Correlation Based Feature Subset Evaluation";
    }
}
