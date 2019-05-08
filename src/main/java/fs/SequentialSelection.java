package fs;

import fitness.FitnessEvaluator;
import lons.examples.ConcreteBinarySolution;
import utils.DataSetInstanceSplitter;
import weka.attributeSelection.ASSearch;
import weka.core.Instances;

import static fs.FeatureSelectorUtils.convertAttributeIndexArrayToBinarySolutionFormat;

public abstract class SequentialSelection implements FeatureSelectionAlgorithm {
    public FeatureSelectionResult seqSelect(ASSearch search, FitnessEvaluator fitnessEvaluator, Instances data, DataSetInstanceSplitter splitter) throws  Exception{
        WrapperSplitSetsEvalutator wrapper = new WrapperSplitSetsEvalutator(fitnessEvaluator);
        wrapper.buildEvaluator(data);

        int numberOfAttributes = data.numAttributes()-1;

        int [] subAttributes = search.search(wrapper, splitter.getTrainingSet());

        ConcreteBinarySolution subSolution = (ConcreteBinarySolution) ConcreteBinarySolution.constructBinarySolution(convertAttributeIndexArrayToBinarySolutionFormat(subAttributes, numberOfAttributes));

        Double accuracy = fitnessEvaluator.getQuality(subSolution.getDesignVector(), data);

        if(accuracy == null){
            throw new RuntimeException("Unknown solution!");
        }

        return new FeatureSelectionResult(FeatureSelectorUtils.getInstancesFromAttributeInclusionIndicesArr(splitter.getTestingSet(), subAttributes), accuracy, subSolution);
    }
}
