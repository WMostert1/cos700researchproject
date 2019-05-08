package fs;

import fitness.FitnessEvaluator;
import lons.examples.ConcreteBinarySolution;
import utils.DataSetInstanceSplitter;
import weka.attributeSelection.GreedyStepwise;
import weka.core.Instances;

import static fs.FeatureSelectorUtils.convertAttributeIndexArrayToBinarySolutionFormat;
import static utils.GlobalConstants.PERCENTAGE_SPLIT;

public class SequentialForwardSelection extends SequentialSelection implements FeatureSelectionAlgorithm {
    @Override
    public FeatureSelectionResult apply(Instances data, FitnessEvaluator fitnessEvaluator) throws Exception {
        GreedyStepwise greedyStepwise = new GreedyStepwise();
        greedyStepwise.setOptions(weka.core.Utils.splitOptions("weka.attributeSelection.GreedyStepwise -N -1 -num-slots 1 "));

        DataSetInstanceSplitter splitter = new DataSetInstanceSplitter(data, PERCENTAGE_SPLIT);
        return seqSelect(greedyStepwise, fitnessEvaluator, data, splitter);
    }

    @Override
    public String getAlgorithmName() {
        return "Sequential Forward Selection";
    }
}
