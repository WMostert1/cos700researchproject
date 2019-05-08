package fs;

import fitness.FitnessEvaluator;
import lons.examples.ConcreteBinarySolution;
import utils.DataSetInstanceSplitter;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.core.Instances;

import java.util.HashMap;
import java.util.Map;

import static fs.FeatureSelectorUtils.convertAttributeIndexArrayToBinarySolutionFormat;

import static fs.FeatureSelectorUtils.getIndiceArrayFromInfoArr;
import static utils.GlobalConstants.PERCENTAGE_SPLIT;

public class RankerInformationGainMethod extends GenericRankerFeatureSelection implements FeatureSelectionAlgorithm {
    @Override
    public FeatureSelectionResult apply(Instances data, FitnessEvaluator fitnessEvaluator) throws Exception {
            DataSetInstanceSplitter splitter = new DataSetInstanceSplitter(data, PERCENTAGE_SPLIT);
            InfoGainAttributeEval infoGainAttributeEval = new InfoGainAttributeEval();
            infoGainAttributeEval.setMissingMerge(true);
            infoGainAttributeEval.buildEvaluator(splitter.getTrainingSet());
            return rank(infoGainAttributeEval, fitnessEvaluator, splitter, data);
        }

    @Override
    public String getAlgorithmName() {
        return "Ranker Info Gain Filter Method";
    }
}
