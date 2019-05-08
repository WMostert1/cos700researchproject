package fs;

import fitness.FitnessEvaluator;
import lons.examples.ConcreteBinarySolution;
import utils.DataSetInstanceSplitter;
import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.Ranker;
import weka.core.Instances;

import java.util.HashMap;
import java.util.Map;

import static fs.FeatureSelectorUtils.convertAttributeIndexArrayToBinarySolutionFormat;
import static fs.FeatureSelectorUtils.getIndiceArrayFromInfoArr;

public abstract class GenericRankerFeatureSelection {
    public FeatureSelectionResult rank(ASEvaluation evaluation, FitnessEvaluator fitnessEvaluator, DataSetInstanceSplitter splitter, Instances originalData) throws Exception{
        Ranker ranker = new Ranker();
        int numberOfAttributes = originalData.numAttributes()-1;
        ranker.search(evaluation, splitter.getTrainingSet());
        double [][] ranked = ranker.rankedAttributes();

        Map<Integer, Double> performancePerNumberOfFeaturesIncluded = new HashMap<>();

        for(int i = 0; i < ranked.length; i++){

            int [] subAttributes = getIndiceArrayFromInfoArr(ranked, i);
            boolean [] solution =  convertAttributeIndexArrayToBinarySolutionFormat(subAttributes, numberOfAttributes);


            Double accuracy = fitnessEvaluator.getQuality(solution, originalData);

            if(accuracy == null){
                throw new RuntimeException("Unknown solution!");
            }
            performancePerNumberOfFeaturesIncluded.put(i, accuracy);
        }

        int bestFeatureIndex = 0;
        double bestAccuracy = performancePerNumberOfFeaturesIncluded.get(0);
        for( int i = 1; i < ranked.length; i++){
            if(performancePerNumberOfFeaturesIncluded.get(i) > bestAccuracy) {
                bestAccuracy = performancePerNumberOfFeaturesIncluded.get(i);
                bestFeatureIndex = i;
            }
        }

        int [] attributes = new int[bestFeatureIndex+1];
        for(int i = 0; i <= bestFeatureIndex; i++){
            attributes[i] = (int)ranked[i][0];
        }

        if(attributes.length == 1 && attributes[0] == 0){
            throw new RuntimeException("Can not choose NO features.");
        }


        return new FeatureSelectionResult(FeatureSelectorUtils.getInstancesFromAttributeInclusionIndicesArr(splitter.getTestingSet(), attributes),
                bestAccuracy, ConcreteBinarySolution.constructBinarySolution(convertAttributeIndexArrayToBinarySolutionFormat(attributes,numberOfAttributes)));
    }
}
