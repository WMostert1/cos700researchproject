package fs;

import com.google.common.collect.Lists;
import fitness.FitnessEvaluator;
import lons.examples.BinarySolution;
import lons.examples.ConcreteBinarySolution;
import utils.DataSetInstanceSplitter;
import utils.GlobalConstants;
import weka.attributeSelection.GeneticSearch;
import weka.core.Instances;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.security.SecureRandom;
import java.util.List;

import static fs.FeatureSelectorUtils.convertAttributeIndexArrayToBinarySolutionFormat;
import static utils.GlobalConstants.PERCENTAGE_SPLIT;

public class GeneticSearchWrapperMethod implements FeatureSelectionAlgorithm {
    private final int GA_RUNS;
    private static SecureRandom random = new SecureRandom();

    public GeneticSearchWrapperMethod(int numberOfRuns){
        this.GA_RUNS = numberOfRuns;
    }

    public GeneticSearchWrapperMethod() {
        this(10);
    }

    @Override
    public FeatureSelectionResult apply(Instances data, FitnessEvaluator fitnessEvaluator) throws Exception {
            List<FeatureSelectionResult> bundles = Lists.newArrayList();
            FeatureSelectionResult bestBundle = null;

            DataSetInstanceSplitter splitter = new DataSetInstanceSplitter(data, PERCENTAGE_SPLIT);

            WrapperSplitSetsEvalutator wrapper = new WrapperSplitSetsEvalutator(fitnessEvaluator);
            wrapper.buildEvaluator(data);

            int numberOfAttributes = data.numAttributes()-1;


            for(int i = 0; i < GA_RUNS; i++) {

                GeneticSearch geneticSearch = new GeneticSearch();
                geneticSearch.setSeed(random.nextInt());
                int[] subAttributes = geneticSearch.search(wrapper, splitter.getTrainingSet());
                ConcreteBinarySolution subSolution = (ConcreteBinarySolution) ConcreteBinarySolution.constructBinarySolution(convertAttributeIndexArrayToBinarySolutionFormat(subAttributes, numberOfAttributes));
                Double accuracy = fitnessEvaluator.getQuality(subSolution.getDesignVector(), data);

                if (accuracy == null) {
                    throw new RuntimeException("Unknown solution!");
                }

                FeatureSelectionResult bundle = new FeatureSelectionResult(FeatureSelectorUtils.getInstancesFromAttributeInclusionIndicesArr(splitter.getTestingSet(), subAttributes),
                        accuracy,
                        subSolution);
                bundles.add(bundle);
                if (bestBundle == null) {
                    bestBundle = bundle;
                }

                if (BigDecimal.valueOf(bestBundle.getAccuracy()).compareTo(BigDecimal.valueOf(bundle.getAccuracy())) < 0) {
                    bestBundle = bundle;
                }
            }

            double meanAccuracy = 0.0;
            for(FeatureSelectionResult bundle : bundles){
                meanAccuracy += bundle.getAccuracy();
            }

            meanAccuracy /= GA_RUNS;

            BigDecimal bdMeanAccuracy = new BigDecimal(meanAccuracy);
            bdMeanAccuracy = bdMeanAccuracy.setScale(GlobalConstants.DECIMAL_PLACES, RoundingMode.HALF_UP);

//            BigDecimal bdSuccessRatio = new BigDecimal(successCount/(double)GA_RUNS);
//            bdSuccessRatio = bdSuccessRatio.setScale(GlobalConstants.DECIMAL_PLACES, RoundingMode.HALF_UP);
            bestBundle.setAccuracy(bdMeanAccuracy.doubleValue());

            return bestBundle;
            //return new GAResult(bestBundle, bdMeanAccuracy.doubleValue(), bdSuccessRatio.doubleValue());

            //return null;
    }

    @Override
    public String getAlgorithmName() {
        return "Genetic Search Wrapper Method";
    }
}
