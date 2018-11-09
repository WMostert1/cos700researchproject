import com.google.common.collect.Lists;
import lons.examples.BinarySolution;
import lons.examples.ConcreteBinarySolution;
import weka.attributeSelection.*;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.security.SecureRandom;
import java.util.*;

/**
 * Created by bbdnet1339 on 2016/08/08.
 */
public class FeatureSelectorUtils {
    private static SecureRandom random = new SecureRandom();
    private static final double percentageSplit = LandscapeEvaluator.trainingPercentage;
    private static final int GA_RUNS = 30;

    public static  String booleanArrayToBitString(boolean [] arr){
        String bitString = "";
        for(boolean b : arr)
            bitString += b ? "1" : "0";
        return bitString;
    }

    public static boolean [] bitStringToBooleanArray(String bitString){
        boolean [] boolArr = new boolean[bitString.length()];
        for(int i = 0; i < bitString.length();i++){
            boolArr[i] = bitString.charAt(i) != '0';
        }
        return boolArr;
    }

    public static boolean [] convertAttributeIndexArrayToBinarySolutionFormat(int [] featureIndexArray, int dimensions){
        boolean [] boolArray = new boolean [dimensions];
        ArrayList<Integer> attributes = new ArrayList<>();
        for(int i : featureIndexArray){
            attributes.add(i);
        }

        for(int i = 0; i < dimensions; i++){
            boolArray [i] = attributes.contains(i);
        }

        return boolArray;
    }

    //The result of this will be the indexes of the attributes to REMOVE
    public static int [] booleanArrayToWekaAttribSelectionArrayToRemove(boolean [] arr){
        List<Integer> indicesToInclude = new ArrayList<Integer>();
        for(int i = 0; i < arr.length; i++){
            if(!arr[i])
                indicesToInclude.add(i);
        }
        final int [] retArr = new int[indicesToInclude.size()];
        for(int i = 0; i < retArr.length;i++){
            retArr[i] = indicesToInclude.get(i);
        }
        return retArr;
    }


    public static Instances getInstancesFromAttributeInclusionIndicesArr(Instances data, int [] columnsToKeep) throws Exception{
        Remove remove = new Remove();

        remove.setInvertSelection(true);

        boolean classIncluded = false;
        int i = 0;
        while(!classIncluded && i < columnsToKeep.length){
            if(columnsToKeep[i] == data.numAttributes() -1)
                classIncluded = true;
            i++;
        }

        if(!classIncluded){
            int y = 0;
            int [] newColumnsToKeep = new int[columnsToKeep.length+1];
            for(; y < columnsToKeep.length; y++)
                newColumnsToKeep[y] = columnsToKeep[y];
            newColumnsToKeep[y] = data.numAttributes()-1;
            columnsToKeep = newColumnsToKeep;
        }

        if (columnsToKeep.length < 2)
            throw new RuntimeException("All attributes except for class will be deleted.");

        remove.setAttributeIndicesArray(columnsToKeep);
        remove.setInputFormat(data);
        Instances instNew = Filter.useFilter(data, remove);

        return instNew;

    }

    public static Instances getInstancesFromAttributeRemovalIndicesArr(Instances data, int [] columnsToRemove) throws Exception{
        Remove remove = new Remove();
        remove.setAttributeIndicesArray(columnsToRemove);

        for(int i =0 ; i < columnsToRemove.length; i++){
            if (columnsToRemove[i] == data.numAttributes() - 1){
                throw new RuntimeException("Can not delete class attribute!!!");
            }
        }

        data.setClassIndex(data.numAttributes() - 1);
        remove.setInputFormat(data);
        Instances instNew = Filter.useFilter(data, remove);
        return instNew;

    }
    //This transforms the bit string to weka instances with the selected features present in the attributes
    public static Instances getInstancesFromBitString(Instances data, boolean [] bitString) throws Exception {
       return getInstancesFromAttributeRemovalIndicesArr(data, booleanArrayToWekaAttribSelectionArrayToRemove(bitString));
    }

//    private static Instances getDataSet(ConverterUtils.DataSource dataSource) throws Exception {
//        Instances data = dataSource.getDataSet();
//
//        if(data.classIndex() == -1){
//            data.setClassIndex(data.numAttributes()-1);
//        }
//
//        return data;
//    }

    private static int [] getIndiceArrayFromInfoArr(double [][] infoArr, int index){
        int [] ret = new int[index+1];
        for(int i = 0; i <= index; i++){
            ret[i] = (int)infoArr[i][0];
        }
        return ret;
    }

    public static FSBundle getFilterMethodAttributes(Instances data, Map<ConcreteBinarySolution, Double> fitnessMap) throws Exception {
        int numberOfAttributes = data.numAttributes()-1;
        DataSetInstanceSplitter splitter = new DataSetInstanceSplitter(data, percentageSplit);

        InfoGainAttributeEval filterMethod = new InfoGainAttributeEval();
        filterMethod.setMissingMerge(true);
        filterMethod.buildEvaluator(splitter.getTrainingSet());
        Ranker ranker = new Ranker();

        ranker.search(filterMethod, splitter.getTrainingSet());
        double [][] ranked = ranker.rankedAttributes();

        Map<Integer, Double> performancePerNumberOfFeaturesIncluded = new HashMap<>();

        for(int i = 0; i < ranked.length; i++){

            int [] subAttributes = getIndiceArrayFromInfoArr(ranked, i);
            ConcreteBinarySolution subSolution = (ConcreteBinarySolution) ConcreteBinarySolution.constructBinarySolution(convertAttributeIndexArrayToBinarySolutionFormat(subAttributes, numberOfAttributes));


            Double accuracy = fitnessMap.get(subSolution);

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


        return new FSBundle(FeatureSelectorUtils.getInstancesFromAttributeInclusionIndicesArr(splitter.getTestingSet(), attributes),
                                            bestAccuracy, ConcreteBinarySolution.constructBinarySolution(convertAttributeIndexArrayToBinarySolutionFormat(attributes,numberOfAttributes)));
    }



//    public static FSBundle getWrapperMethodAttributes(Instances data) throws Exception {
////        Instances data = getDataSet(dataSource);
//        DataSetInstanceSplitter splitter = new DataSetInstanceSplitter(data, percentageSplit);
////        data = splitter.getTrainingSet();
//        WrapperSubsetEval wrapper = new WrapperSubsetEval();
//        wrapper.setOptions(weka.core.Utils.splitOptions("weka.attributeSelection.WrapperSubsetEval -B weka.classifiers.lazy.IBk -F 5 -T 0.01 -R 1 -E DEFAULT -- -K 3 -W 0 -A \"weka.core.neighboursearch.LinearNNSearch -A \\\"weka.core.EuclideanDistance -R first-last\\\"\""));
//        wrapper.buildEvaluator(splitter.getTrainingSet());
//
//        BestFirst bestFirst = new BestFirst();
//        bestFirst.setOptions(weka.core.Utils.splitOptions("-K 3 -W 0 -A weka.core.neighboursearch.LinearNNSearch -A \"weka.core.EuclideanDistance -R first-last\""));
//
//
//        int [] attributes = bestFirst.search(wrapper, data);
////        DataSetInstanceSplitter subSplitter = new DataSetInstanceSplitter(splitter.getTestingSet(), 50.0);
//        IBkClassifier classifier = new IBkClassifier();
//        Instances subTraining =  FeatureSelectorUtils.getInstancesFromAttributeInclusionIndicesArr(splitter.getTrainingSet(), attributes);
//        Instances subTesting =  FeatureSelectorUtils.getInstancesFromAttributeInclusionIndicesArr(splitter.getTestingSet(), attributes);
//
//        double accuracy = classifier.getClassificationAccuracy(subTraining, subTesting);
//
//        return new FSBundle(FeatureSelectorUtils.getInstancesFromAttributeInclusionIndicesArr(splitter.getTestingSet(), attributes),accuracy, attributes);
//    }


    public static GAResult getWrapperGAAttributes(Instances data, List<BinarySolution> globalOptimas, Map<ConcreteBinarySolution, Double> fitnessMap) throws Exception{
        System.out.println("Starting the GA Wrapper method.");
        List<FSBundle> bundles = Lists.newArrayList();
        FSBundle bestBundle = null;

        DataSetInstanceSplitter splitter = new DataSetInstanceSplitter(data, percentageSplit);

        WrapperSplitSetsEvalutator wrapper = new WrapperSplitSetsEvalutator(fitnessMap);
        wrapper.buildEvaluator(data);

        int numberOfAttributes = data.numAttributes()-1;

        for(int i = 0; i < GA_RUNS; i++) {
            GeneticSearch geneticSearch = new GeneticSearch();
            geneticSearch.setSeed(random.nextInt());
            int[] subAttributes = geneticSearch.search(wrapper, splitter.getTrainingSet());
            ConcreteBinarySolution subSolution = (ConcreteBinarySolution) ConcreteBinarySolution.constructBinarySolution(convertAttributeIndexArrayToBinarySolutionFormat(subAttributes, numberOfAttributes));

            Double accuracy = fitnessMap.get(subSolution);

            if(accuracy == null){
                throw new RuntimeException("Unknown solution!");
            }

            System.out.println(MainApplication.calculatePercentageCap(accuracy, fitnessMap.get(globalOptimas.get(0))));
            FSBundle bundle = new FSBundle(FeatureSelectorUtils.getInstancesFromAttributeInclusionIndicesArr(splitter.getTestingSet(), subAttributes),
                    accuracy,
                    subSolution);
            bundles.add(bundle);
            if(bestBundle == null){
                bestBundle = bundle;
            }

            if(BigDecimal.valueOf(bestBundle.getAccuracy()).compareTo(BigDecimal.valueOf(bundle.getAccuracy())) < 0 ){
                bestBundle = bundle;
            }
        }

        double meanAccuracy = 0.0;
        int successCount = 0;
        for(FSBundle bundle : bundles){
            meanAccuracy += bundle.getAccuracy();
            if(globalOptimas.contains(bundle.getBinarySolution())){
                successCount++;
            }
        }

        meanAccuracy /= GA_RUNS;

        BigDecimal bdMeanAccuracy = new BigDecimal(meanAccuracy);
        bdMeanAccuracy = bdMeanAccuracy.setScale(MainApplication.DECIMAL_PLACES, RoundingMode.HALF_UP);

        BigDecimal bdSuccessRatio = new BigDecimal(successCount/(double)GA_RUNS);
        bdSuccessRatio = bdSuccessRatio.setScale(MainApplication.DECIMAL_PLACES, RoundingMode.HALF_UP);


        return new GAResult(bestBundle, bdMeanAccuracy.doubleValue(), bdSuccessRatio.doubleValue());
    }

    public static FSBundle getWrapperGreedyStepwiseAttributes(Instances data, Map<ConcreteBinarySolution, Double> fitnessMap) throws Exception{
        DataSetInstanceSplitter splitter = new DataSetInstanceSplitter(data, percentageSplit);

        WrapperSplitSetsEvalutator wrapper = new WrapperSplitSetsEvalutator(fitnessMap);
        wrapper.buildEvaluator(data);

        int numberOfAttributes = data.numAttributes()-1;

        GreedyStepwise greedyStepwise = new GreedyStepwise();
        greedyStepwise.setOptions(weka.core.Utils.splitOptions("weka.attributeSelection.GreedyStepwise -N -1 -num-slots 1 "));

        int [] subAttributes = greedyStepwise.search(wrapper, splitter.getTrainingSet());

        ConcreteBinarySolution subSolution = (ConcreteBinarySolution) ConcreteBinarySolution.constructBinarySolution(convertAttributeIndexArrayToBinarySolutionFormat(subAttributes, numberOfAttributes));

        Double accuracy = fitnessMap.get(subSolution);

        if(accuracy == null){
            throw new RuntimeException("Unknown solution!");
        }

        return new FSBundle(FeatureSelectorUtils.getInstancesFromAttributeInclusionIndicesArr(splitter.getTestingSet(), subAttributes), accuracy, subSolution);

    }


}
