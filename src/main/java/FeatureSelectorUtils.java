import weka.attributeSelection.*;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

import java.util.*;

/**
 * Created by bbdnet1339 on 2016/08/08.
 */
public class FeatureSelectorUtils {
    private static double percentageSplit = 50.0;

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

    private static Instances getDataSet(ConverterUtils.DataSource dataSource) throws Exception {
        Instances data = dataSource.getDataSet();

        if(data.classIndex() == -1){
            data.setClassIndex(data.numAttributes()-1);
        }

        return data;
    }

    private static int [] getIndiceArrayFromInfoArr(double [][] infoArr, int index){
        int [] ret = new int[index+1];
        for(int i = 0; i <= index; i++){
            ret[i] = (int)infoArr[i][0];
        }
        return ret;
    }

    public static FSBundle getFilterMethodAttributes(ConverterUtils.DataSource dataSource) throws Exception {

        Instances data = getDataSet(dataSource);
        DataSetInstanceSplitter splitter = new DataSetInstanceSplitter(data, percentageSplit);
        data = splitter.getTrainingSet();
        InfoGainAttributeEval filterMethod = new InfoGainAttributeEval();
        filterMethod.setMissingMerge(true);
        filterMethod.buildEvaluator(data);
        Ranker ranker = new Ranker();

        ranker.search(filterMethod, data);
        double [][] ranked = ranker.rankedAttributes();

        Map<Integer, Double> performancePerNumberOfFeaturesIncluded = new HashMap<>();


        DataSetInstanceSplitter subSplitter = new DataSetInstanceSplitter(splitter.getTestingSet(), 50.0);
        IBkClassifier classifier = new IBkClassifier();
        for(int i = 0; i < ranked.length; i++){

            int [] attributes = getIndiceArrayFromInfoArr(ranked, i);
            Instances subTraining = subSplitter.getTrainingSet();
            subTraining = FeatureSelectorUtils.getInstancesFromAttributeInclusionIndicesArr(subTraining, attributes);
            Instances subTesting = subSplitter.getTestingSet();
            subTesting = FeatureSelectorUtils.getInstancesFromAttributeInclusionIndicesArr(subTesting, attributes);

            double accuracy = classifier.getClassificationAccuracy(subTraining, subTesting);
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


        return new FSBundle(FeatureSelectorUtils.getInstancesFromAttributeInclusionIndicesArr(splitter.getTestingSet(), attributes),
                                            bestAccuracy, attributes);
    }

    public static FSBundle getWrapperMethodAttributes(ConverterUtils.DataSource dataSource) throws Exception {
        Instances data = getDataSet(dataSource);
        DataSetInstanceSplitter splitter = new DataSetInstanceSplitter(data, percentageSplit);
        data = splitter.getTrainingSet();
        WrapperSubsetEval wrapper = new WrapperSubsetEval();
        wrapper.setOptions(weka.core.Utils.splitOptions("weka.attributeSelection.WrapperSubsetEval -B weka.classifiers.lazy.IBk -F 5 -T 0.01 -R 1 -E DEFAULT -- -K 3 -W 0 -A \"weka.core.neighboursearch.LinearNNSearch -A \\\"weka.core.EuclideanDistance -R first-last\\\"\""));
        wrapper.buildEvaluator(data);

        BestFirst bestFirst = new BestFirst();
        bestFirst.setOptions(weka.core.Utils.splitOptions("-K 3 -W 0 -A weka.core.neighboursearch.LinearNNSearch -A \"weka.core.EuclideanDistance -R first-last\""));


        int [] attributes = bestFirst.search(wrapper, data);
        DataSetInstanceSplitter subSplitter = new DataSetInstanceSplitter(splitter.getTestingSet(), 50.0);
        IBkClassifier classifier = new IBkClassifier();
        Instances subTraining = subSplitter.getTrainingSet();
        subTraining = FeatureSelectorUtils.getInstancesFromAttributeInclusionIndicesArr(subTraining, attributes);
        Instances subTesting = subSplitter.getTestingSet();
        subTesting = FeatureSelectorUtils.getInstancesFromAttributeInclusionIndicesArr(subTesting, attributes);

        double accuracy = classifier.getClassificationAccuracy(subTraining, subTesting);

        return new FSBundle(FeatureSelectorUtils.getInstancesFromAttributeInclusionIndicesArr(splitter.getTestingSet(), attributes),accuracy, attributes);
    }

    public static FSBundle getEmbeddedMethodAttributes(ConverterUtils.DataSource dataSource) throws Exception{
        Instances data = getDataSet(dataSource);
        DataSetInstanceSplitter splitter = new DataSetInstanceSplitter(data, percentageSplit);
        data = splitter.getTrainingSet();
        WrapperSubsetEval wrapper = new WrapperSubsetEval();
        wrapper.setOptions(weka.core.Utils.splitOptions("weka.attributeSelection.WrapperSubsetEval -B weka.classifiers.lazy.IBk -F 5 -T 0.01 -R 1 -E DEFAULT -- -K 3 -W 0 -A \"weka.core.neighboursearch.LinearNNSearch -A \\\"weka.core.EuclideanDistance -R first-last\\\"\""));
        wrapper.buildEvaluator(data);

        GeneticSearch geneticSearch = new GeneticSearch();

        int [] attributes = geneticSearch.search(wrapper, data);
        DataSetInstanceSplitter subSplitter = new DataSetInstanceSplitter(splitter.getTestingSet(), 50.0);
        IBkClassifier classifier = new IBkClassifier();
        Instances subTraining = subSplitter.getTrainingSet();
        subTraining = FeatureSelectorUtils.getInstancesFromAttributeInclusionIndicesArr(subTraining, attributes);
        Instances subTesting = subSplitter.getTestingSet();
        subTesting = FeatureSelectorUtils.getInstancesFromAttributeInclusionIndicesArr(subTesting, attributes);

        double accuracy = classifier.getClassificationAccuracy(subTraining, subTesting);

        return new FSBundle(FeatureSelectorUtils.getInstancesFromAttributeInclusionIndicesArr(splitter.getTestingSet(), attributes),accuracy, attributes);
    }


}
