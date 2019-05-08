import classifiers.IBkClassifier;
import classifiers.IClassify;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import fitness.FitnessEvaluator;
import fs.*;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import utils.MathUtils;
import utils.OutputFormatter;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

import java.io.*;
import java.math.BigDecimal;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;


/**
 * Created by bbdnet1339 on 2016/08/05.
 *
 */
public class JournalApplication {
    public static final int DECIMAL_PLACES = 8;
    private static void sortIntArr(int [] arr){
        for(int i = 0; i < arr.length; i++){
            for(int j = 0; j < arr.length; j++){
                if(i == j)
                    continue;

                if(arr[j] > arr[i]){
                    int temp = arr[i];
                    arr[i] = arr[j];
                    arr[j] = temp;
                }
            }
        }
    }

    private List<String> getResourceFiles( String path ) throws IOException {
        List<String> filenames = new ArrayList<>();

        try(
                InputStream in = getResourceAsStream( path );
                BufferedReader br = new BufferedReader( new InputStreamReader( in ) ) ) {
            String resource;

            while( (resource = br.readLine()) != null ) {
                filenames.add( resource );
            }
        }

        return filenames;
    }

    private InputStream getResourceAsStream( String resource ) {
        final InputStream in
                = getContextClassLoader().getResourceAsStream( resource );

        return in == null ? getClass().getResourceAsStream( resource ) : in;
    }

    private ClassLoader getContextClassLoader() {
        return Thread.currentThread().getContextClassLoader();
    }



    private static Instances getDataSet(String name) throws Exception {
        DataSource source = new DataSource(EvoCOPPaperApplication.class.getResourceAsStream("data-sets/used/" + name));

        Instances originalData = source.getDataSet();

        originalData.setClassIndex(originalData.numAttributes()-1);

        return originalData;
    }

    private static String intArrToStr(int [] arr){
        sortIntArr(arr);
        StringBuilder ret = new StringBuilder("[ ");
        for(int i : arr){
            ret.append(i).append(" ");
        }
        return ret+"]";
    }

    private static final String VOWEL = "vowel";
    private static final String BREAST_W = "breast-w";
    private static final String ZOO = "zoo";

    public static void main(String [] args) throws Exception {
        List<String> dataSets = new ArrayList<>();
        boolean moveToErrorFolder = false;

        File folder = new File("./src/main/resources/data-sets/used");
        File[] listOfFiles = folder.listFiles();
        int numberOfDataSets = Integer.MAX_VALUE;
        for (int i = 0; i < listOfFiles.length && i < numberOfDataSets; i++) {
            if (listOfFiles[i].isFile()) {
                    dataSets.add(listOfFiles[i].getName());
            }
        }


        List<String> badSets = new ArrayList<>();

        System.out.println("Using " + dataSets.size() + " data sets");
        for (String dataSetName : dataSets) {
            System.out.println(dataSetName);
        }

        OutputFormatter outF = new OutputFormatter("out/relative-fs.csv");
        outF.addAsColumns("Dataset", "Feature Selection Algorithm", "Baseline", "Relative Fitness");

        for (String dataSet : dataSets) {

                try {
                    Instances originalData = getDataSet(dataSet);

                    System.out.println("------------  " + dataSet + "  ------------");
                    System.out.println("------------- " + originalData.numAttributes() + " Features --------------");
                    saveDataSetInfo(new DataSetInfo(dataSet, originalData.numAttributes() - 1, originalData.numInstances(), originalData.numClasses()));

                    IClassify classifier = new IBkClassifier();
                    FitnessEvaluator fitnessEvaluator = new FitnessEvaluator(classifier, originalData.numAttributes() - 1, 0.0);
//                    Map<ConcreteBinarySolution, Double> fitnessMap = filterEval.eval(originalData);


                    //Get the baseline fitness value
                    boolean[] allFeaturesDesign = new boolean[originalData.numAttributes() - 1];
                    for (int i = 0; i < allFeaturesDesign.length; i++) {
                        allFeaturesDesign[i] = true;
                    }


                    //Get solutions based on different feature selection algorithms
                    DescriptiveStatistics descriptiveStatistics = new DescriptiveStatistics();

                    //ConcreteBinarySolution allFeaturesSelectedSolution = (ConcreteBinarySolution) ConcreteBinarySolution.constructBinarySolution(allFeaturesDesign);
                    BigDecimal baselineFitness = MathUtils.doubleToBigDecimal(fitnessEvaluator.getQuality(allFeaturesDesign, originalData));

                    List<FeatureSelectionAlgorithm> algorithms = Lists.newArrayList();
                    algorithms.add(new RankerInformationGainMethod());
                    algorithms.add(new SequentialForwardSelection());
                    algorithms.add(new RankerPearsonCorrelationMethod());
                    algorithms.add(new CorrelationbasedFeatureSubsetMethod());
                    algorithms.add(new SequentialBackwardSelection());
                    algorithms.add(new GeneticSearchWrapperMethod());


                    for(FeatureSelectionAlgorithm fsa : algorithms){
                        FeatureSelectionResult fsResults = fsa.apply(originalData, fitnessEvaluator);
                        BigDecimal fitness = MathUtils.doubleToBigDecimal(fsResults.getAccuracy()).subtract(baselineFitness).divide(BigDecimal.valueOf(2.0), MathUtils.ROUNDING_MODE);
                        descriptiveStatistics.addValue(fitness.doubleValue());
                        outF.addAsColumns(dataSet, fsa.getAlgorithmName(), Double.toString(baselineFitness.doubleValue()), Double.toString(fitness.doubleValue()));
                    }

//                    FeatureSelectionAlgorithm rankerInfoGain = new RankerInformationGainMethod();
//                    FeatureSelectionResult rankerInfoGainMethodResults = rankerInfoGain.apply(originalData, fitnessEvaluator);
//                    double infoGainFitness = rankerInfoGainMethodResults.getAccuracy() - baselineFitness;
//                    descriptiveStatistics.addValue(infoGainFitness);
//                    outF.addAsColumns(dataSet, rankerInfoGain.getAlgorithmName(), Double.toString(baselineFitness), Double.toString(infoGainFitness));
//
////                    GeneticSearchWrapperMethod geneticSearchWrapperMethod = new GeneticSearchWrapperMethod();
////                    FeatureSelectionResult geneticSearchWrapperResults = geneticSearchWrapperMethod.apply(originalData, fitnessEvaluator);
////                    double geneticSearchFitness = geneticSearchWrapperResults.getAccuracy() - baselineFitness;
////                    descriptiveStatistics.addValue(geneticSearchFitness);
////                    outF.addAsColumns(dataSet, geneticSearchWrapperMethod.getAlgorithmName(), Double.toString(baselineFitness), Double.toString(geneticSearchFitness));
//
//                    SequentialForwardSelection sequentialForwardSelection = new SequentialForwardSelection();
//                    FeatureSelectionResult greedyStepwiseWrapperResult = sequentialForwardSelection.apply(originalData, fitnessEvaluator);
//                    double greedyStepwiseFitness = greedyStepwiseWrapperResult.getAccuracy() - baselineFitness;
//                    descriptiveStatistics.addValue(greedyStepwiseFitness);
//                    outF.addAsColumns(dataSet, sequentialForwardSelection.getAlgorithmName(), Double.toString(baselineFitness), Double.toString(greedyStepwiseFitness));
//
////                    RankerPrincipalComponentsMethod rankerPrincipalComponentsMethod = new RankerPrincipalComponentsMethod();
////                    FeatureSelectionResult principalComponentsResult = rankerPrincipalComponentsMethod.apply(originalData, fitnessEvaluator);
////                    double pcaFitness = principalComponentsResult.getAccuracy() - baselineFitness;
////                    descriptiveStatistics.addValue(pcaFitness);
////                    outF.addAsColumns(dataSet, rankerPrincipalComponentsMethod.getAlgorithmName(), Double.toString(baselineFitness), Double.toString(pcaFitness));
//
//                    RankerPearsonCorrelationMethod rankerPearsonCorrelationMethod = new RankerPearsonCorrelationMethod();
//                    FeatureSelectionResult correlationPearsonResult = rankerPearsonCorrelationMethod.apply(originalData, fitnessEvaluator);
//                    double correlationPearsonFitness = correlationPearsonResult.getAccuracy() - baselineFitness;
//                    descriptiveStatistics.addValue(correlationPearsonFitness);
//                    outF.addAsColumns(dataSet, rankerPearsonCorrelationMethod.getAlgorithmName(), Double.toString(baselineFitness), Double.toString(correlationPearsonFitness));
//
//                    CorrelationbasedFeatureSubsetMethod correlationbasedFeatureSubsetMethod = new CorrelationbasedFeatureSubsetMethod();
//                    FeatureSelectionResult intercorrelationResult = correlationbasedFeatureSubsetMethod.apply(originalData, fitnessEvaluator);
//                    double correlationSubsetFitness = intercorrelationResult.getAccuracy() - baselineFitness;
//                    descriptiveStatistics.addValue(correlationSubsetFitness);
//                    outF.addAsColumns(dataSet, correlationbasedFeatureSubsetMethod.getAlgorithmName(), Double.toString(baselineFitness), Double.toString(correlationSubsetFitness));

                    outF.addEmptyRow();
                    outF.addAsColumns("STD DEVIATION", Double.toString(descriptiveStatistics.getStandardDeviation()));
                    outF.addAsColumns("Number of features", Integer.toString(originalData.numAttributes()-1));
                    outF.addEmptyRow();
                    outF.addEmptyRow();
                    /*
                        baseline : 0.5
                        fitness: -0.3
                        relative = fitness - baseline
                     */

                    outF.save();
                } catch (Exception e) {
                    if(moveToErrorFolder) {
                        Files.move(Paths.get("./src/main/resources/data-sets/used/" + dataSet), Paths.get("./src/main/resources/data-sets/error/" + dataSet));
                    }

                    System.err.println(e.getMessage());
                    System.err.println("Skipping " + dataSet + " and moved to error folder.");
                    badSets.add(dataSet);
                }
        }



        for (String s : badSets) {
            System.err.println(s);
        }
    }

    private static int dataSetNumber = 0;


    private static void buildDataSetInfo(OutputFormatter outF, String name, int noAttrs, int noInstances, int noClasses){
        if(dataSetNumber == 0){
            outF.addAsColumns("Datasets");
            outF.addAsColumns("Index", "Name", "No. Attributes", "No. Instances", "No. Classes");
        }

        outF.addAsColumns(Integer.toString(dataSetNumber++), name, Integer.toString(noAttrs),
                Integer.toString(noInstances), Integer.toString(noClasses));
    }

    public static class DataSetInfo{
        private String name;
        int noAttrs,  noInstances,  noClasses;

        public DataSetInfo(String name, int noAttrs, int noInstances, int noClasses) {
            this.name = name;
            this.noAttrs = noAttrs;
            this.noInstances = noInstances;
            this.noClasses = noClasses;
        }

        public String getName() {
            return name;
        }

        public int getNoAttrs() {
            return noAttrs;
        }

        public int getNoInstances() {
            return noInstances;
        }

        public int getNoClasses() {
            return noClasses;
        }
    }
    private static List<DataSetInfo> dataSetReport = Lists.newArrayList();
    private static void saveDataSetInfo(DataSetInfo info){
        dataSetReport.add(info);
    }
    private static void flushDataSetInfo(OutputFormatter outF){
        for(DataSetInfo dataSetInfo : dataSetReport){
            buildDataSetInfo(outF, dataSetInfo.getName(), dataSetInfo.getNoAttrs(), dataSetInfo.getNoInstances(), dataSetInfo.getNoClasses());
        }
    }

    private static List<String> algorithmNames = Lists.newArrayList();
    private static void buildFSInfo(OutputFormatter outF, String algorithmName, String dataSetName, int numberOfGlobalOptima,
                                    boolean foundLocalOpt, boolean foundGlobalOpt, double percentageCap, double successRatio, double fsAccuracy, double globalAccuracy){
        if(!algorithmNames.contains(algorithmName)){
            algorithmNames.add(algorithmName);
            outF.addEmptyRow();
            outF.addAsColumns(algorithmName);
            outF.addAsColumns("Dataset name", "No. Global Optima", "Found Local Optima", "Found Global Optima", "Percentage Cap", "Success Ratio", "Feat. S. Fitness", "Global Opt. Fitness");
        }

        outF.addAsColumns(dataSetName, Integer.toString(numberOfGlobalOptima), Boolean.toString(foundLocalOpt),
                Boolean.toString(foundGlobalOpt), Double.toString(percentageCap), Double.toString(successRatio), Double.toString(fsAccuracy), Double.toString(globalAccuracy));

    }

    private static class FSReportInfo{
        private String algorithmName;
        private String dataSetName;
        private int numberOfGlobalOptima;
        private boolean foundLocalOpt;
        private boolean foundGlobalOpt;
        private double percentageCap;
        private double fsAccuracy;
        private double globalAccuracy;
        private double successRatio;

        public FSReportInfo(String algorithmName, String dataSetName, int numberOfGlobalOptima, boolean foundLocalOpt, boolean foundGlobalOpt, double percentageCap, double successRatio, double fsAccuracy, double globalAccuracy) {
            this.algorithmName = algorithmName;
            this.dataSetName = dataSetName;
            this.numberOfGlobalOptima = numberOfGlobalOptima;
            this.foundLocalOpt = foundLocalOpt;
            this.foundGlobalOpt = foundGlobalOpt;
            this.percentageCap = percentageCap;
            this.fsAccuracy = fsAccuracy;
            this.globalAccuracy = globalAccuracy;
            this.successRatio = successRatio;
        }

        public double getSuccessRatio() {
            return successRatio;
        }

        public String getAlgorithmName() {
            return algorithmName;
        }

        public String getDataSetName() {
            return dataSetName;
        }

        public int getNumberOfGlobalOptima() {
            return numberOfGlobalOptima;
        }

        public boolean isFoundLocalOpt() {
            return foundLocalOpt;
        }

        public boolean isFoundGlobalOpt() {
            return foundGlobalOpt;
        }

        public double getPercentageCap() {
            return percentageCap;
        }

        public double getFsAccuracy() {
            return fsAccuracy;
        }

        public double getGlobalAccuracy() {
            return globalAccuracy;
        }
    }

    private static HashMap<String, HashMap<String, FSReportInfo>> fsReportInfo = Maps.newHashMap();

    private static void saveFSInfo(FSReportInfo reportInfo){
        HashMap<String, FSReportInfo> algorithmReport = fsReportInfo.get(reportInfo.algorithmName);
        if(algorithmReport == null){
            algorithmReport = Maps.newHashMap();
        }

        algorithmReport.put(reportInfo.getDataSetName(), reportInfo);
        fsReportInfo.put(reportInfo.getAlgorithmName(), algorithmReport);
    }

    private static void flushFSInfo(OutputFormatter outF){
        for(String algorithmName : fsReportInfo.keySet()){
            for(String dataSetName : fsReportInfo.get(algorithmName).keySet()) {
                FSReportInfo report = fsReportInfo.get(algorithmName).get(dataSetName);
                buildFSInfo(outF, algorithmName, dataSetName, report.getNumberOfGlobalOptima(), report.isFoundLocalOpt(),
                        report.isFoundGlobalOpt(), report.getPercentageCap(), report.getSuccessRatio(), report.getFsAccuracy(), report.getGlobalAccuracy());
            }
        }
    }
}
