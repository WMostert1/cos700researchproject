import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import lons.EdgeType;
import lons.LONGenerator;
import lons.RVisualizationFormatter;
import lons.Weight;
import lons.examples.*;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

import java.io.*;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.*;


/**
 * Created by bbdnet1339 on 2016/08/05.
 *
 */
public class GECCO2019Application {
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

        File folder = new File("./src/main/resources/data-sets/used");
        File[] listOfFiles = folder.listFiles();

        final int maxFeatures = 15;
        final int minFeatures = 7;
        final int minInstances = 500;
        final int maxInstances = 3000;

        for (int i = 0; i < listOfFiles.length; i++) {
            if (listOfFiles[i].isFile()) {
                Instances data = getDataSet(listOfFiles[i].getName());
                if (data.numAttributes() <= maxFeatures &&
                        data.numAttributes() >= minFeatures &&
                        data.numInstances() >= minInstances &&
                        data.numInstances() <= maxInstances) {
                    dataSets.add(listOfFiles[i].getName());
                }
            }
        }
        List<String> badSets = new ArrayList<>();

        System.out.println("Found " + dataSets.size() + " data sets with less than " + maxFeatures);
        for (String dataSetName : dataSets) {
            System.out.println(dataSetName);
        }
        OutputFormatter outF = new OutputFormatter("out/aboveBaselineSummaryNeuralNet.csv");
        outF.addAsColumns("Data Set", "Classifier", "Number of Features", "Baseline Fitness", "Total Solutions", "# Solutions Above Baseline", "Above baseline ratio", "Above Baseline MIN", "Above Baseline Max", "Above Baseline STDEV", "Above Baseline Skewness");
        Map<String, IClassify> classifiersMap = Maps.newHashMap();
//        classifiersMap.put("knn-3", new IBkClassifier());
//        classifiersMap.put("J48", new J48Classifier());
        classifiersMap.put("ANN", new NeuralNetClassifier());


        for (String dataSet : dataSets) {
            for (String classifierName : classifiersMap.keySet()) {

                try {
                    Instances originalData = getDataSet(dataSet);

                    System.out.println("------------  " + dataSet + "  ------------");
                    System.out.println("------------- " + originalData.numAttributes() + " Features --------------");
                    System.out.println("------------- Classifier : " + classifierName + " --------------");
                    saveDataSetInfo(new DataSetInfo(dataSet, originalData.numAttributes() - 1, originalData.numInstances(), originalData.numClasses()));

                    IClassify classifier = classifiersMap.get(classifierName);
                    LandscapeEvaluator filterEval = new LandscapeEvaluator(classifier);
                    Map<ConcreteBinarySolution, Double> fitnessMap = filterEval.eval(originalData);

                    boolean[] allFeaturesDesign = new boolean[originalData.numAttributes() - 1];
                    for (int i = 0; i < allFeaturesDesign.length; i++) {
                        allFeaturesDesign[i] = true;
                    }

                    ConcreteBinarySolution allFeaturesSelectedSolution = (ConcreteBinarySolution) ConcreteBinarySolution.constructBinarySolution(allFeaturesDesign);
                    Double baselineFitness = fitnessMap.get(allFeaturesSelectedSolution);
                    if(baselineFitness == null){
                        baselineFitness = filterEval.getQuality(allFeaturesDesign, originalData);
                    }

                    Map<ConcreteBinarySolution, Double> aboveBaselineSolutions = new HashMap<>();
                    for (ConcreteBinarySolution solution : fitnessMap.keySet()) {
                        Double solutionFitness = fitnessMap.get(solution);
                        if (solutionFitness.compareTo(baselineFitness) > 0) {
                            aboveBaselineSolutions.put(solution, solutionFitness);
                        }
                    }

                    System.out.println("Classifier : "+ classifierName);
                    System.out.println("Baseline fitness : " + baselineFitness);
                    System.out.println("Total number of solutions : " + fitnessMap.size());
                    System.out.println("Total number of solutions above baseline : " + aboveBaselineSolutions.size());
                    BigDecimal bd = new BigDecimal(aboveBaselineSolutions.size() * 1.0 / fitnessMap.size());
                    bd = bd.setScale(EvoCOPPaperApplication.DECIMAL_PLACES, RoundingMode.HALF_UP);
                    System.out.println("Above baseline ratio : " + bd.toString());

                    DescriptiveStatistics descriptiveStatistics = new DescriptiveStatistics();
                    for(Double fitness : aboveBaselineSolutions.values()){
                        descriptiveStatistics.addValue(fitness);
                    }



//                outF.addAsColumns("Data Set", "Baseline Fitness", "Total Solutions", "# Solutions Above Baseline", "Above baseline ratio");
                    outF.addAsColumns(dataSet,
                            classifierName,
                            Integer.toString(originalData.numAttributes() - 1), //number of features
                            baselineFitness.toString(), //baseline fitness
                            Integer.toString(fitnessMap.size()), //# of solutions
                            Integer.toString(aboveBaselineSolutions.size()), //# of solutions above baseline
                            bd.toString(),//ratio of solutions above baseline to no of solutions
                            Double.toString(descriptiveStatistics.getMin()),
                            Double.toString(descriptiveStatistics.getMax()),
                            Double.toString(descriptiveStatistics.getStandardDeviation()),
                            Double.toString(descriptiveStatistics.getSkewness()));



                    OutputFormatter aboveBaslineValues = new OutputFormatter("out/ab_" + dataSet + "_"+classifierName+".csv");
                    for (Double d : aboveBaselineSolutions.values()) {
                        aboveBaslineValues.addAsColumns(d.toString().replace(".", ","));
                    }
                    aboveBaslineValues.save();


                } catch (Exception e) {

                    System.err.println(e.getMessage());
                    System.err.println("Skipping " + dataSet);
                    badSets.add(dataSet);
                }
            }

            outF.save();
            for (String s : badSets) {
                System.err.println(s);
            }
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
