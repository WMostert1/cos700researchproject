import classifiers.IBkClassifier;
import classifiers.IClassify;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import fitness.FitnessEvaluator;
import fs.*;
import fs.pso.BoothFunction;
import fs.pso.GbestPSO;
import fs.pso.Particle;
import landscape.FitnessDistributionMeasure;
import landscape.NeutralityMeasure;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.commons.math3.util.Pair;
import utils.CompositeOutputFormatter;
import utils.CsvOutputFormatter;
import utils.LatexOutputFormatter;
import utils.MathUtils;
import utils.OutputFormatter;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

import java.io.*;
import java.math.BigDecimal;
import java.math.MathContext;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.text.DecimalFormat;
import java.util.*;

import static utils.MathUtils.bigDecimalToString;
import static utils.MathUtils.doubleToBigDecimal;
import static utils.MathUtils.doubleToString;


/**
 * Created by bbdnet1339 on 2016/08/05.
 */
public class JournalApplication {

    private static void sortIntArr(int[] arr) {
        for (int i = 0; i < arr.length; i++) {
            for (int j = 0; j < arr.length; j++) {
                if (i == j) {
                    continue;
                }

                if (arr[j] > arr[i]) {
                    int temp = arr[i];
                    arr[i] = arr[j];
                    arr[j] = temp;
                }
            }
        }
    }

    public static void startAll(Collection<Thread> c) {
        for(Thread t : c) t.start();
    }

    public static void waitFor(Collection<Thread> c) throws InterruptedException {
        for(Thread t : c) t.join();
    }

    private InputStream getResourceAsStream(String resource) {
        final InputStream in
                = getContextClassLoader().getResourceAsStream(resource);

        return in == null ? getClass().getResourceAsStream(resource) : in;
    }

    private ClassLoader getContextClassLoader() {
        return Thread.currentThread().getContextClassLoader();
    }


    public static Instances getDataSet(String name) throws Exception {
        DataSource source = new DataSource(EvoCOPPaperApplication.class.getResourceAsStream("data-sets/used/" + name));

        Instances originalData = source.getDataSet();

        originalData.setClassIndex(originalData.numAttributes() - 1);

        return originalData;
    }

    public static void main(String[] args) throws Exception {
        //Paramaters for the data sets to be used
//        int numberOfMaximumDataSetsToFind = 30;
//        int minimumNumberOfAttributes = 10;
//        int maximumNumberOfDataInstances = 800;

        int numberOfMaximumDataSetsToFind = 30;
        int minimumNumberOfAttributes = 10;
        int maximumNumberOfDataInstances = 800;

        List<String> dataSets = new ArrayList<>();
        boolean moveToErrorFolder = false;

        File folder = new File("./src/main/resources/data-sets/used");
        File[] listOfFiles = folder.listFiles();

        for (int i = 0; i < listOfFiles.length && dataSets.size() < numberOfMaximumDataSetsToFind; i++) {
            if (listOfFiles[i].isFile()) {
                String dataSetName = listOfFiles[i].getName();
                Instances originalData = getDataSet(dataSetName);
                if (originalData.numAttributes() < minimumNumberOfAttributes ||
                        originalData.numInstances() > maximumNumberOfDataInstances) {
                    continue;
                }
                dataSets.add(dataSetName);
            }
        }

        System.out.println("Initializing data sets to use.");
        System.out.println("Found  " + dataSets.size() + " data sets to use.");

        for (String dataSetName : dataSets) {
            System.out.println(dataSetName);
        }

        File directory = new File("out/fitness");
        if (!directory.exists()) {
            directory.mkdir();
            // If you require it to make the entire directory path including parents,
            // use directory.mkdirs(); here instead.
        }

        List<String> badSets = new ArrayList<>();

        OutputFormatter outF = new CsvOutputFormatter("out/relative-fs.csv", "Dataset", "Feature Selection Algorithm", "Baseline", "Relative Fitness");
        OutputFormatter outStats = new CsvOutputFormatter("out/stats.csv");
        OutputFormatter outNeutrality = new CsvOutputFormatter("out/neutrality.csv");
        OutputFormatter outRuggedness = new CsvOutputFormatter("out/rugedness.csv");
        OutputFormatter outFitnessFrequency = new CsvOutputFormatter("out/fitnessfrequency.csv");

        List<FeatureSelectionAlgorithm> algorithms = Lists.newArrayList();

        algorithms.add(new RandomFeatureSelection());
        //algorithms.add(new FullPSOSearchFeatureSelection());
        algorithms.add(new AMSO());
        algorithms.add(new RankerInformationGainMethod());
        algorithms.add(new SequentialForwardSelection());
        algorithms.add(new RankerPearsonCorrelationMethod());
        //algorithms.add(new CorrelationbasedFeatureSubsetMethod());
        algorithms.add(new SequentialBackwardSelection());
        algorithms.add(new GeneticSearchWrapperMethod());

        List<String> algorithmNames = Lists.newArrayList();
        for(FeatureSelectionAlgorithm fsa : algorithms){
            algorithmNames.add(fsa.getAlgorithmName());
        }

        //Set up output format data structure

        String[][] bfiTable = new String[dataSets.size()][];
        for (int i = 0; i < bfiTable.length; i++) {
            bfiTable[i] = new String[algorithms.size()+1];
        }

        for (int i = 0; i < dataSets.size() ; i++) {
            bfiTable[i][0] = dataSets.get(i);
        }

//        for (int i = 0; i < algorithms.size(); i++) {
//            bfiTable[0][i] = algorithms.get(i).getAlgorithmName();
//        }

        String[][] fitnessTable = new String[dataSets.size() + 1][];
        for (int i = 0; i < fitnessTable.length; i++) {
            fitnessTable[i] = new String[algorithms.size() + 2];
        }

        for (int i = 1; i < dataSets.size() + 1; i++) {
            fitnessTable[i][0] = dataSets.get(i - 1);
        }

        fitnessTable[0][1] = "Baseline";

        for (int i = 2; i < algorithms.size() + 1; i++) {
            fitnessTable[0][i] = algorithms.get(i - 1).getAlgorithmName();
        }


        NeutralityMeasure neutralityMeasure = new NeutralityMeasure();
        dataSetNumber = 0;

        Map<String, ArrayList<BigDecimal>> statsValues = new HashMap<>();
        for (String dataSet : dataSets) {

            try {
                Instances originalData = getDataSet(dataSet);


                System.out.println("------------  " + dataSet + "  ------------");
                System.out.println("------------- " + originalData.numAttributes() + " Features --------------");
                System.out.println("------------- " + originalData.numInstances() + " Instances ------------");
                saveDataSetInfo(new DataSetInfo(dataSet, originalData.numAttributes() - 1, originalData.numInstances(), originalData.numClasses()));

                IClassify classifier = new IBkClassifier();
                FitnessEvaluator fitnessEvaluator = new FitnessEvaluator(classifier, originalData.numAttributes() - 1);
//                  Map<ConcreteBinarySolution, Double> fitnessMap = filterEval.eval(originalData);


                //Get the baseline fitness value
                boolean[] allFeaturesDesign = new boolean[originalData.numAttributes() - 1];
                for (int i = 0; i < allFeaturesDesign.length; i++) {
                    allFeaturesDesign[i] = true;
                }


                //Get solutions based on different feature selection algorithms
                DescriptiveStatistics descriptiveStatistics = new DescriptiveStatistics();

                //ConcreteBinarySolution allFeaturesSelectedSolution = (ConcreteBinarySolution) ConcreteBinarySolution.constructBinarySolution(allFeaturesDesign);

                BigDecimal baselineFitness = doubleToBigDecimal(fitnessEvaluator.getQuality(allFeaturesDesign, originalData));
                fitnessTable[dataSetNumber][1] = baselineFitness.toString();
                int algorithmNumber = 1;

                List<Thread> fsaThreads = Lists.newArrayList();


                for (FeatureSelectionAlgorithm fsa : algorithms) {

                    fsaThreads.add(new Thread(new FSRunner(dataSet, fsa, originalData, fitnessEvaluator, baselineFitness, dataSetNumber, algorithmNumber++, bfiTable, outF, descriptiveStatistics, statsValues)));

//                    try {
//                        System.out.println(dataSet + " :: " + fsa.getAlgorithmName() + " :: RUNNING");
//                        FeatureSelectionResult fsResults = fsa.apply(originalData, fitnessEvaluator);
//                        BigDecimal fitness = doubleToBigDecimal(fsResults.getAccuracy()).subtract(baselineFitness);
//                        descriptiveStatistics.addValue(fitness.doubleValue());
//
//                        outF.addAsColumns(dataSet, fsa.getAlgorithmName(), Double.toString(baselineFitness.doubleValue()),
//                                Double.toString(fitness.doubleValue()));
//                        System.out.println(dataSet + " :: " + fsa.getAlgorithmName() + " :: COMPLETED (" + fitness.toString() + ")");
//                        bfiTable[dataSetNumber][algorithmNumber++] = bigDecimalToString(fitness);
//                        fitnessTable[dataSetNumber][algorithmNumber] = doubleToBigDecimal(fsResults.getAccuracy()).toString();
//                    } catch (Exception e) {
//                        bfiTable[dataSetNumber][algorithmNumber++] = "ERR";
//                        fitnessTable[dataSetNumber][algorithmNumber] = "ERR";
//                        System.err.println(e.getMessage());
//                    }
//
//                    if (fsa instanceof StochasticFeatureSelection) {
//                        ArrayList<BigDecimal> iterationValues = ((StochasticFeatureSelection) fsa).getIterationFitnessValues();
//                        File dsDirectory = new File("out/fitness/" + dataSet);
//                        if (!dsDirectory.exists()) {
//                            dsDirectory.mkdir();
//                            // If you require it to make the entire directory path including parents,
//                            // use directory.mkdirs(); here instead.
//                        }
//                        OutputFormatter outputFormatter = new CsvOutputFormatter("out/fitness/" + dataSet + "/" + fsa.getClass().getSimpleName() + ".csv");
//                        ((StochasticFeatureSelection) fsa).recordFitnessValues(outputFormatter, iterationValues, baselineFitness);
//                        if (!statsValues.containsKey(fsa.getAlgorithmName())) {
//                            statsValues.put(fsa.getAlgorithmName(), iterationValues);
//                        } else {
//                            statsValues.get(fsa.getAlgorithmName()).addAll(iterationValues);
//                        }
//                    }
                }

                startAll(fsaThreads);
                waitFor(fsaThreads);


                //Measures specific to dataset
                System.out.println("Calculating fitness landscape characteristics...");
                outF.addEmptyRow();
                outF.addAsColumns("STD DEVIATION", Double.toString(descriptiveStatistics.getStandardDeviation()));
                outF.addAsColumns("Number of features", Integer.toString(originalData.numAttributes() - 1));

                System.out.println("Calculating neutrality...");
                Pair<BigDecimal, BigDecimal> neatraility = neutralityMeasure.get(fitnessEvaluator, originalData);
                outNeutrality.addAsColumns(dataSet, neatraility.getFirst().toString(), neatraility.getSecond().toString());

                System.out.println("Calculating fitness distribution...");
                FitnessDistributionMeasure fitnessDistributionMeasure = new FitnessDistributionMeasure(20, -1.0, 1.0, 20);
                Map<Integer, Map<boolean[], Double>> fitnessDistro = fitnessDistributionMeasure.get(originalData, fitnessEvaluator);
                outFitnessFrequency.addAsColumns(dataSet);

                for (int i = 0; i < 20; i++) {
                    String key = doubleToString((-1.0 + (0.1 * i)), 2) + "<=f(s)<" + doubleToString(-1.0 + (0.1 * (i + 1)), 2);
                    key = key.replace(",", ".");
                    outFitnessFrequency.addAsColumns(key, Integer.toString(fitnessDistro.get(i).keySet().size()));
                }








                    /*
                        baseline : 0.5
                        fitness: -0.3
                        relative = fitness - baseline
                     */

                    /*
                        fitness = [ -1, 1 ], -1 means penalised to the max and fscore is 0
                                              1 means fscore is 1 and 0 penalty since 1 feature selected

                        baseline = 0...1 - (1) [-1; 0]

                        bfi = [ -1 ; 1 ] - [ -1 ; 0 ]

                        [ -1 ; 2]
                     */
                dataSetNumber++;

                outF.save();
                outNeutrality.save();
                outFitnessFrequency.save();
            } catch (Exception e) {
                if (moveToErrorFolder) {
                    Files.move(Paths.get("./src/main/resources/data-sets/used/" + dataSet), Paths.get("./src/main/resources/data-sets/error/" + dataSet));
                }

                System.err.println(e.getMessage());
                System.err.println("Skipping " + dataSet + " and moved to error folder.");
                badSets.add(dataSet);
            }


        }

        for (String fsa : statsValues.keySet()) {
            List<String> vals = Lists.newArrayList();
            statsValues.get(fsa).forEach((v) -> vals.add(v.toString()));
            vals.add(0, fsa);
            outStats.addAsColumns(vals.toArray(new String[vals.size()]));
        }
        outStats.save();

        algorithmNames.add(0, "");
        OutputFormatter bfiOut = new CompositeOutputFormatter(
                new CsvOutputFormatter("out/bfiTable.csv", algorithmNames.toArray(new String[0])),
                new LatexOutputFormatter("out/bfiTable.tex", "BFI Table", "tbl:bfi", algorithmNames.toArray(new String[0]))
        );


        bfiOut.addAsColumns();
        for (String[] strings : bfiTable) {
            bfiOut.addAsColumns(strings);
        }
        bfiOut.save();

        OutputFormatter fitnessOut = new CsvOutputFormatter("out/fitnessTable.csv");
        for (String[] strings : fitnessTable) {
            fitnessOut.addAsColumns(strings);
        }
        fitnessOut.save();


        for (String s : badSets) {
            System.err.println(s);
        }
    }

    private static int dataSetNumber = 0;


    private static void buildDataSetInfo(OutputFormatter outF, String name, int noAttrs, int noInstances, int noClasses) {
        if (dataSetNumber == 0) {
            outF.addAsColumns("Datasets");
            outF.addAsColumns("Index", "Name", "No. Attributes", "No. Instances", "No. Classes");
        }

        outF.addAsColumns(Integer.toString(dataSetNumber++), name, Integer.toString(noAttrs),
                Integer.toString(noInstances), Integer.toString(noClasses));
    }

    public static class DataSetInfo {

        private String name;
        int noAttrs, noInstances, noClasses;

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

    private static void saveDataSetInfo(DataSetInfo info) {
        dataSetReport.add(info);
    }

    private static void flushDataSetInfo(OutputFormatter outF) {
        for (DataSetInfo dataSetInfo : dataSetReport) {
            buildDataSetInfo(outF, dataSetInfo.getName(), dataSetInfo.getNoAttrs(), dataSetInfo.getNoInstances(), dataSetInfo.getNoClasses());
        }
    }

    private static List<String> algorithmNames = Lists.newArrayList();

    private static void buildFSInfo(OutputFormatter outF, String algorithmName, String dataSetName, int numberOfGlobalOptima,
                                    boolean foundLocalOpt, boolean foundGlobalOpt, double percentageCap, double successRatio, double fsAccuracy, double globalAccuracy) {
        if (!algorithmNames.contains(algorithmName)) {
            algorithmNames.add(algorithmName);
            outF.addEmptyRow();
            outF.addAsColumns(algorithmName);
            outF.addAsColumns("Dataset name", "No. Global Optima", "Found Local Optima", "Found Global Optima", "Percentage Cap", "Success Ratio", "Feat. S. Fitness", "Global Opt. Fitness");
        }

        outF.addAsColumns(dataSetName, Integer.toString(numberOfGlobalOptima), Boolean.toString(foundLocalOpt),
                Boolean.toString(foundGlobalOpt), Double.toString(percentageCap), Double.toString(successRatio), Double.toString(fsAccuracy), Double.toString(globalAccuracy));

    }

    private static class FSReportInfo {

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

    private static void saveFSInfo(FSReportInfo reportInfo) {
        HashMap<String, FSReportInfo> algorithmReport = fsReportInfo.get(reportInfo.algorithmName);
        if (algorithmReport == null) {
            algorithmReport = Maps.newHashMap();
        }

        algorithmReport.put(reportInfo.getDataSetName(), reportInfo);
        fsReportInfo.put(reportInfo.getAlgorithmName(), algorithmReport);
    }

    private static void flushFSInfo(OutputFormatter outF) {
        for (String algorithmName : fsReportInfo.keySet()) {
            for (String dataSetName : fsReportInfo.get(algorithmName).keySet()) {
                FSReportInfo report = fsReportInfo.get(algorithmName).get(dataSetName);
                buildFSInfo(outF, algorithmName, dataSetName, report.getNumberOfGlobalOptima(), report.isFoundLocalOpt(),
                        report.isFoundGlobalOpt(), report.getPercentageCap(), report.getSuccessRatio(), report.getFsAccuracy(), report.getGlobalAccuracy());
            }
        }
    }
}
