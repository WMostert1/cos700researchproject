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
import utils.*;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

import java.io.*;
import java.math.BigDecimal;
import java.math.MathContext;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.text.DecimalFormat;
import java.time.Instant;
import java.util.*;

import static fs.FSRunner.RUN_DELIMINATOR;
import static utils.MathUtils.bigDecimalToString;
import static utils.MathUtils.doubleToBigDecimal;
import static utils.MathUtils.doubleToString;


/**
 * Created by bbdnet1339 on 2016/08/05.
 */
public class JournalApplication {

    public static final String DATA_SET_PATH = "data-sets/used/";

    public static void startAll(Collection<Thread> c) {
        for (Thread t : c) {
            t.start();
        }
    }

    public static void waitFor(Collection<Thread> c) throws InterruptedException {
        for (Thread t : c) {
            t.join();
        }
    }

    public static Instances getDataSet(String name) throws Exception {
        DataSource source = new DataSource(EvoCOPPaperApplication.class.getResourceAsStream(DATA_SET_PATH + name));

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
        int maximumNumberOfAttributes = 1000;
        int maximumNumberOfDataInstances = 800;

        List<String> dataSets = new ArrayList<>();
        boolean moveToErrorFolder = false;
        //"./src/main/resources/data-sets/used"
        File folder = new File("./src/main/resources/" + DATA_SET_PATH);
        File[] listOfFiles = folder.listFiles();

        //"arcene.arff"
        List<String> excludeList = Lists.newArrayList();


        for (int i = 0; i < listOfFiles.length && dataSets.size() < numberOfMaximumDataSetsToFind; i++) {
            if (listOfFiles[i].isFile()) {
                String dataSetName = listOfFiles[i].getName();

                if(excludeList.contains(dataSetName)){
                    continue;
                }

                try {
                    Instances originalData = getDataSet(dataSetName);

                    if (originalData.numAttributes() < minimumNumberOfAttributes ||
                            originalData.numAttributes() > maximumNumberOfAttributes ||
                            originalData.numInstances() > maximumNumberOfDataInstances) {
                        continue;
                    }

                    System.out.println("Number of attributes: " + originalData.numAttributes() + " : " + dataSetName);
                } catch (Exception e) {
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

        new File("out/"+RUN_DELIMINATOR).mkdir();
        File directory = new File("out/"+RUN_DELIMINATOR+"/fitness");
        if (!directory.exists()) {
            directory.mkdir();
            // If you require it to make the entire directory path including parents,
            // use directory.mkdirs(); here instead.
        }

        List<String> badSets = new ArrayList<>();

        OutputFormatter outF = new CsvOutputFormatter("out/"+RUN_DELIMINATOR+"/relative-fs.csv", "Dataset", "Feature Selection Algorithm", "Baseline", "Relative Fitness");
        OutputFormatter outStats = new CsvOutputFormatter("out/"+RUN_DELIMINATOR+"/stats.csv");
        OutputFormatter outNeutrality = new CsvOutputFormatter("out/"+RUN_DELIMINATOR+"/neutrality.csv");
        OutputFormatter outRuggedness = new CsvOutputFormatter("out/"+RUN_DELIMINATOR+"/rugedness.csv");
        OutputFormatter outFitnessFrequency = new CsvOutputFormatter("out/"+RUN_DELIMINATOR+"/fitnessfrequency.csv");

        List<FeatureSelectionAlgorithm> algorithms = Lists.newArrayList();

        algorithms.add(new RandomFeatureSelection());
        //algorithms.add(new FullPSOSearchFeatureSelection());
        algorithms.add(new AMSO());
        algorithms.add(new GeneticSearchWrapperMethod());
        algorithms.add(new SequentialBackwardSelection());
        algorithms.add(new SequentialForwardSelection());
        algorithms.add(new RankerPearsonCorrelationMethod());
        //algorithms.add(new CorrelationbasedFeatureSubsetMethod());
        algorithms.add(new RankerInformationGainMethod());


        List<String> algorithmNames = Lists.newArrayList();
        for (FeatureSelectionAlgorithm fsa : algorithms) {
            algorithmNames.add(fsa.getAlgorithmName());
        }

        //Set up output format data structure

        String[][] bfiTable = new String[dataSets.size()][];
        for (int i = 0; i < bfiTable.length; i++) {
            bfiTable[i] = new String[algorithms.size() + 1];
        }

        for (int i = 0; i < dataSets.size(); i++) {
            bfiTable[i][0] = dataSets.get(i);
        }

//        for (int i = 0; i < algorithms.size(); i++) {
//            bfiTable[0][i] = algorithms.get(i).getAlgorithmName();
//        }

        String[][] fitnessTable = new String[dataSets.size()][];
        for (int i = 0; i < fitnessTable.length; i++) {
            //dataset name, baseline, algorithms
            fitnessTable[i] = new String[2 + algorithms.size()];
        }

        for (int i = 0; i < dataSets.size(); i++) {
            fitnessTable[i][0] = dataSets.get(i);
        }

        for (int i = 2; i < algorithms.size() + 1; i++) {
            fitnessTable[0][i] = algorithms.get(i - 1).getAlgorithmName();
        }


        NeutralityMeasure neutralityMeasure = new NeutralityMeasure();
        dataSetNumber = 0;

        Map<String, ArrayList<BigDecimal>> statsValues = new HashMap<>();

        //build data set information
        for (String dataSet : dataSets) {
            Instances originalData = getDataSet(dataSet);
            saveDataSetInfo(new DataSetInfo(dataSet, originalData.numAttributes(), originalData.numInstances(), originalData.numClasses()));
        }
        flushDataSetInfo(new CompositeOutputFormatter(new LatexOutputFormatter("out/"+RUN_DELIMINATOR+"/dataSetInformation.tex", "Datasets", "tbl:datasets",
                "Identifier", "Name", "# Attributes", "# Instances", "# Classes"
        ), new CsvOutputFormatter("out/dataSetInformation.csv", "Index", "Name", "No. Attributes", "No. Instances", "No. Classes")));

        dataSetNumber = 0;

        for (String dataSet : dataSets) {

            try {
                Instances originalData = getDataSet(dataSet);


                System.out.println("------------  " + dataSet + "  ------------");
                System.out.println("------------- " + originalData.numAttributes() + " Features --------------");
                System.out.println("------------- " + originalData.numInstances() + " Instances ------------");


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

                    fsaThreads.add(new Thread(new FSRunner(dataSet, fsa, originalData, fitnessEvaluator, baselineFitness, dataSetNumber, algorithmNumber++, bfiTable, fitnessTable, outF, descriptiveStatistics, statsValues)));

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
//                System.out.println("Calculating fitness landscape characteristics...");
//                outF.addEmptyRow();
//                outF.addAsColumns("STD DEVIATION", Double.toString(descriptiveStatistics.getStandardDeviation()));
//                outF.addAsColumns("Number of features", Integer.toString(originalData.numAttributes() - 1));
//
//                System.out.println("Calculating neutrality...");
                Pair<BigDecimal, BigDecimal> neatraility = neutralityMeasure.get(fitnessEvaluator, originalData);
                outNeutrality.addAsColumns(dataSet, neatraility.getFirst().toString(), neatraility.getSecond().toString());
//
//                System.out.println("Calculating fitness distribution...");
//                FitnessDistributionMeasure fitnessDistributionMeasure = new FitnessDistributionMeasure(20, -1.0, 1.0, 20);
//                Map<Integer, Map<boolean[], Double>> fitnessDistro = fitnessDistributionMeasure.get(originalData, fitnessEvaluator);
//                outFitnessFrequency.addAsColumns(dataSet);
//
//                for (int i = 0; i < 20; i++) {
//                    String key = doubleToString((-1.0 + (0.1 * i)), 2) + "<=f(s)<" + doubleToString(-1.0 + (0.1 * (i + 1)), 2);
//                    key = key.replace(",", ".");
//                    outFitnessFrequency.addAsColumns(key, Integer.toString(fitnessDistro.get(i).keySet().size()));
//                }

                dataSetNumber++;

                outF.save();
                outNeutrality.save();
                outFitnessFrequency.save();

                System.out.println(((int) ((double) dataSetNumber / dataSets.size())) + "% finished.");
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

        //Generate the bfi tables
        OutputFormatter bfiOutCsv = new CsvOutputFormatter("out/"+RUN_DELIMINATOR+"/bfiTable.csv", algorithmNames.toArray(new String[0]));
        OutputFormatter bfiOutLatex = new LatexOutputFormatter("out/"+RUN_DELIMINATOR+"/bfiTable.tex", "BFI Table", "tbl:bfi", algorithmNames.toArray(new String[0]));


        bfiOutCsv.addAsColumns();
        for (String[] strings : bfiTable) {
            String[] csvStrings = new String[strings.length];
            for (int i = 0; i < strings.length; i++) {
                csvStrings[i] = strings[i].substring(0, 2 + GlobalConstants.REPORTED_DECIMAL_PLACES);
            }
            bfiOutCsv.addAsColumns(csvStrings);
            bfiOutLatex.addAsColumns(strings);
        }
        bfiOutCsv.save();
        bfiOutLatex.save();

//        //Generate the bfi table with number of features instead of data sets for scatter plot
//        OutputFormatter bfiScatterOut = new CsvOutputFormatter("out/bfiTableScatter.csv", algorithmNames.toArray(new String[0]));
//
//        int i = 0;
//        for (String[] strings : bfiTable) {
//            String [] csvStrings = new String[strings.length];
//            csvStrings[0] = Integer.toString(getDataSet(dataSets.get(i++)).numAttributes());
//            for(int k = 0; k < strings.length;k++){
//                csvStrings[k] = strings[k].substring(0, 2 + GlobalConstants.REPORTED_DECIMAL_PLACES);
//            }
//            bfiScatterOut.addAsColumns(csvStrings);
//        }
//        bfiScatterOut.save();


        //Save data set info table


        List<String> fitnessTableHeaders = Lists.newArrayList(algorithmNames);
        fitnessTableHeaders.add(0, "Baseline");
        OutputFormatter fitnessOut = new CsvOutputFormatter("out/"+RUN_DELIMINATOR+"/fitnessTable.csv", fitnessTableHeaders.toArray(new String[0]));
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
        outF.addAsColumns("D"+Integer.toString(++dataSetNumber), name, Integer.toString(noAttrs),
                Integer.toString(noInstances), Integer.toString(noClasses));
    }


    private static List<DataSetInfo> dataSetReport = Lists.newArrayList();

    private static void saveDataSetInfo(DataSetInfo info) {
        dataSetReport.add(info);
    }

    private static void flushDataSetInfo(OutputFormatter outF) throws FileNotFoundException, UnsupportedEncodingException {
        for (DataSetInfo dataSetInfo : dataSetReport) {
            buildDataSetInfo(outF, dataSetInfo.getName().replace(".arff",""), dataSetInfo.getNoAttrs(), dataSetInfo.getNoInstances(), dataSetInfo.getNoClasses());
        }
        outF.save();
    }


}
