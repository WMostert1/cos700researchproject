import classifiers.IBkClassifier;
import classifiers.IClassify;
import com.google.common.base.Strings;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import fitness.FitnessEvaluator;
import fs.AMSO;
import fs.CorrelationbasedFeatureSubsetMethod;
import fs.EDAWrapperMethod;
import fs.FeatureSelectionAlgorithm;
import fs.FeatureSelectionResult;
import fs.FullPSOSearchFeatureSelection;
import fs.GeneticSearchWrapperMethod;
import fs.RankerInformationGainMethod;
import fs.RankerPearsonCorrelationMethod;
import fs.RankerReliefFMethod;
import fs.SequentialForwardSelection;
import landscape.HDILMeasure;
import landscape.NeutralityMeasure;
import mutators.UniformSampleMutator;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.commons.math3.util.Pair;
import utils.CsvOutputFormatter;
import utils.MathUtils;
import utils.OutputFormatter;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

import java.io.File;
import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static fs.RankerInformationGainMethod.buildInfoGainEvaluator;
import static utils.GlobalConstants.SAMPLE_SCALE;

public class InstanceSpaceAnalysis {

    public static String DATA_SET_PATH = "/Users/wmostert/Development/cos700researchproject/StatisticalTests/openml_datasets/";

    private static List<String> findDataSets(int minimumNumberOfAttributes, int maximumNumberOfAttributes, int maximumNumberOfDataInstances, int minimumNumberOfInstances) {
        List<String> dataSets = new ArrayList<>();
        File folder = new File(DATA_SET_PATH);
        File[] listOfFiles = folder.listFiles();
        for (int i = 0; i < listOfFiles.length; i++) {
            if (listOfFiles[i].isFile()) {
                String dataSetName = listOfFiles[i].getName();
                try {
                    Instances originalData = getDataSet(dataSetName);

                    if (originalData.numAttributes() < minimumNumberOfAttributes ||
                            originalData.numAttributes() > maximumNumberOfAttributes ||
                            originalData.numInstances() < minimumNumberOfInstances ||
                            originalData.numInstances() > maximumNumberOfDataInstances) {
                        continue;
                    }

                    System.out.println("name: " + dataSetName + ", #features: " + originalData.numAttributes() + ", #instances: " + originalData.numInstances());
                } catch (Exception e) {
                    continue;
                }
                dataSets.add(dataSetName);
//                return dataSets;
            }
        }
        return dataSets;
    }

    private static Instances getDataSet(String name) throws Exception {
        ConverterUtils.DataSource source = new ConverterUtils.DataSource("file://" + DATA_SET_PATH + name);
        //ConverterUtils.DataSource source = new ConverterUtils.DataSource(EvoCOPPaperApplication.class.getResourceAsStream(DATA_SET_PATH + name));

        Instances originalData = source.getDataSet();

        originalData.setClassIndex(originalData.numAttributes() - 1);

        return originalData;
    }

    private static void print(String val) {
        System.out.println(val);
    }


    private static BigDecimal getPerformanceOfAlgorithm(FitnessEvaluator fitnessEvaluator, Instances instances, FeatureSelectionAlgorithm algorithm) throws Exception {
        FeatureSelectionResult featureSelectionResult = algorithm.apply(instances, fitnessEvaluator);
        return MathUtils.doubleToBigDecimal(featureSelectionResult.getAccuracy());
    }

    private static Map<boolean[], Double> getSampleFitness(FitnessEvaluator fitnessEvaluator, Instances instances) throws Exception {
        Map<boolean[], Double> sample = Maps.newHashMap();
        UniformSampleMutator sampler = new UniformSampleMutator();
        for (long i = 0L; i < SAMPLE_SCALE * instances.numAttributes(); i++) {
            boolean[] solution = sampler.mutate(new boolean[instances.numAttributes()]);
            sample.put(solution, fitnessEvaluator.getQuality(solution, instances));
        }
        return sample;
    }

    private static DescriptiveStatistics getInfoGainToClassStats(Instances instances) throws Exception {
        InfoGainAttributeEval eval = buildInfoGainEvaluator(instances);
        DescriptiveStatistics stats = new DescriptiveStatistics();
        for (int i = 0; i < instances.numAttributes() - 1; i++) {
            stats.addValue(eval.evaluateAttribute(i));
        }
        return stats;
    }

    private static DescriptiveStatistics getInfoGainToFeaturesStats(Instances instances) throws Exception {
        InfoGainAttributeEval eval = buildInfoGainEvaluator(instances);
        DescriptiveStatistics stats = new DescriptiveStatistics();
        for (int j = 0; j < instances.numAttributes() - 1; j++) {
            instances.setClassIndex(j);
            for (int i = 0; i < instances.numAttributes() - 1; i++) {
                stats.addValue(eval.evaluateAttribute(i));
            }
        }
        instances.setClassIndex(instances.numAttributes() - 1);
        return stats;
    }


    public static void main(String[] args) throws Exception {
//        int minimumNumberOfAttributes = 10;
//        int maximumNumberOfAttributes = 500;
//        int maximumNumberOfDataInstances = 1200;
//        int minimumNumberOfInstances = 100;

        int minimumNumberOfAttributes = 500;
        int maximumNumberOfAttributes = 3000;
        int maximumNumberOfDataInstances = 2000;
        int minimumNumberOfInstances = 1200;

        Options options = new Options();

        Option minFeatures = new Option("minF", "minFeatures", true, "Minimum number of features to use");
        minFeatures.setRequired(false);
        options.addOption(minFeatures);

        Option maxFeatures = new Option("maxF", "maxFeatures", true, "Maximum number of features to use");
        maxFeatures.setRequired(false);
        options.addOption(maxFeatures);

        Option minInstances = new Option("minI", "minInstances", true, "Minimum number of instances to use");
        minInstances.setRequired(false);
        options.addOption(minInstances);

        Option maxInstances = new Option("maxI", "maxInstances", true, "Maximum number of instances to use");
        maxInstances.setRequired(false);
        options.addOption(maxInstances);

        Option datasets = new Option("ds", "datasetPath", true, "Path to datasets");
        datasets.setRequired(false);
        options.addOption(datasets);

        CommandLineParser parser = new DefaultParser();
        HelpFormatter formatter = new HelpFormatter();
        CommandLine cmd = null;

        try {
            cmd = parser.parse(options, args);
        } catch (ParseException e) {
            System.out.println(e.getMessage());
            formatter.printHelp("utility-name", options);
            System.exit(1);
        }

        String minFeaturesValue = cmd.getOptionValue("minFeatures");
        if (!Strings.isNullOrEmpty(minFeaturesValue)) {
            System.out.println("Using minFeatures = " + minFeaturesValue);
            minimumNumberOfAttributes = Integer.parseInt(minFeaturesValue);
        }

        String maxFeaturesValue = cmd.getOptionValue("maxFeatures");
        if (!Strings.isNullOrEmpty(maxFeaturesValue)) {
            System.out.println("Using maxFeatures = " + maxFeaturesValue);
            maximumNumberOfAttributes = Integer.parseInt(maxFeaturesValue);
        }

        String minInstancesValue = cmd.getOptionValue("minInstances");
        if (!Strings.isNullOrEmpty(minInstancesValue)) {
            System.out.println("Using minInstances = " + minInstancesValue);
            minimumNumberOfInstances = Integer.parseInt(minInstancesValue);
        }

        String maxInstancesValue = cmd.getOptionValue("maxInstances");
        if (!Strings.isNullOrEmpty(maxInstancesValue)) {
            System.out.println("Using maxInstances = " + maxInstancesValue);
            maximumNumberOfDataInstances = Integer.parseInt(maxInstancesValue);
        }

        String dataSetPathValue = cmd.getOptionValue("datasetPath");
        if (!Strings.isNullOrEmpty(dataSetPathValue)) {
            System.out.println("Using datasetPath = " + dataSetPathValue);
            DATA_SET_PATH = dataSetPathValue;
        }


        List<String> dataSets = findDataSets(minimumNumberOfAttributes, maximumNumberOfAttributes, maximumNumberOfDataInstances, minimumNumberOfInstances);
        print("Found " + dataSets.size() + " datasets.");
        NeutralityMeasure neutralityMeasure = new NeutralityMeasure();
        IClassify classifier = new IBkClassifier();

        List<FeatureSelectionAlgorithm> algorithms = Lists.newArrayList();
        algorithms.add(new AMSO());
//        algorithms.add(new FullPSOSearchFeatureSelection());
        algorithms.add(new GeneticSearchWrapperMethod());
        algorithms.add(new EDAWrapperMethod());

        algorithms.add(new SequentialForwardSelection());
        algorithms.add(new RankerPearsonCorrelationMethod());
//        algorithms.add(new CorrelationbasedFeatureSubsetMethod());
        algorithms.add(new RankerInformationGainMethod());
//        algorithms.add(new RankerPrincipalComponentsMethod());
        algorithms.add(new RankerReliefFMethod());


        List<String> featureNames = Lists.newArrayList("feature_num_features",
                "feature_num_classes",
                "feature_num_instances",
                "feature_neutral_m1",
                "feature_neutral_m2",
                "feature_hdil_min",
                "feature_hdil_max",
                "feature_hdil_median",
                "feature_hdil_stdev",
                "feature_hdil_skewness",
                "feature_infog_class_min",
                "feature_infog_class_max",
                "feature_infog_class_median",
                "feature_infog_class_stdev",
                "feature_infog_class_skewness",
                "feature_infog_feature_min",
                "feature_infog_feature_max",
                "feature_infog_feature_median",
                "feature_infog_feature_stdev",
                "feature_infog_feature_skewness");
        List<String> headers = Lists.newArrayList("Instances");
        headers.addAll(featureNames);
        for (FeatureSelectionAlgorithm fs : algorithms) {
            headers.add("algo_" + fs.getAlgorithmName());
        }

        OutputFormatter outF = new CsvOutputFormatter("out/fs-metadata.csv", headers.toArray(new String[]{}));

        for (String dataset : dataSets) {
            System.out.println("Starting " + dataset);
            try {
                Instances instances = getDataSet(dataset);
                FitnessEvaluator fitnessEvaluator = new FitnessEvaluator(classifier, instances.numAttributes() - 1);

                List<String> metadata = Lists.newArrayList();
                //Instances
                metadata.add(dataset);
                //feature_num_features
                metadata.add(Integer.toString(instances.numAttributes() - 1));
                //feature_num_classes
                metadata.add(Integer.toString(instances.numClasses()));
                //feature_num_instances
                metadata.add(Integer.toString(instances.numInstances()));

                Pair<BigDecimal, BigDecimal> neutrality = neutralityMeasure.get(fitnessEvaluator, instances);
                //feature_neutral_m1
                metadata.add(MathUtils.bigDecimalToString(neutrality.getFirst()));
                //feature_neutral_m2
                metadata.add(MathUtils.bigDecimalToString(neutrality.getSecond()));

                //hdil
                Map<Integer, Double> hdil = new HDILMeasure(20, BigDecimal.ZERO.doubleValue(), BigDecimal.ONE.doubleValue()).get(getSampleFitness(fitnessEvaluator, instances));
                DescriptiveStatistics hdil_stats = new DescriptiveStatistics();
                hdil.forEach((x, y) -> hdil_stats.addValue(y));
                //feature_hdil_min
                metadata.add(MathUtils.doubleToString(hdil_stats.getMin()));
                //feature_hdil_max
                metadata.add(MathUtils.doubleToString(hdil_stats.getMax()));
                //"feature_hdil_median
                metadata.add(MathUtils.doubleToString(hdil_stats.getPercentile(50.0)));
                //feature_hdil_stdev
                metadata.add(MathUtils.doubleToString(hdil_stats.getStandardDeviation()));
                //feature_hdil_skewness
                metadata.add(MathUtils.doubleToString(hdil_stats.getSkewness()));

                //mutual info to class
                DescriptiveStatistics infog_class_stats = getInfoGainToClassStats(instances);
                //feature_infog_class_min
                metadata.add(MathUtils.doubleToString(infog_class_stats.getMin()));
                //feature_infog_class_max
                metadata.add(MathUtils.doubleToString(infog_class_stats.getMax()));
                //feature_infog_class_median
                metadata.add(MathUtils.doubleToString(infog_class_stats.getPercentile(50.0)));
                //feature_infog_class_stdev
                metadata.add(MathUtils.doubleToString(infog_class_stats.getStandardDeviation()));
                //feature_infog_class_skewness
                metadata.add(MathUtils.doubleToString(infog_class_stats.getSkewness()));

                DescriptiveStatistics infog_feature_stats = getInfoGainToFeaturesStats(instances);
                //feature_infog_feature_min
                metadata.add(MathUtils.doubleToString(infog_feature_stats.getMin()));
                //feature_infog_feature_max
                metadata.add(MathUtils.doubleToString(infog_feature_stats.getMax()));
                //feature_infog_feature_median
                metadata.add(MathUtils.doubleToString(infog_feature_stats.getPercentile(50.0)));
                //feature_infog_feature_stdev
                metadata.add(MathUtils.doubleToString(infog_feature_stats.getStandardDeviation()));
                //feature_infog_feature_skewness
                metadata.add(MathUtils.doubleToString(infog_feature_stats.getSkewness()));

                List<String> algorithmPerformance = Lists.newArrayList();
                for (FeatureSelectionAlgorithm fs : algorithms) {
                    algorithmPerformance.add(MathUtils.bigDecimalToString(getPerformanceOfAlgorithm(fitnessEvaluator, instances, fs)));
                }

                metadata.addAll(algorithmPerformance);
                outF.addAsColumns(metadata.toArray());
                System.out.println("Finished " + dataset);

                outF.save();
            } catch (Exception e) {
                System.err.println("Failed on " + dataset);
                e.printStackTrace();
            }
        }


    }
}
