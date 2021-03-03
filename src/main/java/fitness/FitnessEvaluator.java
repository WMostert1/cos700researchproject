package fitness;

import classifiers.IClassify;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import exceptions.DataException;
import fs.FeatureSelectorUtils;
import landscape.FitnessDistributionMeasure;
import lons.examples.ConcreteBinarySolution;
import samplers.ExhaustiveSampler;
import samplers.SolutionSampler;
import utils.DataSetInstanceSplitter;
import utils.MathUtils;
import weka.core.Attribute;
import weka.core.Instances;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import static utils.GlobalConstants.PERCENTAGE_SPLIT;

/**
 * Created by bbdnet1339 on 2016/08/08.
 */
public class FitnessEvaluator {


    private IClassify classifier;
    private static final int NUMBER_OF_THREADS = 5;
    private ExecutorService es = Executors.newCachedThreadPool();
    private boolean multiThread = false;
    private final BigDecimal initialNumberOfAttributes;
    private Map<String, Double> fitnessCache = Maps.newHashMap();
    private final static Double WORST_FITNESS = 0.0;
    private boolean verbose = false;

    public static final BigDecimal classifierScalingConstant = BigDecimal.valueOf(1.0);
    public static final BigDecimal penaltyScalingConstant = BigDecimal.valueOf(0.25);


    public FitnessEvaluator(IClassify classifier, int originalNumberOfAttributes, boolean verbose) {
        this.classifier = classifier;
        this.initialNumberOfAttributes = MathUtils.doubleToBigDecimal((double) originalNumberOfAttributes);
        this.verbose = verbose;
    }


    public FitnessEvaluator(IClassify classifier, int originalNumberOfAttributes) {
        this.classifier = classifier;
        this.initialNumberOfAttributes = MathUtils.doubleToBigDecimal((double) originalNumberOfAttributes);
    }

    public FitnessEvaluator(IClassify classifier, boolean multiThread, int originalNumberOfAttributes, double penaltyVal) {
        this.multiThread = multiThread;
        this.classifier = classifier;
        this.initialNumberOfAttributes = MathUtils.doubleToBigDecimal((double) originalNumberOfAttributes);
        ;
    }

    //Baseline Fitness Improvement
    public Double getQuality(boolean[] solution, Instances data) throws Exception {
        String key = FeatureSelectorUtils.booleanArrayToBitString(solution);
        if (!key.contains("1")) {
            //No features exist, return -1.0
            return WORST_FITNESS;
        }

        Double cachedFitness = fitnessCache.get(key);
        if (cachedFitness != null) {
            return cachedFitness;
        }

        Instances currentInstances = FeatureSelectorUtils.getInstancesFromBitString(data, solution);

        if (verbose) {
            String attributes = "Using attributes : ";
            for (int i = 0; i < currentInstances.numAttributes(); i++) {
                Attribute attr = currentInstances.attribute(i);
                attributes += attr.name() + ",";
            }
            System.out.println(attributes);
        }

        DataSetInstanceSplitter splitter = new DataSetInstanceSplitter(currentInstances, PERCENTAGE_SPLIT);
        Instances trainingSet = splitter.getTrainingSet();
        Instances testingSet = splitter.getTestingSet();

        BigDecimal classifierAccuracy = MathUtils.doubleToBigDecimal(classifier.getClassificationAccuracy(trainingSet, testingSet));
        classifierAccuracy = classifierAccuracy.multiply(classifierScalingConstant);

        BigDecimal numberOfAttributes = MathUtils.doubleToBigDecimal((double) (currentInstances.numAttributes() - 1));

        if (numberOfAttributes.compareTo(initialNumberOfAttributes) > 0) {
            throw new RuntimeException("Remember to construct this class for each data set");
        }

        //The smaller this value the better
        //nfs = number of features selected
        //nf = number of features available
        //penalty(nfs) = 1/9 (10^((nfs-1)/(nf-1)) - 1)
        BigDecimal exponent = (numberOfAttributes.subtract(BigDecimal.ONE)).divide(initialNumberOfAttributes.subtract(BigDecimal.ONE), MathUtils.ROUNDING_MODE);
        BigDecimal penalty = BigDecimal.valueOf(Math.pow(10.0, exponent.doubleValue())).subtract(BigDecimal.ONE).divide(BigDecimal.valueOf(9.0), MathUtils.ROUNDING_MODE);
        penalty = penalty.multiply(penaltyScalingConstant);

        Double fitness = classifierAccuracy.subtract(penalty).doubleValue();

        if (verbose) {
            System.out.println("Classifier accuracy = " + classifierAccuracy.toString());
            System.out.println("Penalty value = " + penalty.toString());
            System.out.println("Fitness accuracy = " + fitness.toString());
        }
        fitnessCache.put(key, fitness);
        return fitness;

    }


    private void calculateFitness(Instances data, Map<ConcreteBinarySolution, Double> fitnessMap) throws Exception {
        //RandomWalkSampler sampler = new RandomWalkSampler(new UniformSampleMutator(), data.numAttributes(), 0.01);
        SolutionSampler<boolean[]> sampler = new ExhaustiveSampler(data.numAttributes() - 1);
        boolean[] point;

        do {
            point = sampler.getSample();

            ConcreteBinarySolution solution = (ConcreteBinarySolution) ConcreteBinarySolution.constructBinarySolution(point);

            if (fitnessMap.get(solution) != null) {
                System.err.println("Found same solution");
                continue;
            }

            fitnessMap.put(solution, getQuality(point, data));
            // optional
            sampler.showProgress();
        }
        while (!sampler.isDone());
    }

    private void calculateFitnessAsync(Instances data, Map<ConcreteBinarySolution, Double> fitnessMap) {
        SolutionSampler<boolean[]> sampler = new ExhaustiveSampler(data.numAttributes() - 1);
        List<List<boolean[]>> listOfBatchSolutions = Lists.newArrayList();
        for (int i = 0; i < NUMBER_OF_THREADS; i++) {
            listOfBatchSolutions.add(((ExhaustiveSampler) sampler).getBatchSample((int) (Math.pow(2.0, (double) data.numAttributes() - 1) / NUMBER_OF_THREADS) + 1));
        }
        for (List<boolean[]> points : listOfBatchSolutions) {
            es.execute(() -> {
                for (boolean[] point : points) {
                    ConcreteBinarySolution solution = (ConcreteBinarySolution) ConcreteBinarySolution.constructBinarySolution(point);

                    if (fitnessMap.get(solution) != null) {
                        System.err.println("Found same solution");
                        continue;
                    }
                    try {
                        fitnessMap.put(solution, getQuality(point, data));
                    } catch (Exception e) {
                        System.err.println(e.getMessage());
                    }
                }
            });
        }
    }

    private boolean wait(int minutes) throws InterruptedException {
        es.shutdown();
        return es.awaitTermination(minutes, TimeUnit.MINUTES);
    }

    protected Map<ConcreteBinarySolution, Double> sampleLandscape(Instances data) throws Exception {
        HashMap<ConcreteBinarySolution, Double> fitnessMap = new HashMap<>();

        if (multiThread) {
            calculateFitnessAsync(data, fitnessMap);
            this.wait(60);
        } else {
            calculateFitness(data, fitnessMap);
        }

        return fitnessMap;
    }

    public Map<ConcreteBinarySolution, Double> eval(Instances data) throws Exception {
        if (data.classIndex() == -1) {
            throw new DataException("Class index for data set is not specified.");
        }

        if (data.numAttributes() > 32) {
            throw new DataException("Data sets with more than 31 attributes not currently supported.");
        }


        //At this point it will be a randomly initialised point - initial point

        Map<ConcreteBinarySolution, Double> fitnessMap;

        fitnessMap = sampleLandscape(data);

        System.out.println("Populated fitness landscape, samples: " + fitnessMap.size());
        return fitnessMap;

    }

    public Map<String, Double> getFitnessCache() {
        return fitnessCache;
    }
}
