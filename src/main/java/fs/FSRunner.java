package fs;

import fitness.FitnessEvaluator;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import utils.CsvOutputFormatter;
import utils.OutputFormatter;
import weka.core.Instances;

import java.io.File;
import java.math.BigDecimal;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Map;

import static utils.MathUtils.bigDecimalToString;
import static utils.MathUtils.doubleToBigDecimal;

public class FSRunner implements Runnable {

    public static final String RUN_DELIMINATOR = Instant.now().toString() + "|kc(" + FitnessEvaluator.classifierScalingConstant + ")|kp(" + FitnessEvaluator.penaltyScalingConstant + ")";

    private String dataSet;

    private FeatureSelectionAlgorithm fsa;

    private Instances originalData;

    private FitnessEvaluator fitnessEvaluator;

    private BigDecimal baselineFitness;

    private int dataSetNumber;

    private int algorithmNumber;

    private String[][] bfiTable;

    private String[][] fitnessTable;

    private OutputFormatter outF;

    private DescriptiveStatistics descriptiveStatistics;

    Map<String, ArrayList<BigDecimal>> statsValues;

    public FSRunner(final String dataSet, final FeatureSelectionAlgorithm fsa, final Instances originalData, final FitnessEvaluator fitnessEvaluator, final BigDecimal baselineFitness, final int dataSetNumber, final int algorithmNumber, final String[][] bfiTable, final String[][] fitnessTable, final OutputFormatter outF, final DescriptiveStatistics descriptiveStatistics, final Map<String, ArrayList<BigDecimal>> statsValues) {
        this.dataSet = dataSet;
        this.fsa = fsa;
        this.originalData = originalData;
        this.fitnessEvaluator = fitnessEvaluator;
        this.baselineFitness = baselineFitness;
        this.dataSetNumber = dataSetNumber;
        this.algorithmNumber = algorithmNumber;
        this.bfiTable = bfiTable;
        this.outF = outF;
        this.descriptiveStatistics = descriptiveStatistics;
        this.statsValues = statsValues;
        this.fitnessTable = fitnessTable;
    }

    private boolean isBFSOnLargeDataSet() {
        return (fsa instanceof SequentialBackwardSelection && originalData.numAttributes() > 1000);
    }

    @Override
    public void run() {
        try {
            System.out.println(dataSet + "[" + dataSetNumber + "] :: " + fsa.getAlgorithmName() + "[" + algorithmNumber + "] :: RUNNING");
            FeatureSelectionResult fsResults;
            if (!isBFSOnLargeDataSet()) {
                fsResults = fsa.apply(originalData, fitnessEvaluator);
            } else {
                fsResults = new FeatureSelectionResult(originalData, -1.0, null);
            }

            fitnessTable[dataSetNumber][1 + algorithmNumber] = bigDecimalToString(doubleToBigDecimal(fsResults.getAccuracy()));
            BigDecimal fitness = doubleToBigDecimal(fsResults.getAccuracy()).subtract(baselineFitness);
            descriptiveStatistics.addValue(fitness.doubleValue());

            outF.addAsColumns(dataSet, fsa.getAlgorithmName(), Double.toString(baselineFitness.doubleValue()),
                    Double.toString(fitness.doubleValue()));
            System.out.println(dataSet + " :: " + fsa.getAlgorithmName() + " :: COMPLETED (" + fitness.toString() + ")");
            bfiTable[dataSetNumber][algorithmNumber] = bigDecimalToString(fitness);
            //fitnessTable[dataSetNumber][algorithmNumber] = doubleToBigDecimal(fsResults.getAccuracy()).toString();
        } catch (Exception e) {
            bfiTable[dataSetNumber][algorithmNumber] = "ERR";
            // fitnessTable[dataSetNumber][algorithmNumber] = "ERR";
            System.err.println(e.getMessage());
        }

        if (fsa instanceof StochasticFeatureSelection) {
            ArrayList<BigDecimal> iterationValues = ((StochasticFeatureSelection) fsa).getIterationFitnessValues();

            DescriptiveStatistics stdDevStats = new DescriptiveStatistics();
            iterationValues.forEach((x) -> stdDevStats.addValue(x.doubleValue()));
            bfiTable[dataSetNumber][algorithmNumber] = bfiTable[dataSetNumber][algorithmNumber] + "±" + bigDecimalToString(doubleToBigDecimal(stdDevStats.getStandardDeviation()));

            File dsDirectory = new File("out/" + RUN_DELIMINATOR + "/fitness/" + dataSet);
            if (!dsDirectory.exists()) {
                dsDirectory.mkdir();
                // If you require it to make the entire directory path including parents,
                // use directory.mkdirs(); here instead.
            }
            OutputFormatter outputFormatter = new CsvOutputFormatter("out/" + RUN_DELIMINATOR + "/fitness/" + dataSet + "/" + fsa.getAlgorithmName().replace(" ", "") + ".csv");
            ((StochasticFeatureSelection) fsa).recordFitnessValues(outputFormatter, iterationValues, baselineFitness);
            if (!statsValues.containsKey(fsa.getAlgorithmName())) {
                statsValues.put(fsa.getAlgorithmName(), iterationValues);
            } else {
                statsValues.get(fsa.getAlgorithmName()).addAll(iterationValues);
            }
        }
    }
}
