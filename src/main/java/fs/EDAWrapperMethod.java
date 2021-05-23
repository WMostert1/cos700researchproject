package fs;

import com.google.common.collect.Lists;
import fitness.FitnessEvaluator;
import lons.examples.ConcreteBinarySolution;
import org.javatuples.Pair;
import utils.MathUtils;
import weka.core.Instances;

import java.math.BigDecimal;
import java.security.SecureRandom;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

public class EDAWrapperMethod extends StochasticFeatureSelection implements FeatureSelectionAlgorithm {

    private double selectionPercentage = 0.4;
    private int populationSize = 20; // TODO: Consider this again
    private SecureRandom random = new SecureRandom();
    private int numberOfIterations = 100;
    private int numberOfRuns = 30;

    @Override
    public FeatureSelectionResult apply(final Instances data, final FitnessEvaluator fitnessEvaluator) throws Exception {
        //initialise the population to random bit strings
        List<Pair<boolean[], BigDecimal>> currentPopulation = Lists.newArrayList();
        for (int i = 0; i < populationSize; i++) {
            boolean[] randomSolution = getRandomPoint(data.numAttributes() - 1);
            BigDecimal fitness = MathUtils.doubleToBigDecimal(fitnessEvaluator.getQuality(randomSolution, data));
            currentPopulation.add(new Pair<>(randomSolution, fitness));
        }
        List<FeatureSelectionResult> runResults = Lists.newArrayList();
        for (int RUN = 0; RUN < numberOfRuns; RUN++) {
            int iterationNo = 0;
            do {
                currentPopulation = getSelectionFromPopulation(currentPopulation);
                // build the probability model
                BigDecimal[] probabilityModel = new BigDecimal[data.numAttributes() - 1];
                Arrays.fill(probabilityModel, BigDecimal.ZERO);
                for (Pair<boolean[], BigDecimal> solution : currentPopulation) {
                    for (int i = 0; i < probabilityModel.length; i++) {
                        if (solution.getValue0()[i]) {
                            probabilityModel[i] = probabilityModel[i].add(BigDecimal.ONE);
                        }
                    }
                }
                for (int i = 0; i < probabilityModel.length; i++) {
                    probabilityModel[i] = probabilityModel[i].divide(BigDecimal.valueOf(currentPopulation.size()), MathUtils.ROUNDING_MODE);
                }

                while (currentPopulation.size() != populationSize) {
                    boolean[] solutionFromProbabilityModel = getSolutionFromProbabilityModel(probabilityModel);
                    BigDecimal fitness = MathUtils.doubleToBigDecimal(fitnessEvaluator.getQuality(solutionFromProbabilityModel, data));
                    currentPopulation.add(new Pair<>(solutionFromProbabilityModel, fitness));
                }
            } while (++iterationNo < numberOfIterations && !testHomogeneousPopulation(currentPopulation));

            currentPopulation.sort(Comparator.comparing(Pair::getValue1));
            Collections.reverse(currentPopulation);
            Pair<boolean[], BigDecimal> bestIndividual = currentPopulation.get(0);
            FeatureSelectionResult featureSelectionResult = new FeatureSelectionResult(data,
                    bestIndividual.getValue1().doubleValue(),
                    ConcreteBinarySolution.constructBinarySolution(bestIndividual.getValue0()));
            runResults.add(featureSelectionResult);
        }

        double meanAccuracy = 0.0;


        for(FeatureSelectionResult bundle : runResults){
            meanAccuracy += bundle.getAccuracy();
        }

        meanAccuracy /= numberOfRuns;

        return new FeatureSelectionResult(data, meanAccuracy, null);
    }

    private List<Pair<boolean[], BigDecimal>> getSelectionFromPopulation(List<Pair<boolean[], BigDecimal>> currentPopulation) {
        // sort the population by fitness value
        currentPopulation.sort(Comparator.comparing(Pair::getValue1));
        Collections.reverse(currentPopulation);
        int cutOffSize = (int) (selectionPercentage * populationSize) - 1;
        while (currentPopulation.size() != cutOffSize) {
            currentPopulation.remove(currentPopulation.size() - 1);
        }
        return currentPopulation;
    }

    @Override
    public String getAlgorithmName() {
        return "EDA";
    }

    public boolean[] getSolutionFromProbabilityModel(BigDecimal[] probabilityModel) {
        boolean[] solution = new boolean[probabilityModel.length];
        for (int i = 0; i < solution.length; i++) {
            solution[i] = MathUtils.doubleToBigDecimal(random.nextDouble()).compareTo(probabilityModel[i]) <= 0;
        }
        return solution;
    }

    public boolean[] getRandomPoint(int size) {
        boolean[] arr = new boolean[size];
        for (int i = 0; i < size; i++) {
            arr[i] = random.nextBoolean();
        }
        return arr;
    }

    private boolean testHomogeneousPopulation(List<Pair<boolean[], BigDecimal>> population) {
        boolean[] testSolution = population.get(0).getValue0();
        for (int i = 1; i < population.size(); i++) {
            if (!Arrays.equals(testSolution, population.get(i).getValue0())) {
                return false;
            }
        }
        return true;
    }

    @Override
    public ArrayList<BigDecimal> getIterationFitnessValues() {
        return null;
    }
}
