package fs;

import com.google.common.collect.Lists;
import fitness.FitnessEvaluator;
import fs.pso.*;
import lons.examples.ConcreteBinarySolution;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import weka.attributeSelection.Ranker;
import weka.attributeSelection.SymmetricalUncertAttributeEval;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

public class AMSO  extends StochasticFeatureSelection implements FeatureSelectionAlgorithm {
    private int numberOfSubSwarms = 3;
    private int populationSize = 50;
    private BigDecimal c = BigDecimal.valueOf(1.49445);
    private int numberOfIteration = 100;
    int subSwarmSize = populationSize/numberOfSubSwarms;
    int betaSubswarmUpdatingTHreshold = 7;
    private static final int NO_RUNS = 30;
    private ArrayList<BigDecimal> iterationFitnessValues = Lists.newArrayList();

    @Override
    public FeatureSelectionResult apply(Instances data, FitnessEvaluator fitnessEvaluator) throws Exception {
        iterationFitnessValues.clear();
        //Start by ranking features based on symmetrical uncertainty
        SymmetricalUncertAttributeEval symmetricalUncertAttributeEval = new SymmetricalUncertAttributeEval();
        symmetricalUncertAttributeEval.buildEvaluator(data);
        Ranker ranker = new Ranker();
        ranker.search(symmetricalUncertAttributeEval, data);
        double [][] ranked = ranker.rankedAttributes();
        int numberOfAttributes = data.numAttributes() - 1;

        DescriptiveStatistics stats = new DescriptiveStatistics();

        List<FeatureSelectionResult> fsResults = Lists.newArrayList();
        FeatureSelectionResult bestFsResult = null;

        for(int runNo = 0; runNo < NO_RUNS; runNo++) {
            ArrayList<AMSOSubswarm> subSwarms = Lists.newArrayList();

            for (int i = 0; i < numberOfSubSwarms; i++) {
                //Contains only features relevant to the sub swarm
                int[] columnsToRemove = getColumnIndicesToRemove(data, ranked, numberOfAttributes, i + 1);
                Instances subPopulationFeatures = getSubSetFeaturesBasedOnSubPoluation(data, columnsToRemove);
                //Subswarm size
                ParticleFitnessCalculator fitnessCalculator = new KnnParticleFitnessCalculator(fitnessEvaluator, subPopulationFeatures);
                subSwarms.add(new AMSOSubswarm(fitnessCalculator,
                        numberOfAttributes,
                        subSwarmSize,
                        BigDecimal.ZERO,
                        BigDecimal.ONE,
                        numberOfIteration,
                        c,
                        columnsToRemove));
            }

            int countSinceLastIncrease = 0;
            Particle previousGbest = null;
            for (int iteration = 0; iteration < numberOfIteration; iteration++) {

                ArrayList<Particle> subswarmGbests = Lists.newArrayList();
                for (AMSOSubswarm subSwarm : subSwarms) {
                    subswarmGbests.add(subSwarm.optimize());
                }

                //Sort descending
                subswarmGbests.sort((o1, o2) -> -1 * o1.getFitness().compareTo(o2.getFitness()));

                Particle gbestParticle = subswarmGbests.get(0);

                if (previousGbest == null) {
                    previousGbest = gbestParticle.clone();
                } else if (gbestParticle.getFitness().compareTo(previousGbest.getFitness()) > 0) {
                    countSinceLastIncrease = 0;
                    previousGbest = gbestParticle.clone();
                } else {
                    countSinceLastIncrease++;
                }
                //Do subswarm updating
                if (countSinceLastIncrease >= betaSubswarmUpdatingTHreshold) {
                    int newMaxLength = previousGbest.getEnabledDimensions();


                    subSwarms.sort(Comparator.comparingInt(AMSOSubswarm::getEnabledDimensionsLength));
                    for (int subSwarmNo = 0; subSwarmNo < numberOfSubSwarms; subSwarmNo++) {
                        AMSOSubswarm subswarm = subSwarms.get(subSwarmNo);
                        if (subswarm.getEnabledDimensionsLength() == newMaxLength) {
                            continue;
                        }

                        int[] columnsToRemove = getColumnIndicesToRemove(data, ranked, newMaxLength, subSwarmNo + 1);
                        Instances newData = getSubSetFeaturesBasedOnSubPoluation(data, columnsToRemove);
                        subSwarms.get(subSwarmNo).subswarmUpdate(columnsToRemove, new KnnParticleFitnessCalculator(fitnessEvaluator, newData), newData.numAttributes() - 1);
                    }

                }

            }

            FeatureSelectionResult fsr = new FeatureSelectionResult(data, previousGbest.getFitness().doubleValue(), ConcreteBinarySolution.constructBinarySolution(KnnParticleFitnessCalculator.dimensionsToBooleanArray(previousGbest)));
            iterationFitnessValues.add(previousGbest.getFitness());
            if(bestFsResult == null){
                bestFsResult = fsr;
            }else if(bestFsResult.getAccuracy() < fsr.getAccuracy()){
                bestFsResult = fsr;
            }
            fsResults.add(fsr);
            stats.addValue(fsr.getAccuracy());
        }


        return new FeatureSelectionResult(bestFsResult.getData(), stats.getMean(), bestFsResult.getBinarySolution());
    }


    private int [] getColumnIndicesToRemove(Instances data, double [][] rankedFeatures, int maxNumberOfFeatures, int subPopulationNumber){
        int numberOfFeaturesToUse = subPopulationNumber * (maxNumberOfFeatures / numberOfSubSwarms);
        if (numberOfFeaturesToUse > maxNumberOfFeatures){
            numberOfFeaturesToUse = maxNumberOfFeatures;
        }

        if(numberOfFeaturesToUse <= 1){
            //throw new RuntimeException("Too few features to use in this analysis for AMSO.");
            numberOfFeaturesToUse = 1;
        }

        int columnsToRemoveSize = data.numAttributes() - 1  - numberOfFeaturesToUse;

        int [] columnsToRemove = new int[columnsToRemoveSize];

        int removalColumnsIndex = 0;
        for (int i = numberOfFeaturesToUse; i < data.numAttributes()-1;i++){
            columnsToRemove[removalColumnsIndex++] = (int)rankedFeatures[i][0];
        }

        return columnsToRemove;
    }

    private Instances getSubSetFeaturesBasedOnSubPoluation(Instances data, int [] columnsToRemove ) throws Exception {
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

    @Override
    public String getAlgorithmName() {
        return "AMSO Feature Selection";
    }

    @Override
    public ArrayList<BigDecimal> getIterationFitnessValues() {
        return iterationFitnessValues;
    }
}
