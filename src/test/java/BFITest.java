import classifiers.IBkClassifier;
import com.google.common.collect.Lists;
import com.google.common.primitives.Booleans;
import fitness.FitnessEvaluator;
import org.junit.Test;
import utils.DataSetInstanceSplitter;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.core.Instances;

import java.util.Arrays;
import java.util.List;

import static utils.GlobalConstants.TRAINING_PERCENTAGE;

public class BFITest {


    public BFITest() throws Exception {
    }

    private static <T> List<T> list(T... vals) {
        return Lists.newArrayList(vals);

    }

    @Test
    public void testCorrectBFI() throws Exception {
        List<List<Boolean>> solutionsToUse = list(
                list(true, true, true, true), //1.) all features
                list(false, false, true, false), //2.) f3 relevant feature
                list(false, true, false, false), //3.) f2 relevant feature
                list(false, true, true, false), //4.) Both relevant features
                list(false, true, false, true), //5.) One relevant, one irellevant
                list(true, false, false, false), // 6.)Only unique feature
                list(false, false, false, true), //7.) Only irellevant feature
                list(true, true, false, true) //9.)  One relevant two irtellevant
        );

        Instances data = JournalApplication.getDataSet("artificial_test.arff");
        FitnessEvaluator fitnessEvaluator = new FitnessEvaluator(new IBkClassifier(), data.numAttributes() - 1, true);
        Double baselineFitness = fitnessEvaluator.getQuality(new boolean[]{true, true, true, true}, data);



        int j = 1;
        for (List<Boolean> sol : solutionsToUse) {
            System.out.println("--------"+j+++"----------");
            boolean[] solution = Booleans.toArray(sol);


            data = JournalApplication.getDataSet("artificial_test.arff");

            DataSetInstanceSplitter splitter = new DataSetInstanceSplitter(data, TRAINING_PERCENTAGE);
            InfoGainAttributeEval infoGainAttributeEval = new InfoGainAttributeEval();
            infoGainAttributeEval.setMissingMerge(true);
            infoGainAttributeEval.buildEvaluator(splitter.getTrainingSet());

            infoGainAttributeEval.evaluateAttribute(0);

            fitnessEvaluator = new FitnessEvaluator(new IBkClassifier(), data.numAttributes() - 1, true);


            System.out.println("Using solution: " + Arrays.toString(solution));

            Double quality = fitnessEvaluator.getQuality(solution, data);
            System.out.println("BFI: " + (quality - baselineFitness));

        }
//
//        boolean[] solution = new boolean[]{true, true, false, true};
//
//        Instances data = JournalApplication.getDataSet("artificial_test.arff");
//        FitnessEvaluator fitnessEvaluator = new FitnessEvaluator(new IBkClassifier(), data.numAttributes() - 1, true);
//
//
//        System.out.println("Using solution: " + Arrays.toString(solution));
//
//        Double quality = fitnessEvaluator.getQuality(solution, data);
//        System.out.println(quality);
    }



    /*
        using all features, the fitness evaluator gets 0.0

     */

}
