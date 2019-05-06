import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;

import java.math.BigDecimal;
import java.math.RoundingMode;

public class J48Classifier implements IClassify {
    @Override
    public double getClassificationAccuracy(Instances training, Instances testing) throws Exception {
        Classifier j48 = new J48();


        //This is to make sure that NO feature included has a really bad fitness
        if(training.classIndex() == 0 || testing.classIndex() ==0){
            return Double.MIN_VALUE;
        }

        if(training.classIndex() < 1 || testing.classIndex() < 1){
            throw new RuntimeException("Class index not set.");
        }

        j48.buildClassifier(training);
        Evaluation eval = new Evaluation(training);
        eval.evaluateModel(j48, testing);
        BigDecimal bd = new BigDecimal(eval.kappa());
        bd = bd.setScale(EvoCOPPaperApplication.DECIMAL_PLACES, RoundingMode.HALF_UP);
        return bd.doubleValue();
    }
}
