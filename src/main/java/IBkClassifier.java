import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.core.Instances;

import java.math.BigDecimal;
import java.math.RoundingMode;

/**
 * Created by bbdnet1339 on 2016/08/08.
 */
public class IBkClassifier implements IClassify {

    //Returned as the kappa statistic
    @Override
    public double getClassificationAccuracy(Instances training, Instances testing) throws Exception {
        Classifier ibk = new IBk();
        String[] options = weka.core.Utils.splitOptions("-K 3 -W 0 -A \"weka.core.neighboursearch.LinearNNSearch -A \\\"weka.core.EuclideanDistance -R first-last\\\"\"");
        ibk.setOptions(options);

        //This is to make sure that NO feature included has a really bad fitness
        if(training.classIndex() == 0 || testing.classIndex() ==0){
            return Double.MIN_VALUE;
        }

        if(training.classIndex() < 1 || testing.classIndex() < 1){
            throw new RuntimeException("Class index not set.");
        }

        ibk.buildClassifier(training);

        Evaluation eval = new Evaluation(training);
        eval.evaluateModel(ibk, testing);
        BigDecimal bd = new BigDecimal(eval.kappa());
        bd = bd.setScale(EvoCOPPaperApplication.DECIMAL_PLACES, RoundingMode.HALF_UP);
        return bd.doubleValue();
    }


}
