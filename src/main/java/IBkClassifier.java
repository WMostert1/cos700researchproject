import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.core.Instances;

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
        ibk.buildClassifier(training);
        Evaluation eval = new Evaluation(training);
        eval.evaluateModel(ibk, testing);
        return eval.kappa();
    }


}
