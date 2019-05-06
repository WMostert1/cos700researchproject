import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.lazy.IBk;
import weka.core.Instances;

import java.math.BigDecimal;
import java.math.RoundingMode;

public class NeuralNetClassifier implements IClassify {
    private int repeats;

    public NeuralNetClassifier(){
        this(1);
    }

    public NeuralNetClassifier(int repeats){
        this.repeats = repeats;
    }

    @Override
    public double getClassificationAccuracy(Instances training, Instances testing) throws Exception {
        Classifier neuralNet = new MultilayerPerceptron();

        //This is to make sure that NO feature included has a really bad fitness
        if(training.classIndex() == 0 || testing.classIndex() == 0){
            return Double.MIN_VALUE;
        }

        if(training.classIndex() < 1 || testing.classIndex() < 1){
            throw new RuntimeException("Class index not set.");
        }

        DescriptiveStatistics descriptiveStatistics = new DescriptiveStatistics();

        for(int i = 0; i < repeats; i++){

        neuralNet.buildClassifier(training);

        Evaluation eval = new Evaluation(training);
        eval.evaluateModel(neuralNet, testing);

        descriptiveStatistics.addValue(eval.kappa());
        }

        BigDecimal bd = new BigDecimal(descriptiveStatistics.getMean());
        bd = bd.setScale(EvoCOPPaperApplication.DECIMAL_PLACES, RoundingMode.HALF_UP);
        return bd.doubleValue();
    }
}
