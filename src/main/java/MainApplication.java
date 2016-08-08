import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemovePercentage;


/**
 * Created by bbdnet1339 on 2016/08/05.
 */
public class MainApplication {
    public static void main(String [] args) throws Exception {
        DataSource source = new DataSource(MainApplication.class.getResourceAsStream("data-sets/car.arff"));
        Instances data = source.getDataSet();

        RandomWalk walk = new RandomWalk(100,new UniformBitMutator(), new IBkClassifier());
        walk.doWalk(data);

    }
}
