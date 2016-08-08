import weka.core.Instances;

/**
 * Created by bbdnet1339 on 2016/08/08.
 */
public interface IClassify {
    double getClassificationAccuracy(Instances training, Instances testing) throws Exception;
}
