import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemovePercentage;

/**
 * Created by bbdnet1339 on 2016/08/08.
 *
 */
public class InstanceSplitter {
    private double trainingPercentage;
    private Instances data;

    public InstanceSplitter(Instances inst,double trainingPercentage) throws Exception {
        this.trainingPercentage = trainingPercentage;
        this.data = inst;
    }

    public Instances getTrainingSet() throws Exception {
        try {
            RemovePercentage percentageFilter = new RemovePercentage();
            percentageFilter.setInvertSelection(true);
            percentageFilter.setPercentage(this.trainingPercentage);
            percentageFilter.setInputFormat(data);
            Instances trainingData = Filter.useFilter(data, percentageFilter);
            if (trainingData.classIndex() == -1)
                trainingData.setClassIndex(trainingData.numAttributes() - 1);
            return trainingData;
        }catch (Exception e){
            return null;
        }
    }

    public Instances getTestingSet() throws Exception {
        RemovePercentage percentageFilter = new RemovePercentage();
        percentageFilter.setPercentage(this.trainingPercentage);
        percentageFilter.setInputFormat(data);
        Instances testingData = Filter.useFilter(data,percentageFilter);
        if (testingData.classIndex() == -1)
            testingData.setClassIndex(testingData.numAttributes() - 1);
        return testingData;
    }
}
