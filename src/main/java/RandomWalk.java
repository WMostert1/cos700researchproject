import weka.core.Instance;
import weka.core.Instances;

import java.security.SecureRandom;
import java.util.HashMap;
import java.util.Map;

/**
 * Created by bbdnet1339 on 2016/08/08.
 *
 */
public class RandomWalk {
    private BitMutator mutator;
    private int stepMultiplier;
    private OutputFormatter outF;

    private FeatureSelector fSelector;
    private IClassify classifier;
    public double trainingPercentage = 66.0;

    public RandomWalk(int stepMultiplier, BitMutator mutator, IClassify classifier,String dataSetName){
        this.stepMultiplier = stepMultiplier;
        this.mutator = mutator;
        this.classifier = classifier;
        fSelector = new FeatureSelector();
        outF = new OutputFormatter(dataSetName+".csv");
    }

    public void doWalk(Instances data) throws Exception {
        outF.addAsColumns(new String[]{"Number of Attributes for Data Set: ", Integer.toString(data.numAttributes())});
        outF.nextRow();

        outF.addAsColumns(new String[]{"Number of Instances in Data Set: ",Integer.toString(data.numInstances())});
        outF.nextRow();


        if(data.classIndex() == -1){
            data.setClassIndex(data.numAttributes()-1);
        }
        outF.addAsColumns(new String[]{"Number of Classes in Data Set: ",Integer.toString(data.numClasses())});
        outF.nextRow();


        boolean [] point = mutator.getRandomPoint(data.numAttributes());
        Map<boolean [], Double> history = new HashMap<>();

        System.out.println("Binary string is of length "+point.length);


        System.out.println("Doing "+stepMultiplier*point.length+" steps.");

        long steps = stepMultiplier*data.numAttributes();


        outF.addAsColumns(new String[]{"Sample size: ",Long.toString(steps)});
        outF.nextRow();

        outF.addEmptyRow();
        for(long i = 0; i < steps; i++){
            System.out.println(i);
            Instances currentInstances = fSelector.getInstancesFromBitString(data,point);
            InstanceSplitter splitter = new InstanceSplitter(currentInstances,trainingPercentage);
            history.put(point, classifier.getClassificationAccuracy(splitter.getTrainingSet(),splitter.getTestingSet()));
            point = mutator.mutate(point);
        }

        history.forEach((x,y)->{
            System.out.println(fSelector.booleanArrayToBitString(x));
            System.out.println(Double.toString(y));
            System.out.println("----------------");
        });

        HDILMeasure hdilM = new HDILMeasure(10,-1.0,1.0,outF);
        hdilM.get(history);
    }

}
