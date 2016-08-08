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
    private int steps;

    private FeatureSelector fSelector;
    private IClassify classifier;
    public double trainingPercentage = 66.0;

    public RandomWalk(int steps, BitMutator mutator, IClassify classifier){
        this.steps = steps;
        this.mutator = mutator;
        this.classifier = classifier;
        fSelector = new FeatureSelector();
    }

    public void doWalk(Instances data) throws Exception {
        boolean [] point = mutator.getRandomPoint(data.numAttributes());
        Map<boolean [], Double> history = new HashMap<>();

        for(int i = 0; i < steps; i++){
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
    }

}
