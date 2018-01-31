import weka.core.Instances;

import java.util.HashMap;
import java.util.Map;

/**
 * Created by bbdnet1339 on 2016/08/08.
 *
 */
public class LandscapeEvaluator {
    private BitMutator mutator;
    private int stepMultiplier;
    private OutputFormatter outF;
    
    private IClassify classifier;
    public double trainingPercentage = 66.0;

    public LandscapeEvaluator(int stepMultiplier, BitMutator mutator, IClassify classifier, String dataSetName, OutputFormatter outputFormatter){
        this.stepMultiplier = stepMultiplier;
        this.mutator = mutator;
        this.classifier = classifier;
       this.outF = outputFormatter;
    }

    public void eval(Instances data) throws Exception {
        outF.addAsColumns(new String[]{"Number of Attributes for Data Set: ", Integer.toString(data.numAttributes())});
        outF.nextRow();

        outF.addAsColumns(new String[]{"Number of Instances in Data Set: ",Integer.toString(data.numInstances())});
        outF.nextRow();


        if(data.classIndex() == -1){
            data.setClassIndex(data.numAttributes()-1);
        }
        outF.addAsColumns(new String[]{"Number of Classes in Data Set: ",Integer.toString(data.numClasses())});
        outF.nextRow();


        //At this point it will be a randomly initialised point - initial point
        boolean [] point = mutator.getRandomPoint(data.numAttributes());
        Map<boolean [], Double> history = new HashMap<>();


        long steps = stepMultiplier*data.numAttributes();
        System.out.println("Doing "+steps+" steps.");

        outF.addAsColumns(new String[]{"Sample size: ",Long.toString(steps)});
        outF.nextRow();

        outF.addEmptyRow();


        int percent = 1;
        for(long i = 0; i < steps; i++){
            if((i+ 1)/((double)steps) > percent/100.0){
                percent++;
                System.out.println(percent+"%");
            }
            Instances currentInstances = FeatureSelectorUtils.getInstancesFromBitString(data,point);
            DataSetInstanceSplitter splitter = new DataSetInstanceSplitter(currentInstances,trainingPercentage);
            Instances trainingSet = splitter.getTrainingSet();
            Instances testingSet = splitter.getTestingSet();

            history.put(point, classifier.getClassificationAccuracy(trainingSet, testingSet));
            point = mutator.mutate(point);
        }

//        history.forEach((x,y)->{
//            System.out.println(FeatureSelectorUtils.booleanArrayToBitString(x));
//            System.out.println(Double.toString(y));
//            System.out.println("----------------");
//        });

        HDILMeasure hdilM = new HDILMeasure(20,-1.0,1.0,outF);
        hdilM.get(history);
    }

}
