import com.fasterxml.jackson.databind.ObjectMapper;
import exceptions.DataException;
import lons.examples.BinarySolution;
import lons.examples.ConcreteBinarySolution;
import mutators.BitMutator;
import samplers.ExhaustiveSampler;
import samplers.RandomWalkSampler;
import samplers.SolutionSampler;
import weka.core.Instances;

import javax.xml.crypto.Data;
import java.io.PrintWriter;
import java.util.HashMap;
import java.util.Map;

/**
 * Created by bbdnet1339 on 2016/08/08.
 *
 */
public class LandscapeEvaluator {


    private IClassify classifier;
    public static final double trainingPercentage = 50.0;

    public LandscapeEvaluator(IClassify classifier){

        this.classifier = classifier;

    }

    private Double getQuality(boolean [] solution, Instances data) throws Exception {
        Instances currentInstances = FeatureSelectorUtils.getInstancesFromBitString(data, solution);
        DataSetInstanceSplitter splitter = new DataSetInstanceSplitter(currentInstances,trainingPercentage);
        Instances trainingSet = splitter.getTrainingSet();
        Instances testingSet = splitter.getTestingSet();

        return classifier.getClassificationAccuracy(trainingSet, testingSet);
    }

    protected Map<ConcreteBinarySolution, Double> sampleLandscape(Instances data) throws Exception {
        HashMap<ConcreteBinarySolution, Double> fitnessMap = new HashMap<>();



        //RandomWalkSampler sampler = new RandomWalkSampler(mutator, data.numAttributes(), stepMultiplier);
        SolutionSampler<boolean []> sampler = new ExhaustiveSampler(data.numAttributes()-1);
        boolean [] point;
        do{
            point = sampler.getSample();
            ConcreteBinarySolution solution = (ConcreteBinarySolution) ConcreteBinarySolution.constructBinarySolution(point);

            if(fitnessMap.get(solution) != null){
                System.out.println("Found same solution");
                continue;
            }

            fitnessMap.put(solution, getQuality(point, data));
            sampler.showProgress();
        }
        while(!sampler.isDone());
//
//        try (PrintWriter out = new PrintWriter("./"+dataSetName +".fitness.db")) {
//            System.out.println("Saving fitness DB for "+dataSetName);
//            out.print(mapper.writeValueAsString(fitnessMap));
//            System.out.println("Saved fitness DB for "+dataSetName);
//        }

        return fitnessMap;
    }

    public Map<ConcreteBinarySolution, Double> eval(Instances data) throws Exception {
        if(data.classIndex() == -1){
            throw new DataException("Class index for data set is not specified.");
        }

        if(data.numAttributes() > 32){
            throw new DataException("Data sets with more than 31 attributes not currently supported.");
        }



        //At this point it will be a randomly initialised point - initial point

        Map<ConcreteBinarySolution, Double> fitnessMap;

        fitnessMap = sampleLandscape(data);

        System.out.println("Populated fitness landscape, samples: "+fitnessMap.size());
        return fitnessMap;

    }

}
