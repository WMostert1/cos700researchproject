import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.common.base.Stopwatch;
import com.google.common.collect.Lists;
import exceptions.DataException;
import lons.examples.BinarySolution;
import lons.examples.ConcreteBinarySolution;
import mutators.BitMutator;
import mutators.UniformBitMutator;
import mutators.UniformSampleMutator;
import samplers.ExhaustiveSampler;
import samplers.RandomWalkSampler;
import samplers.SolutionSampler;
import weka.core.Instances;

import javax.xml.crypto.Data;
import java.io.PrintWriter;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

/**
 * Created by bbdnet1339 on 2016/08/08.
 *
 */
public class LandscapeEvaluator {


    private IClassify classifier;
    public static final double trainingPercentage = 50.0;
    private static final int NUMBER_OF_THREADS = 5;
    private ExecutorService es = Executors.newCachedThreadPool();
    private boolean multiThread = false;

    public LandscapeEvaluator(IClassify classifier){

        this.classifier = classifier;
    }

    public LandscapeEvaluator(IClassify classifier, boolean multiThread){
        this.multiThread = multiThread;
        this.classifier = classifier;

    }

    public Double getQuality(boolean [] solution, Instances data) throws Exception {
        Instances currentInstances = FeatureSelectorUtils.getInstancesFromBitString(data, solution);
        DataSetInstanceSplitter splitter = new DataSetInstanceSplitter(currentInstances,trainingPercentage);
        Instances trainingSet = splitter.getTrainingSet();
        Instances testingSet = splitter.getTestingSet();

        return classifier.getClassificationAccuracy(trainingSet, testingSet);
    }

    private void calculateFitness(Instances data, Map<ConcreteBinarySolution, Double> fitnessMap) throws Exception {
        //RandomWalkSampler sampler = new RandomWalkSampler(new UniformSampleMutator(), data.numAttributes(), 0.01);
        SolutionSampler<boolean []> sampler = new ExhaustiveSampler(data.numAttributes()-1);
        boolean [] point;

        do{
            point = sampler.getSample();

            ConcreteBinarySolution solution = (ConcreteBinarySolution) ConcreteBinarySolution.constructBinarySolution(point);

            if(fitnessMap.get(solution) != null){
                System.err.println("Found same solution");
                continue;
            }

            fitnessMap.put(solution, getQuality(point, data));
           // optional
            sampler.showProgress();
        }
        while(!sampler.isDone());
    }

    private void calculateFitnessAsync(Instances data, Map<ConcreteBinarySolution, Double> fitnessMap) {
        SolutionSampler<boolean []> sampler = new ExhaustiveSampler(data.numAttributes()-1);
        List<List<boolean[]>> listOfBatchSolutions = Lists.newArrayList();
        for(int i = 0 ; i < NUMBER_OF_THREADS; i++){
            listOfBatchSolutions.add(((ExhaustiveSampler) sampler).getBatchSample((int)(Math.pow(2.0, (double)data.numAttributes()-1)/NUMBER_OF_THREADS)+1));
        }
        for(List<boolean[]> points : listOfBatchSolutions) {
            es.execute(() -> {
                for (boolean[] point : points) {
                    ConcreteBinarySolution solution = (ConcreteBinarySolution) ConcreteBinarySolution.constructBinarySolution(point);

                    if (fitnessMap.get(solution) != null) {
                        System.err.println("Found same solution");
                        continue;
                    }
                    try {
                        fitnessMap.put(solution, getQuality(point, data));
                    } catch (Exception e) {
                        System.err.println(e.getMessage());
                    }
                }
            });
        }
    }

    private boolean wait(int minutes) throws InterruptedException {
        es.shutdown();
        return es.awaitTermination(minutes, TimeUnit.MINUTES);
    }

    protected Map<ConcreteBinarySolution, Double> sampleLandscape(Instances data) throws Exception {
        HashMap<ConcreteBinarySolution, Double> fitnessMap = new HashMap<>();

        if(multiThread){
            calculateFitnessAsync(data, fitnessMap);
            this.wait(60);
        }else{
            calculateFitness(data, fitnessMap);
        }

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
