package lons.examples;

import lons.Problem;

import java.security.SecureRandom;
import java.util.Map;

public class FSBinaryProblem implements BinaryProblem {
    Map<ConcreteBinarySolution, Double> fitnessMap;
    private static final  SecureRandom RANDOM = new SecureRandom();
    private BinarySolution[] keys;


    public FSBinaryProblem(Map<ConcreteBinarySolution, Double> fitnessMap) {
        this.fitnessMap = fitnessMap;
        this.keys = new BinarySolution[fitnessMap.size()];
        keys = fitnessMap.keySet().toArray(keys);

        System.out.println("Exhaustive solutions size: "+this.keys.length);
    }

    @Override
    public double getQuality(BinarySolution s) {
        return fitnessMap.get(s);
    }

    @Override
    public BinarySolution[] getExhaustiveSetOfSolutions() throws UnsupportedOperationException {
        return keys;
    }

    @Override
    public BinarySolution getRandomSolution() {
        int size = fitnessMap.size();
        int item = RANDOM.nextInt(size); // In real life, the Random object should be rather more shared than this
        return ConcreteBinarySolution.constructBinarySolution((boolean[]) fitnessMap.keySet().toArray()[item]);
    }


}
