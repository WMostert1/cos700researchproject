package lons.examples;


import java.util.ArrayList;
import java.util.Random;
/**
 * NKModelProblem represents tunable NK model problems.
 * 
 * @author Jonathan Fieldsend
 * @version 11/07/2018
 */
public class NKModelProblem implements BinaryProblem
{
    final int n;
    final int k;
    private double[] randomQualityMapping;
    private int[][] randomIndexMapping;
    private static Random rng = new Random();
    static int numberOfFunctionCalls;
    /**
     * If k is entered > n-1, then will automatically be set at n-1. Also, 
     * k automatically limited to a maximum of 30 (due to limit of integer size in Java)
     */
  
    NKModelProblem(int n, int k){
        this.n = n;
        if (k<0) k=0;
        if (k>30) k=30;
        this.k = (k > n-1) ? n-1 : k;
            
        setRandomQualityMapping(BinaryProblem.bitPowers[k+2]-1); 
        setRandomIndexMapping(n,k);
    }
    
    /**
     * Limited to 
     */
    
    @Override
    public BinarySolution[] getExhaustiveSetOfSolutions() {
        BinarySolution[] designs = new BinarySolution[BinaryProblem.getBitPower(n)];
        for (int i=0; i< designs.length; i++) {
             designs[i] = ConcreteBinarySolution.constructBinarySolution(BinaryProblem.getBitStringCorrespondingToIndex(i,n));
        }
        return designs;
    }
    
    @Override
    public BinarySolution getRandomSolution() {
        boolean[] design = new boolean[n];
        for (int i=0; i<n; i++)
            design[i] = rng.nextBoolean();
        return ConcreteBinarySolution.constructBinarySolution(design);
    }
    
    private void setRandomQualityMapping(int maxIndex) {
        randomQualityMapping = new double[maxIndex];
        // assign random fitness to all possible 2^(k+1) bitstrings
        for (int i=0; i<maxIndex; i++) 
            randomQualityMapping[i]=rng.nextDouble();
    }
    
    private void setRandomIndexMapping(int n, int k) {
        randomIndexMapping = new int[n][k];
        ArrayList<Integer> indices = new ArrayList<Integer>(n);
        for (int i=0; i<n; i++)
            indices.add(i);
        for (int i=0; i<n; i++) {
            // randomly permute the indices
            java.util.Collections.shuffle(indices,rng);
            int offset = 0;
            for (int j=0; j<k; j++) {
                if (indices.get(j)==i) // need to make sure not linked to itself
                    offset = 1;
                randomIndexMapping[i][j] = indices.get(j+offset);
            }
        }
    }
    
    @Override
    public double getQuality(BinarySolution s) {
        double fitness = 0.0;
        for (int i=0; i<n; i++)
            fitness += randomQualityMapping[getCorrespondingIndex(i,s.getDesignVector())];
        NKModelProblem.numberOfFunctionCalls++; //instrumentation for comprarison between approaches
        return fitness;
    }
    
    private int getCorrespondingIndex(int startIndex, boolean[] x) {
        int index = x[startIndex] ? 1 : 0;
        for (int j=0; j<k; j++)
            index += x[randomIndexMapping[startIndex][j]] ? bitPowers[j+1] : 0;
        
        return index;
    }
    
    static void setSeed(int seed) {
        rng.setSeed(seed);
    }
    
}
