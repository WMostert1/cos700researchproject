package lons.examples;

import lons.Neighbourhood;
import java.util.HashMap;

/**
 * BinaryHammingNeighbourhood.  
 * 
 * @author Jonathan Fieldsend
 * @version 11/07/2018
 */
public class BinaryHammingNeighbourhood implements Neighbourhood<BinarySolution>
{
    private int maxDistance=2;
    private static HashMap<Integer,boolean[][]> allBitFlipMasks = new HashMap<>();
    
    /**
     * Sets max hamming distance used in this neighbour for distance neighbour calls. 
     * Minimum 1-distance neighbour.
     */
    @Override
    public void setMaxDistance(int maxDistance) {
        this.maxDistance = maxDistance > 0 ? maxDistance : 1;
        allBitFlipMasks.clear();
    }
    
    @Override
    public BinarySolution[] neighbouringSolutions(BinarySolution s){
        boolean[][] bitStringNeighbours = getSingleBitFlipNeighbours(s);
        return generateNeighbours(bitStringNeighbours);
    }
    
    @Override
    public BinarySolution[] neighbouringSolutionsUpToDistance(BinarySolution s){
        boolean[][] bitStringNeighbours = getBitStringNeighboursUpToDistance(s, maxDistance);
        return generateNeighbours(bitStringNeighbours);
    }
    
    private static BinarySolution[] generateNeighbours(boolean[][] bitStringNeighbours) {
        BinarySolution[] neighbours = new BinarySolution[bitStringNeighbours.length];
        
        for (int i=0; i<neighbours.length; i++) {
            neighbours[i] = ConcreteBinarySolution.constructBinarySolution(bitStringNeighbours[i]);
        }
        return neighbours;
    }
    
    private static boolean[][] getSingleBitFlipNeighbours(BinarySolution s) {
        boolean[] x = s.getDesignVector();
        boolean[][] allNeighbours = new boolean[x.length][x.length];
        for  (int i=0; i < x.length; i++)
            for  (int j=0; j < x.length; j++)
                allNeighbours[i][j] = (i==j) ? ! x[j] : x[j];
        return allNeighbours;    
    }
        
    private static boolean[][] getBitStringNeighboursUpToDistance(BinarySolution s, int maxBitFlips) {
        boolean[] x = s.getDesignVector();
        boolean[][] allMasks;
        if (!allBitFlipMasks.containsKey(x.length)) {
            System.out.println("Generating masks");
            int totalNumberOfNeighbours = x.length;
            int neighboursAtPreviousK = x.length;
            for (int k=1; k<maxBitFlips; k++) {
                neighboursAtPreviousK = neighboursAtPreviousK*(x.length-k)/(k+1);
                totalNumberOfNeighbours += neighboursAtPreviousK;
            }
            System.out.print(" neighbour num: " + totalNumberOfNeighbours);
            allMasks = new boolean[totalNumberOfNeighbours][x.length];
            int index = 0;
            for (int k=1; k <= maxBitFlips; k++) {
                index = flippedBitsCombinations(k, allMasks, x.length, index);
            }
            allBitFlipMasks.put(x.length, allMasks);
        } else {
            allMasks =  allBitFlipMasks.get(x.length);
        }
        boolean[][] allneighbours = new boolean[allMasks.length][x.length];
        for (int i=0; i < allMasks.length; i++)
            for (int k=0; k < x.length; k++)
                allneighbours[i][k] = allMasks[i][k] ^ x[k];
        return allneighbours;
    }
    
    
    
    private static int flippedBitsCombinations(int numberOfBitsToFlip, boolean[][] flippedCombinations, int n, int index) {
        if (n < numberOfBitsToFlip)
            numberOfBitsToFlip = n;
        boolean[] bitMask = new boolean[n];    
        for (int i=0; i<n; i++) // initialise mask
            bitMask[i] = i<numberOfBitsToFlip ? true : false;
        do {
            boolean[] copyBitMask = new boolean[bitMask.length];
            for (int i=0; i<bitMask.length; i++)
                copyBitMask[i] = bitMask[i];
            flippedCombinations[index++] = copyBitMask;
        } while (lexicographicIncrement(bitMask));
        return index;
    }
    
    private static boolean lexicographicIncrement(boolean[] bitMask) {
        boolean maximum = true;
        int position = bitMask.length;
        int tracker = 1;
        while ((--position > -1) && (!bitMask[position] || maximum )){
            if (bitMask[position])  
                ++tracker;
            else
                maximum = false;
        }
        if (position < 0)
            return false;
        bitMask[position] = false;
        while (++position < bitMask.length)
            bitMask[position] = tracker-- > 0 ? true : false;
        return true;    
    }
    
    
    private static boolean[] flipWithMask(boolean[] x, boolean[] mask) {
        boolean[] result = new boolean[x.length];
        for (int i=0; i<x.length; i++) {
            result[i] = mask[i] ^ x[i]; //mask[i] ? ! x[i] : x[i];
        }
        return result;
    }
    
    
//     private int[] getIndices(int length, int numberOfFlips, int index) {
//         int[] toFlip = new int[numberOfFlips];
//         
//     }
//     
//     @Override
//     public int[] indicesOfNeighbouringSolutions(BinarySolution s){
//         int[] neighbouringIndices = new int[s.getNumberOfElements()];
//         boolean[][] bitStringNeighbours = getBitStringNeighbours(s);
//         for (int i=0; i<neighbouringIndices.length; i++) {
//             neighbouringIndices[i] = BinaryProblem.getIndexOfBitString(bitStringNeighbours[i]);
//         }
//         return neighbouringIndices;
//     }
//     
}
