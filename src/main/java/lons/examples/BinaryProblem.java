package lons.examples;

import lons.Problem;
/**
 * Class defines additional static methods that BinaryProblem can use
 * 
 * @author Jonathan Fieldsend
 * @version 11/07/2018
 */
public interface BinaryProblem extends Problem<BinarySolution>
{
    static final int MAX_BITS_UNIQUE=31;
    static final int[] bitPowers = BinaryProblem.getBitPowers(MAX_BITS_UNIQUE); //precompute for speed
    
    /**
     * Converts array of booleans to an integer value. If array is longer than 31 elements this
     * integer will not be unique across all solutions
     * 
     * @param x array to convert to integer value
     * @return integer representing boolean array
     */
    static int getIndexOfBitString(boolean[] x) {
        int index = 0;
        for (int i=0; i<Math.min(x.length,MAX_BITS_UNIQUE); i++)
            index += x[i] ? BinaryProblem.bitPowers[i] : 0;
        return index;  
    }
    
    /**
     * Converts an index to a corresponding bit string representation. Note that 
     * cannot convert for lengths greater than 31, due to limit on size of Java int 
     * type
     * 
     * @param index integer representation of bitstring to convert
     * @param bitStringLength length of boolean array to generate
     * @return array of booleans corrsponding to index value
     */
    static boolean[] getBitStringCorrespondingToIndex(int index, int bitStringLength) {
        boolean[] bitString = new boolean[bitStringLength];
        for (int i=bitStringLength-1; i>=0; i--) {
            if (index>=bitPowers[i]) {
                bitString[i] = true;
                index -= bitPowers[i];
            } else {
                bitString[i] = false;
            }
        }
        return bitString;
    }
    
    /**
     * Get array representing 2 raised to all non-negative integer values up to power 
     * 
     * @param power power to raise to
     * @return array containing 2^0 up to 2^power
     */
    static int[] getBitPowers(int power) {
        int[] x = new int[power];
        x[0] = 1;
        for (int i = 1; i < power; i++)
            x[i] = x[i-1] * 2;
        return x;
    }
    
    /**
     * Get value of 2 raised to power
     * 
     * @param power power to raise to
     * @return 2^power value
     */
    static int getBitPower(int power) {
        int v = 1;
        for (int i = 1; i <= power; i++)
            v *= 2;
        return v;
    }
}
