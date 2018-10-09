package lons.examples;

import lons.Solution;
/**
 * Class defines additional  methods that BinarySolutions must provide beyond those
 * defined in Solution
 * 
 * @author Jonathan Fieldsend
 * @version 11/07/2018
 */
public abstract class BinarySolution extends Solution
{
    /**
     * Returns this solution as an array of booleans (may be shallow copy)
     * 
     * @return array of booleans representing bitstring solution
     */
    public abstract boolean[] getDesignVector();
    
    /**
     * Returns the value of this solution's index element as a booleans
     * 
     * @param index index of element to return
     * @return booleans representing bit value of the index element
     */
    public abstract boolean getDesignVariable(int index);
}
