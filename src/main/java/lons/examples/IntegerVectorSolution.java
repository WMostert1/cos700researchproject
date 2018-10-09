package lons.examples;

import lons.Solution;

/**
 * Class defines additional methods that IntegerVectorSolutions must provide beyond those
 * defined in Solution
 * 
 * @author Jonathan Fieldsend
 * @version 11/07/2018
 */
public abstract class IntegerVectorSolution extends Solution
{
    public abstract int[] getDesignVector();
    public abstract int getDesignVariable(int index);
    public abstract int getMaxValue();
}
