package lons.examples;

import lons.Problem;

/**
 * Class defines additional methods that IntegerVectorProblems must provide beyond those
 * defined in Problem
 * 
 * @author Jonathan Fieldsend
 * @version 11/07/2018
 */
public interface IntegerVectorProblem extends Problem<IntegerVectorSolution>
{
    static int getIndexOfIntegerVector(int[] x, int resolution) {
        int index = 0;
        int multiplier = 1;
        
        for (int i=0; i<x.length; i++) {
            index += x[i]*multiplier;
            multiplier *= resolution;
            if (multiplier==Integer.MAX_VALUE)
                break; // will be potential hash code clashes, but domain size is larger than max int
        }
        return index;  
    }
    
}
