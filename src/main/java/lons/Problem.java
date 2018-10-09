package lons;


/**
 * Interface defines methods all Problem instances must provide.
 * 
 * @author Jonathan Fieldsend
 * @version 11/07/2018
 */
public interface Problem<K extends Solution>
{
    /**
     * Evaluates the quality of the argument s
     * 
     * @param s solution to evaluate
     * @return quality of s
     */
    double getQuality(K s);
    
    /**
     * Returns the legal domain of this Problem as an array of solutions
     * 
     * @return array of all legal solutions, without duplicates
     * @throws UnsupportedOperationException in situations where this method has 
     * not been implemented/unsupported (e.g. if for problem returning an array of 
     * all potential solution instances is infeasible) 
     */
    K[] getExhaustiveSetOfSolutions() throws UnsupportedOperationException;
    
    /**
     * Returns a random legal solution for this Problem
     * 
     * @return random legal solution
     */
    K getRandomSolution();
}
