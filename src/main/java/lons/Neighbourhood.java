package lons;


/**
 * Interface defines methods all Neighbourhood instances must provide.
 * 
 * @author Jonathan Fieldsend
 * @version 11/07/2018
 */
public interface Neighbourhood<K extends Solution>
{
    /**
     * Method returns an array of solutions which neighbour argument s.
     * 
     * @param s solution whose neighbours will be returned
     * @return array of neighbours of s
     */
    K[] neighbouringSolutions(K s);
    
    /**
     * Method returns an array of solutions which neighbour s, up to and including the 
     * distance currently stored in this Nighbourhood.
     * 
     * @param s solution whose neighbours will be returned
     * @return array of neighbours of s
     */
    K[] neighbouringSolutionsUpToDistance(K s);
    
    /**
     * Method sets the distance used by this Neighbourhood in calls to 
     * neighbouringSolutionsUpToDistance.
     * 
     * @param maxDistance maxiumum distance to use for neighbouringSolutionsUpToDistance
     */
    void setMaxDistance(int maxDistance);
}
