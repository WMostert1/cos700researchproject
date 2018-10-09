package lons;


/**
 * The Solution class, provides some default functionality and methods concrete subtypes
 * must provide
 * 
 * @author Jonathan Fieldsend
 * @version 11/07/2018
 */
public abstract class Solution
{
    /**
     * Method returns the solution converted into an integer (which can be used as an 
     * index)
     * 
     * @return integer representing the solution
     */
    public abstract int getIndex();
    
    /**
     * Method returns the number of elements of the solution (e.g. the bits in a bitstring, 
     * elements in a decision vector, etc.)
     * 
     * @return the number of elements of this Solution
     */
    public abstract int getNumberOfElements();
    
    /**
     * {@inheritDoc}
     */
    @Override
    public int hashCode() {
        return getIndex();
    }
    
    /**
     * Implementation assumes the getIndex method returns a unique integer for each
     * solution, if this is not the case you will need to override this method
     * 
     * @return true if argument is equal to this Solution
     */
    @Override
    public boolean equals(Object obj) {
        if (obj instanceof Solution) 
            if (((Solution) obj).getIndex()==this.getIndex())
                return true;
        
        return false;
    }
}
