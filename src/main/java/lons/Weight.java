package lons;


/**
 * Weight class provides a type that can be stored in collections, which can track 
 * integer values and be incremented
 * 
 * @author Jonathan Fieldsend
 * @version 11/07/2018
 */
public class Weight
{
    private int weight; // cargo

    /**
     * Constructor sets up a Weight instance with a weight of 0
     */
    public Weight(){ }

    /**
     * Constructor sets up a Weight instance with an initial weight as in the argument
     * 
     * @param weight initial weight to be used in this instance
     */
    public Weight(int weight){ 
        this.weight = weight; 
    }
    
    /**
     * Increase weight tracked by 1
     */
    void increment()
    {
        weight++;
    }
    
    /**
     * Returns weight of this Weight instance
     * 
     * @return weight stored
     */
    int getWeight() {
        return weight;
    }
}
