package lons.examples;

import lons.Neighbourhood;
/**
 * InstrumentedBinaryHammingNeighbourhood, instrumented version of BinaryHammingNeighbourhood
 * to track calls.
 * 
 * @author Jonathan Fieldsend
 * @version 11/07/2018
 */
public class InstrumentedBinaryHammingNeighbourhood implements Neighbourhood<BinarySolution>
{
    static int neighbourhoodCalls;
    BinaryHammingNeighbourhood wrappedInstance;
    InstrumentedBinaryHammingNeighbourhood(BinaryHammingNeighbourhood wrappedInstance) {
        this.wrappedInstance = wrappedInstance;
    }
    
    @Override
    public BinarySolution[] neighbouringSolutions(BinarySolution s){
        neighbourhoodCalls++;
        return wrappedInstance.neighbouringSolutions(s);
    }
    
    @Override
    public void setMaxDistance(int maxDistance) {
        wrappedInstance.setMaxDistance(maxDistance);
    }
    
//     @Override
//     public int[] indicesOfNeighbouringSolutions(BinarySolution s){
//         neighbourhoodCalls++;
//         return wrappedInstance.indicesOfNeighbouringSolutions(s);
//     }

    @Override
    public BinarySolution[] neighbouringSolutionsUpToDistance(BinarySolution s){
        neighbourhoodCalls++;
        return wrappedInstance.neighbouringSolutionsUpToDistance(s);
    }
    
//     @Override
//     public int[] indicesOfNeighbouringSolutions(BinarySolution s, int maxDistance){
//         neighbourhoodCalls++;
//         return wrappedInstance.indicesOfNeighbouringSolutions(s, maxDistance);
//     }
}
