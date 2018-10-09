package lons.examples;

import lons.Neighbourhood;
/**
 * IntegerVectorNeighbourhood -- von Neumann nieghbourhood on single integer changes used   
 * 
 * @author Jonathan Fieldsend
 * @version 11/07/2018
 */
public class IntegerVectorNeighbourhood implements Neighbourhood<IntegerVectorSolution>
{
    
    private int maxDistance = 2;
    
    @Override
    public void setMaxDistance(int maxDistance) {
        this.maxDistance = maxDistance > 0 ? maxDistance : 1;
    }
    
    @Override
    public IntegerVectorSolution[] neighbouringSolutions(IntegerVectorSolution s){
        int[][] singleNeighbours = getSingleNeighbours(s);
        return generateNeighbours(singleNeighbours,s.getMaxValue());
    }
    
    @Override
    public IntegerVectorSolution[] neighbouringSolutionsUpToDistance(IntegerVectorSolution s){
        int[][] bitStringNeighbours = null; //getBitStringNeighboursUpToDistance(s, maxDistance);
        return generateNeighbours(bitStringNeighbours, s.getMaxValue());
    }
    
    private static IntegerVectorSolution[] generateNeighbours(int[][] neighbourMatrix, int maxValue) {
        IntegerVectorSolution[] neighbours = new IntegerVectorSolution[neighbourMatrix.length];
        
        for (int i=0; i<neighbours.length; i++) {
            neighbours[i] = ConcreteIntegerVectorSolution.constructIntegerVectorSolution(neighbourMatrix[i],maxValue);
        }
        return neighbours;
    }
    
    private static int[][] getSingleNeighbours(IntegerVectorSolution s) {
        int[] x = s.getDesignVector();
        int upper = s.getMaxValue();
        int numberOfNeighbours = x.length*2;
        for (int i : x){
            if (i==0)
                numberOfNeighbours--;
            else 
                if (i==upper)
                    numberOfNeighbours--;
        }
            
        int[][] allNeighbours = new int[numberOfNeighbours][x.length];
        int index = 0;
        boolean down = true; 
        for  (int i=0; i <numberOfNeighbours; i++) {
            for  (int j=0; j < x.length; j++) {
                allNeighbours[i][j] = x[j];
            }
            boolean inserted = false;
            do {
                if (down) {
                    if (x[index]!=0) { // skip lowest value, 
                        allNeighbours[i][index]-=1;
                        inserted = true;
                    }
                    down=false; //look up next
                } else {
                   if (x[index]!=upper) { // skip highest value, so skip
                        allNeighbours[i][index]+=1;
                        inserted = true;
                   }
                   index++; // move on to next variable after increment considered
                   down = true; // look down next 
                }
            } while (inserted==false);
        }
        return allNeighbours;    
    }
    
}
