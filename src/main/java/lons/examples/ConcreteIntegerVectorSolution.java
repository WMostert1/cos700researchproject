package lons.examples;


/**
 * ConcreteIntegerVectorSolution allows instantiation of solutions
 * that maintain arrays of integers as design vectors
 * 
 * @author Jonathan Fieldsend
 * @version 11/07/2018
 */
public class ConcreteIntegerVectorSolution extends IntegerVectorSolution
{
    private int[] design;
    private int maxValue;
    private int index = -1;
    
    private ConcreteIntegerVectorSolution(int[] design, int maxValue) {
        this.design = design;
        this.maxValue = maxValue;
    }
    
    @Override
    public int getNumberOfElements(){
        return design.length;
    }
    
    @Override
    public int getMaxValue() {
        return maxValue;
    }
    
    @Override
    public int[] getDesignVector() {
        return design;
    }
    
    @Override
    public int getDesignVariable(int index) {
        return design[index];
    }
    
    @Override
    public int getIndex() {
        if (index==-1) {
            index = IntegerVectorProblem.getIndexOfIntegerVector(design,maxValue+1);
        }   
        return index;
    }
    
    @Override
    public boolean equals(Object obj) {
        if (obj instanceof ConcreteIntegerVectorSolution) 
            if (((ConcreteIntegerVectorSolution) obj).getIndex()==this.getIndex())
                if (this.sameState((ConcreteIntegerVectorSolution) obj))
                    return true;
        
        return false;
    }
    
    private boolean sameState(ConcreteIntegerVectorSolution solution) {
        for (int i=0; i<design.length; i++)
            if (design[i]!=solution.design[i])
                return false;
        return true;
    }
    
    public static IntegerVectorSolution constructIntegerVectorSolution(int[] design, int maxValue) {
        return new ConcreteIntegerVectorSolution(design,maxValue);
    }
}
