package lons.examples;


/**
 * InstrumentedNKBinaryProblem, instrumented version of NKBinaryProblem
 * to track calls.
 * 
 * @author Jonathan Fieldsend
 * @version 11/07/2018
 */
public class InstrumentedNKModelProblem implements BinaryProblem
{
    private NKModelProblem wrappedInstance;
    static int evaluationCalls;
    
    InstrumentedNKModelProblem(NKModelProblem wrappedInstance) {
        this.wrappedInstance = wrappedInstance;
    }
    
    public double getQuality(BinarySolution s) {
        evaluationCalls++;
        return wrappedInstance.getQuality(s);
    }
    
    public BinarySolution[] getExhaustiveSetOfSolutions() {
        return wrappedInstance.getExhaustiveSetOfSolutions();
    }
    
    public BinarySolution getRandomSolution() {
        return wrappedInstance.getRandomSolution();
    }
}
