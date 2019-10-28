package fs;

import utils.OutputFormatter;

import java.io.FileNotFoundException;
import java.io.UnsupportedEncodingException;
import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.List;

public abstract class StochasticFeatureSelection {
    abstract public ArrayList<BigDecimal> getIterationFitnessValues();

    public void recordFitnessValues(OutputFormatter outputFormatter, List<BigDecimal> fitness, BigDecimal baselineFitness){
        if(fitness == null || fitness.isEmpty() || outputFormatter == null)
            return;

        for(BigDecimal d : fitness)
            outputFormatter.addAsColumns(d.subtract(baselineFitness).toString());

        try {
            outputFormatter.save();
        } catch (FileNotFoundException | UnsupportedEncodingException e) {
            throw new RuntimeException(e);
        }
    }
}
