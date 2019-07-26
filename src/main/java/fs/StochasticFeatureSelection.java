package fs;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.List;

public interface StochasticFeatureSelection {
    ArrayList<BigDecimal> getIterationFitnessValues();
}
