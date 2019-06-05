package fs.pso;

import java.math.BigDecimal;

public interface ParticleFitnessCalculator {

    BigDecimal calculateFitness(Particle p);
}
