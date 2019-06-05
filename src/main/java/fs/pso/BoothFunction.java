package fs.pso;

import sun.reflect.generics.reflectiveObjects.NotImplementedException;
import utils.MathUtils;

import java.math.BigDecimal;

import static utils.MathUtils.*;

public class BoothFunction implements ParticleFitnessCalculator {
    @Override
    public BigDecimal calculateFitness(Particle p) {
        if(p.getDimensions() != 2){
            throw new RuntimeException("Boothes function only works on two dimensional problems.");
        }

        BigDecimal termOne = p.getPosition(0)
                .add(doubleToBigDecimal(2.0).multiply(p.getPosition(1)))
                .subtract(doubleToBigDecimal(7.0)).pow(2);

        BigDecimal termTwo = doubleToBigDecimal(2.0).multiply(p.getPosition(0))
                .add(p.getPosition(1))
                .subtract(doubleToBigDecimal(5.0)).pow(2);

        BigDecimal val = termOne.add(termTwo);
        return val;
    }

}
