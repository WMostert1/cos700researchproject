package fs.pso;

import com.google.common.collect.Lists;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.List;

public abstract class AbstractPSO {
    protected ParticleFitnessCalculator particleFitnessCalculator;
    protected List<Particle> swarm;
    protected int dimensions;
    protected int iterations;
//    protected int [] disabledDimensions;
    public AbstractPSO(ParticleFitnessCalculator particleFitnessCalculator, int dimensions, int swarmSize, BigDecimal xmin, BigDecimal xmax, int iterations, int [] disabledDimensions) {
        this.dimensions = dimensions;
        this.iterations = iterations;
        this.particleFitnessCalculator = particleFitnessCalculator;
        List<Particle> swarm = Lists.newArrayList();
        for(int i = 0; i < swarmSize; i++){
            swarm.add(new Particle(dimensions, xmin, xmax, disabledDimensions));
        }
        this.swarm = swarm;
    }

    public AbstractPSO(ParticleFitnessCalculator particleFitnessCalculator, int dimensions, int swarmSize, BigDecimal xmin, BigDecimal xmax, int iterations) {
        this(particleFitnessCalculator, dimensions, swarmSize, xmin, xmax, iterations, new int []{});
    }

    abstract protected List<Particle> getNeighbourhood(Particle particle);

    abstract protected void updateVelocities();

    abstract protected void updatePositions();

    abstract public Particle optimize();

    public int getDimensions() {
        return dimensions;
    }
}
