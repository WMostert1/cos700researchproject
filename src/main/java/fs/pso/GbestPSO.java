package fs.pso;

import utils.MathUtils;

import java.math.BigDecimal;
import java.security.SecureRandom;
import java.util.ArrayList;
import java.util.List;

public class GbestPSO extends AbstractPSO {
    private Particle gbestParticle = null;
    private BigDecimal c1, c2;
    private final SecureRandom random;

    public GbestPSO(ParticleFitnessCalculator particleFitnessCalculator, int dimensions, int swarmSize, BigDecimal c1, BigDecimal c2, BigDecimal xmin, BigDecimal xmax, int iterations) {
        super(particleFitnessCalculator, dimensions, swarmSize, xmin, xmax, iterations);
        this.c1 = c1;
        this.c2 = c2;
        random = new SecureRandom();
        //Calculate initial best and velocities
        for(Particle p : swarm){
            p.setpBestPosition((ArrayList<BigDecimal>) p.getPosition().clone());
            p.setFitness(this.particleFitnessCalculator.calculateFitness(p));
            p.setpBestFitness(p.getFitness());
            if(gbestParticle == null){
                gbestParticle = p.clone();
            }else if(p.getFitness().compareTo(gbestParticle.getFitness()) > 0){
                gbestParticle = p.clone();
            }
        }
    }

    @Override
    protected List<Particle> getNeighbourhood(Particle particle) {
        //All particles in the swarm are part of the neighborhood
        return this.swarm;
    }

    @Override
    protected void updateVelocities() {
        for (Particle p : this.swarm) {
            for (int j = 0; j < dimensions; j++) {
                BigDecimal previousVelocity = p.getVelocity(j);
                BigDecimal cognitiveComponent = c1.multiply(MathUtils.doubleToBigDecimal(random.nextDouble())).multiply(
                        p.getpBestPosition(j).subtract(p.getPosition(j))
                );
                BigDecimal socialComponent = c2.multiply(MathUtils.doubleToBigDecimal(random.nextDouble())).multiply(
                        gbestParticle.getPosition(j).subtract(p.getPosition(j))
                );
                BigDecimal newVelocity = previousVelocity.add(cognitiveComponent).add(socialComponent);
                p.setVelocity(j, newVelocity);
            }
        }
    }

    @Override
    protected void updatePositions() {
        for(Particle p : this.swarm){
            for(int j = 0; j < dimensions; j++){
                BigDecimal newPosition = p.getPosition(j).add(p.getVelocity(j));
                p.setPosition(j, newPosition);
            }

            p.setFitness(this.particleFitnessCalculator.calculateFitness(p));
            if(p.getFitness().compareTo(p.getpBestFitness()) > 0){
                p.setpBestPosition((ArrayList<BigDecimal>) p.getPosition().clone());
                p.setpBestFitness(p.getFitness());
            }
            if(p.getFitness().compareTo(gbestParticle.getFitness()) > 0){
                gbestParticle = p.clone();
            }
        }
    }

    @Override
    public Particle optimize() {
        int countSinceLastIncrease = 0;
        BigDecimal previousGbestFitness = null;
        for(int i = 0; i < iterations && countSinceLastIncrease < 30; i++){
            updateVelocities();
            updatePositions();
            if(previousGbestFitness == null){
                previousGbestFitness = gbestParticle.getFitness();
            }else if (gbestParticle.getFitness().compareTo(previousGbestFitness) > 0){
                countSinceLastIncrease = 0;
                previousGbestFitness = gbestParticle.getFitness();
            }else{
                countSinceLastIncrease++;
            }
        }

        return gbestParticle;
    }

}
