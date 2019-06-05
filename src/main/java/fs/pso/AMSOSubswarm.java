package fs.pso;

import com.google.common.collect.Lists;
import utils.MathUtils;

import java.math.BigDecimal;
import java.security.SecureRandom;
import java.util.ArrayList;
import java.util.List;

import static utils.MathUtils.doubleToBigDecimal;

public class AMSOSubswarm extends AbstractPSO {
    private Particle gbestParticle = null;
    private BigDecimal c;
    private final SecureRandom random;
    private List<Particle> groupOne = Lists.newArrayList();
    private List<Particle> groupTwo = Lists.newArrayList();
    private int currentItteration = 1;
    private BigDecimal innertia = null;

    public AMSOSubswarm(ParticleFitnessCalculator particleFitnessCalculator, int dimensions, int swarmSize,
                        BigDecimal xmin, BigDecimal xmax, int iterations, BigDecimal c, int [] disabledDimensions) {
        super(particleFitnessCalculator, dimensions, swarmSize, xmin, xmax, iterations, disabledDimensions);
        //Calculate initial best and velocities
        this.c = c;
        random = new SecureRandom();
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
        return swarm;
    }

    @Override
    protected void updateVelocities() {
        for(int i = 0; i < groupOne.size(); i++){
            Particle winner, loser;
            if(groupOne.get(i).getFitness().compareTo(groupTwo.get(i).getFitness()) > 0){
                winner = groupOne.get(i);
                loser = groupTwo.get(i);
            }else{
                winner = groupTwo.get(i);
                loser = groupOne.get(i);
            }


            for(int j = 0; j < dimensions; j++){
                if(loser.isDisabledDimension(j)){
                    continue;
                }
                BigDecimal velocityUpdateT1 = innertia.multiply(loser.getVelocity(j));
                BigDecimal velocityUpdateT2 = c.multiply(doubleToBigDecimal(random.nextDouble())).multiply(
                        winner.getpBestPosition(j).subtract(loser.getPosition(j))
                );
                loser.setVelocity(i, velocityUpdateT1.add(velocityUpdateT2));
            }
        }
    }

    @Override
    protected void updatePositions() {
        for(int i = 0; i < groupOne.size(); i++){
            Particle loser;
            if(groupOne.get(i).getFitness().compareTo(groupTwo.get(i).getFitness()) > 0){
                loser = groupTwo.get(i);
            }else{
                loser = groupOne.get(i);
            }

            for(int j = 0; j < dimensions; j++){
                if(loser.isDisabledDimension(j)){
                    continue;
                }
               loser.setPosition(j, loser.getPosition(j).add(loser.getVelocity(j)));
            }

            loser.setFitness(this.particleFitnessCalculator.calculateFitness(loser));
            if(loser.getFitness().compareTo(loser.getpBestFitness()) > 0){
                loser.setpBestPosition((ArrayList<BigDecimal>) loser.getPosition().clone());
                loser.setpBestFitness(loser.getFitness());
            }
            if(loser.getFitness().compareTo(gbestParticle.getFitness()) > 0){
                gbestParticle = loser.clone();
            }
        }
    }

    public void subswarmUpdate(int [] columnsToRemove, ParticleFitnessCalculator calculator, int dimensions){

        this.particleFitnessCalculator = calculator;
        for(Particle p : swarm){
            p.setDisabledDimensions(columnsToRemove);
        }

        gbestParticle = null;

        for(Particle particle : swarm) {
            particle.setFitness(this.particleFitnessCalculator.calculateFitness(particle));
            if (particle.getFitness().compareTo(particle.getpBestFitness()) > 0) {
                particle.setpBestPosition((ArrayList<BigDecimal>) particle.getPosition().clone());
                particle.setpBestFitness(particle.getFitness());
            }
            if(gbestParticle == null){
                gbestParticle = particle.clone();
            }else if (particle.getFitness().compareTo(gbestParticle.getFitness()) > 0) {
                gbestParticle = particle.clone();
            }
        }
    }

    @Override
    public Particle optimize() {
        BigDecimal innertiaTermOne = doubleToBigDecimal(0.9);
        BigDecimal innertiaTermTwo = doubleToBigDecimal(0.5).multiply(
                BigDecimal.valueOf(currentItteration).divide(BigDecimal.valueOf(iterations), MathUtils.ROUNDING_MODE)
        );
        innertia = innertiaTermOne.subtract(innertiaTermTwo);
        List<Integer> particleIndexes = Lists.newArrayList();
        for(int i = 0; i < swarm.size(); i++){
            particleIndexes.add(i);
        }
        boolean swop = true;
        while (!particleIndexes.isEmpty()){
            int particleIndex = random.nextInt(particleIndexes.size());
            int swarmIndex = particleIndexes.get(particleIndex);
            if(swop){
                groupOne.add(swarm.get(swarmIndex));
            }else{
                groupTwo.add(swarm.get(swarmIndex));
            }
            particleIndexes.remove(particleIndex);
            swop = !swop;
        }

        //make sure pair wise comparisons are possible, discard extra particle
        if(groupOne.size() > groupTwo.size()){
            groupOne.remove(groupOne.size()-1);
        }else if(groupOne.size() < groupTwo.size()){
            groupTwo.remove(groupTwo.size()-1);
        }

        if(groupOne.size() != groupTwo.size()){
            throw new RuntimeException("Groups are not equal in size. Pairwise comparison not possible.");
        }

        updateVelocities();
        updatePositions();
        groupOne.clear();
        groupTwo.clear();
        //Split in to two groups
        currentItteration++;
        return gbestParticle;
    }

    public int getEnabledDimensionsLength(){
        return swarm.get(0).getEnabledDimensions();
    }
}
