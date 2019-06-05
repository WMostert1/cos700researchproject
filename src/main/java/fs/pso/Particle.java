package fs.pso;

import com.google.common.collect.Lists;
import utils.MathUtils;

import java.math.BigDecimal;
import java.security.SecureRandom;
import java.util.ArrayList;
import java.util.Arrays;

public class Particle {
    protected int dimensions;
    protected ArrayList<BigDecimal> position;
    protected ArrayList<BigDecimal> velocity;
    protected BigDecimal fitness;
    protected ArrayList<BigDecimal> pBestPosition = null;
    protected BigDecimal pBestFitness = null;
    protected SecureRandom random = new SecureRandom();
    protected BigDecimal xmin, xmax;
    protected int [] disabledDimensions;

    public Particle(int dimensions, BigDecimal xmin, BigDecimal xmax, int [] disabledDimensions){
        this.disabledDimensions = disabledDimensions;
        this.dimensions = dimensions;
        this.position = Lists.newArrayList();
        for(int i = 0; i < dimensions; i++){
            BigDecimal randomPosition =  xmin.add(xmax.subtract(xmin).multiply(MathUtils.doubleToBigDecimal(random.nextDouble())));
            this.position.add(randomPosition);
        }
        this.velocity = Lists.newArrayList();
        for(int i = 0; i < dimensions; i++){
            this.velocity.add(BigDecimal.ZERO);
        }
        this.xmin = xmin;
        this.xmax = xmax;
    }

    public Particle(int dimensions, BigDecimal xmin, BigDecimal xmax){
       this(dimensions, xmin, xmax, new int [] {});
    }

    public boolean isDisabledDimension(int j){
        for(int i = 0; i < disabledDimensions.length; i++){
            if(disabledDimensions[i] == j){
                return true;
            }
        }
        return false;
    }

    public Particle clone(){
        Particle p = new Particle(dimensions, xmin, xmax, disabledDimensions.clone());
        p.setpBestFitness(getpBestFitness());
        p.setpBestPosition((ArrayList<BigDecimal>) getpBestPosition().clone());
        p.setPosition((ArrayList<BigDecimal>) getPosition().clone());
        p.setFitness(getFitness());
        p.setVelocity((ArrayList<BigDecimal>) getVelocity().clone());
        return p;
    }

    public ArrayList<BigDecimal> getPosition() {
        return position;
    }

    public BigDecimal getPosition(int dimension) {
        return position.get(dimension);
    }

    public void setPosition(ArrayList<BigDecimal> position) {
        this.position = position;
    }

    public void setPosition(int dimension, BigDecimal value){
        if(value.compareTo(xmin) < 0){
            value = xmin;
        }

        if(value.compareTo(xmax) > 0){
            value = xmax;
        }

        this.position.set(dimension, value);
    }

    public ArrayList<BigDecimal> getVelocity() {
        return velocity;
    }

    public BigDecimal getVelocity(int dimension) {
        return this.velocity.get(dimension);
    }

    public void setVelocity(ArrayList<BigDecimal> velocity) {
        this.velocity = velocity;
    }

    public void setVelocity(int dimension, BigDecimal value){
        this.velocity.set(dimension, value);
    }

    public BigDecimal getFitness() {
        return fitness;
    }

    public void setFitness(BigDecimal fitness) {
        this.fitness = fitness;
    }

    public ArrayList<BigDecimal> getpBestPosition() {
        return pBestPosition;
    }

    public BigDecimal getpBestPosition(int dimension) {
        return pBestPosition.get(dimension);
    }

    public void setpBestPosition(ArrayList<BigDecimal> pBestPosition) {
        this.pBestPosition = pBestPosition;
    }

    public int getDimensions() {
        return dimensions;
    }

    public void setDimensions(int dimensions) {
        this.dimensions = dimensions;
    }

    public BigDecimal getpBestFitness() {
        return pBestFitness;
    }

    public void setpBestFitness(BigDecimal pBestFitness) {
        this.pBestFitness = pBestFitness;
    }

    public int getEnabledDimensions(){
        return dimensions - disabledDimensions.length;
    }

    public int[] getDisabledDimensions() {
        return disabledDimensions;
    }

    public void setDisabledDimensions(int[] disabledDimensions) {
        this.disabledDimensions = disabledDimensions;
    }
}
