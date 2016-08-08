import java.security.SecureRandom;

/**
 * Created by bbdnet1339 on 2016/08/08.
 */
public abstract class BitMutator {
    protected static final double minimumAttributesCovered = 10.0;
    private SecureRandom randomGen;

    public BitMutator(){
        randomGen = new SecureRandom();
    }
    abstract boolean [] mutate(boolean[] arr);

    public boolean satisifesMinimumPercentageAttributes(boolean [] arr){
        if(!arr[arr.length -1])
            return false;//Arff files have the class attrib as last attrib

        int count = 0;

        for(boolean b : arr)
            if(b) count++;
        return (count/(double)arr.length*100.0) >= minimumAttributesCovered;
    }

    public boolean [] getRandomPoint(int size){
        boolean [] arr = new boolean[size];
        for(int i = 0; i < size; i++)
            arr[i] = randomGen.nextBoolean();
        if(satisifesMinimumPercentageAttributes(arr))
            return arr;
        else return getRandomPoint(size);
    }
}
