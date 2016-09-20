import java.security.SecureRandom;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Created by bbdnet1339 on 2016/08/08.
 *
 */
public class UniformBitMutator extends BitMutator{
    private double bitfactor = 0.1;

    public UniformBitMutator(double bitfactor){
        this.bitfactor = bitfactor;
    }

    @Override
    public boolean[] mutate(boolean[] arr) {
        List<Integer> mutationIndices = new ArrayList<>();
        int rIndex = randomGen.nextInt(arr.length-1);

        int numberOfBits = (int)bitfactor*arr.length;
        if (numberOfBits < 1)
            numberOfBits = 1;

        while(mutationIndices.size() != numberOfBits)
            if (!mutationIndices.contains(rIndex))
                mutationIndices.add(rIndex);
            else
                rIndex = randomGen.nextInt(arr.length-1);

        boolean [] newArr = Arrays.copyOf(arr,arr.length);
        for(Integer i : mutationIndices)
            newArr[i] = !newArr[i];

        if(satisifesMinimumPercentageAttributes(newArr))
            return newArr;
        else
            return mutate(arr);
    }

}
