package mutators;

import java.security.SecureRandom;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Created by bbdnet1339 on 2016/08/08.
 *
 */
public class UniformBitMutator extends BitMutator {
    private int numberOfBits = 1;

    public UniformBitMutator() {
    }

    public UniformBitMutator(int numberOfBits){
        this.numberOfBits = numberOfBits;
    }

    @Override
    public boolean[] mutate(boolean[] arr) {
        List<Integer> mutationIndices = new ArrayList<>();
        int rIndex = randomGen.nextInt(arr.length-1);

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

        return newArr;

//        if(satisifesMinimumPercentageAttributes(newArr))
//            return newArr;
//        else
//            return mutate(arr);
    }

}
