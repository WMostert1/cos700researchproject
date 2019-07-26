package mutators;

import java.util.Arrays;

/**
 * Created by bbdnet1339 on 2016/08/11.
 */
public class UniformSampleMutator extends BitMutator {
    @Override
    public boolean[] mutate(boolean[] arr) {
        boolean [] newArr = new boolean[arr.length];
        int i = 0;
        for(;i < arr.length-1;i++){
            newArr[i] = randomGen.nextBoolean();
        }
        newArr[i] = true;
        //if(satisifesMinimumPercentageAttributes(newArr))
            return newArr;
//        else
//            return mutate(arr);
    }
}
