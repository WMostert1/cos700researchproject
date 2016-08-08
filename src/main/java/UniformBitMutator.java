import java.security.SecureRandom;
import java.util.Arrays;

/**
 * Created by bbdnet1339 on 2016/08/08.
 *
 */
public class UniformBitMutator extends BitMutator{

    private SecureRandom randomGen = new SecureRandom();
    @Override
    public boolean[] mutate(boolean[] arr) {
        boolean [] newArr = Arrays.copyOf(arr,arr.length);
        int rIndex = randomGen.nextInt(arr.length-1);
        newArr[rIndex] = !newArr[rIndex];
        if(satisifesMinimumPercentageAttributes(newArr))
        return newArr;
        else
            return mutate(arr);
    }

}
