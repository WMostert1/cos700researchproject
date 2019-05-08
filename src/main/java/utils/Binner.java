package utils;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by bbdnet1339 on 2016/08/08.
 */
public class Binner {
    public Binner(int numberOfBins){

    }

    public void bin(Map<boolean[],Double> results){

    }

    private long getLongValueOfBoolArr(boolean [] boolArr){
        long temp = 0;
        int count = 0;
        for(int i = boolArr.length-1;i>0;i--){
            if(boolArr[i])
                temp += Math.pow(2,count);

            count++;
        }
        return temp;
    }
}
