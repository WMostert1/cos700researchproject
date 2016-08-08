import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

import java.io.BufferedReader;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by bbdnet1339 on 2016/08/08.
 */
public class FeatureSelector {
    public String booleanArrayToBitString(boolean [] arr){
        String bitString = "";
        for(boolean b : arr)
            bitString += b ? "1" : "0";
        return bitString;
    }

    public boolean [] bitStringToBooleanArray(String bitString){
        boolean [] boolArr = new boolean[bitString.length()];
        for(int i = 0; i < bitString.length();i++){
            boolArr[i] = bitString.charAt(i) != '0';
        }
        return boolArr;
    }

    public int [] booleanArrayToWekaAttribSelectionArray(boolean [] arr){
        List<Integer> indicesToInclude = new ArrayList<Integer>();
        for(int i = 0; i < arr.length; i++){
            if(!arr[i])
                indicesToInclude.add(i);
        }
        final int [] retArr = new int[indicesToInclude.size()];
        for(int i = 0; i < retArr.length;i++){
            retArr[i] = indicesToInclude.get(i);
        }
        return retArr;
    }

    public Instances getInstancesFromBitString(Instances data, boolean [] bitString) throws Exception {
        Remove remove = new Remove();
        remove.setAttributeIndicesArray(booleanArrayToWekaAttribSelectionArray(bitString));

            data.setClassIndex(data.numAttributes() - 1);
        //remove.setInvertSelection(false);
        remove.setInputFormat(data);
        Instances instNew = Filter.useFilter(data, remove);
        return instNew;
    }

}
