package utils;

import java.math.BigDecimal;
import java.math.RoundingMode;

public class MathUtils {

    public static RoundingMode ROUNDING_MODE = RoundingMode.HALF_UP;

    public static BigDecimal doubleToBigDecimal(Double value){
        BigDecimal bd = new BigDecimal(value);
        bd = bd.setScale(GlobalConstants.DECIMAL_PLACES, ROUNDING_MODE);
        return bd;
    }
}
