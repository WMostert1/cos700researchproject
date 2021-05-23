package utils;

import java.math.BigDecimal;
import java.math.MathContext;
import java.math.RoundingMode;
import java.text.DecimalFormat;

import static utils.GlobalConstants.REPORTED_DECIMAL_PLACES;

public class MathUtils {

    public static MathContext ROUNDING_MODE = MathContext.DECIMAL32;

    public static BigDecimal doubleToBigDecimal(Double value){
        BigDecimal bd = new BigDecimal(value);
        bd = bd.setScale(GlobalConstants.DECIMAL_PLACES, RoundingMode.HALF_EVEN);
        return bd;
    }

    public static String doubleToString(Double value) {
        return bigDecimalToString(MathUtils.doubleToBigDecimal(value));
    }

    public static String doubleToString(Double value, int places) {
        return bigDecimalToString(MathUtils.doubleToBigDecimal(value), places);
    }

    public static String bigDecimalToString(BigDecimal value) {
        return bigDecimalToString(value, REPORTED_DECIMAL_PLACES);
    }

    public static String bigDecimalToString(BigDecimal value, int places) {

        DecimalFormat df = new DecimalFormat();

        df.setMaximumFractionDigits(places);

        df.setMinimumFractionDigits(places);

        df.setGroupingUsed(false);

        return df.format(value);
    }
}
