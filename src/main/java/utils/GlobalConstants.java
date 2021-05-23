package utils;

import java.math.BigDecimal;

public class GlobalConstants {
    public static final int DECIMAL_PLACES = 7;
    public static final double TRAINING_PERCENTAGE = 60.0;
    public static final int REPORTED_DECIMAL_PLACES = 8;

    public static final int SAMPLE_SCALE = 20;
    public static final BigDecimal K_C = BigDecimal.valueOf(1.0);
    public static final BigDecimal K_P = BigDecimal.valueOf(0.25);

    public static final BigDecimal MAX_FC = BigDecimal.ONE;
    public static final BigDecimal MIN_FC = BigDecimal.ZERO;

    public static final BigDecimal MAX_FP = BigDecimal.ONE;
    public static final BigDecimal MIN_FP = BigDecimal.ZERO;

    public static BigDecimal getBfiMin(){
        return MIN_FC.multiply(K_C)
                .subtract(MAX_FP.multiply(K_P))
                .subtract(MAX_FC).multiply(K_C)
                .add(MIN_FP.multiply(K_P));
    }

    public static BigDecimal getBfiMax(){
        return MAX_FC.multiply(K_C)
                .subtract(MIN_FP.multiply(K_P))
                .subtract(MIN_FC.multiply(K_C))
                .add(MAX_FP.multiply(K_P));
    }
}
