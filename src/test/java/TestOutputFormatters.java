import org.junit.Test;
import utils.LatexOutputFormatter;
import utils.OutputFormatter;

import java.io.FileNotFoundException;
import java.io.UnsupportedEncodingException;
import java.math.BigDecimal;

public class TestOutputFormatters {

    @Test
    public void testLatex() throws FileNotFoundException, UnsupportedEncodingException {
        OutputFormatter outF = new LatexOutputFormatter("out/test.tex", "Test Caption", "test:label", "Column A", "Column B", "Column C");
        outF.addAsColumns("test string", BigDecimal.valueOf(78.123456), 1002.123);
        outF.addAsColumns( BigDecimal.valueOf(0.01123456), 0.12345, "x");
        outF.save();
    }

}
