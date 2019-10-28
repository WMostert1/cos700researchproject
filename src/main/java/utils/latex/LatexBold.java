package utils.latex;

public class LatexBold extends AbstractLatexString {

    public LatexBold(final String value) {
        super("\\textbf{"+value+"}");
    }

    public static LatexBold latexBold(final String value){
        return new LatexBold(value);
    }
}
