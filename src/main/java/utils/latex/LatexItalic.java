package utils.latex;

public class LatexItalic extends AbstractLatexString {

    public LatexItalic(final String value) {
        super("\\textit{"+value+"}");
    }

    public static LatexItalic latexItalic(final String val){
        return new LatexItalic(val);
    }
}
