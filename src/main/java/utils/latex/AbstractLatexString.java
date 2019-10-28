package utils.latex;

public abstract class AbstractLatexString {
    private final String value;

    public AbstractLatexString(final String value) {
        if(value == null || value.isEmpty()){
            this.value = "";
        }else {
            this.value = value;
        }
    }

    @Override
    public String toString() {
        return value;
    }
}
