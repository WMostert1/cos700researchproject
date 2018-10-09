package samplers;

public interface SolutionSampler<T> {

    T getSample();

    boolean isDone();

    void showProgress();

    void reset();
}
