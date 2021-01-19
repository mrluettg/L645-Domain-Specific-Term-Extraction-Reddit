import org.json.simple.parser.ParseException;

import java.io.IOException;

//a class that holds the F score and a term.
// Makes it easy to sort if all these are together.
public  class TermResult implements Comparable<TermResult> {
    String term;
    double tf;
    double idf;
    double F;

    public TermResult(String term, double tf, double idf) {
        this.term = term;
        this.F = tf * idf;
        this.tf = tf;
        this.idf = idf;
    }
    public String getTerm(){
        return term;
    }
    public double getF(){
        return F;
    }

    //comparable so that they can be easily sorted.
    @Override
    public int compareTo(TermResult t) {
        return Double.compare(this.F, t.F);
    }
}