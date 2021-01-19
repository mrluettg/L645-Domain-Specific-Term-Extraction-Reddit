import org.json.simple.parser.ParseException;

import java.io.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Scanner;

public class FindDSTs {
    static String FILEPATH = "D:\\L645_corpora\\project_corpora\\";
    static String TESTCORPUS = "IndianaUniversity";
    public static HashMap<String, Integer> readFrequencyFile(String filepath) throws FileNotFoundException {
        HashMap<String, Integer> termFreq = new HashMap<>();
        Scanner scanner = new Scanner(new File(filepath));
        while(scanner.hasNextLine()){
            String line = scanner.nextLine();
            String[] lineLst = line.split("\t");
            if(lineLst.length > 1){//bug in the py code. case is rare so skipping shouldn't impact algorithm
                termFreq.put(lineLst[0], Integer.parseInt(lineLst[1]));
            }
        }
        return termFreq;
    }

    public static HashMap<String, Integer> readTestFile() throws FileNotFoundException{
        System.out.println("reading test file...");
        return readFrequencyFile(FILEPATH + "test_frequencies\\" + TESTCORPUS + "_MWE_frequencies.txt");
    }
    public static ArrayList<HashMap<String, Integer>> readBaseFiles() throws FileNotFoundException {
        System.out.println("reading base file...");
        ArrayList<HashMap<String, Integer>> index = new ArrayList<>();
        File folder = new File(FILEPATH + "base_frequencies");
        for (final File fileEntry : folder.listFiles()) {
            index.add(readFrequencyFile(fileEntry.getPath()));
        }
        return index;
    }

    public static int getLength(HashMap<String, Integer> frequencies){
        int length = 0;
        String[] keys = frequencies.keySet().toArray(new String[0]);
        for(String key: keys){
            length += frequencies.get(key);
        }
        return length;
    }
    public static HashMap<String, Integer> getLengths(HashMap<String, HashMap<String, Integer>> subredditFrequencies){
        HashMap<String, Integer> subredditTermFreqs = new HashMap<>();
        String[] subreddits = subredditFrequencies.keySet().toArray(new String[0]);
        for(String subreddit: subreddits){
            subredditTermFreqs.put(subreddit, getLength(subredditFrequencies.get(subreddit)));
        }
        return subredditTermFreqs;
    }
    public static ArrayList<TermResult> getTestTFIDFS() throws IOException {
        ArrayList<HashMap<String, Integer>> frequencies= readBaseFiles();
        HashMap<String, Integer> testFrequencies = readTestFile();
        System.out.println("calculating tfidf for each term");
        frequencies.add(testFrequencies);
        double N = frequencies.size();
        ArrayList<TermResult> tfidfs = new ArrayList<>();
        double length = getLength(testFrequencies);
        String[] terms = testFrequencies.keySet().toArray(new String[0]);
        for(String term: terms){
            if(term.contains("_'s")){
                continue;
            }
            double c = testFrequencies.get(term);
            double k = 0;
            for(HashMap<String, Integer> subredditFrequencies: frequencies){
                if(subredditFrequencies.containsKey(term)){
                    k += 1;
                }
            }
            //want to try out some more of these.
            if(k == 0){
                System.out.println("k is zero");
            }
            TermResult tr = new TermResult(term, Math.log(1 + c)/length, Math.log(1+(N-k)/k));
            //TermResult tr = new TermResult(term, c/length, Math.log(1+N/k));
            tfidfs.add(tr);
        }
        return tfidfs;
    }


    public static double findMean(ArrayList<TermResult> termResults) {
        double mean = 0;
        double N = termResults.size();
        for(TermResult tr: termResults){
            mean += tr.F/N;
        }
        return mean;
    }
    //helper for analyze
    //finds the standard deviation of tf-idf terms given the mean.
    public static double findStandardDeviation(ArrayList<TermResult> termResults, double mean){
        double variance = 0;
        double N = termResults.size();
        for(TermResult termResult: termResults){
            variance += Math.pow(termResult.F - mean, 2) / N;
        }
        return Math.sqrt(variance);
    }
    public static void writeResults(ArrayList<TermResult> results, String filePath) throws IOException {
        System.out.println("writing...");
        FileWriter out = new FileWriter(filePath);
        int incrementer = 1;
        for(TermResult result: results){
            out.write(incrementer + "\t" + result.term + "\t" + result.F + "\t" + result.tf + "\t" + result.idf + "\n");
            incrementer++;
        }
        out.close();
    }
    public static void writeEvaluation(ArrayList<TermResult> results, String filePath) throws IOException {
        FileWriter evaluationWriter = new FileWriter(filePath);
        int i = 0;
        for(TermResult tr: results){
            evaluationWriter.write((1 + i) + "\t" + tr.getTerm() + "\t" + "\n");
            i++;
        }
        evaluationWriter.close();
    }
    public static void analyze() throws IOException {
        ArrayList<TermResult> tfidfs = getTestTFIDFS();
        Collections.sort(tfidfs);
        Collections.reverse(tfidfs);
        writeResults(tfidfs, FILEPATH + "all_" + TESTCORPUS + ".txt");
        double mean = findMean(tfidfs);
        double sd = findStandardDeviation(tfidfs, mean);
        double threshold = (9*sd + mean);
        ArrayList<TermResult> topResults = new ArrayList<>();
        int i = 0;
        while(tfidfs.get(i).F > threshold){
            topResults.add(tfidfs.get(i));
            i++;
        }
        writeResults(topResults, FILEPATH + "top_" + TESTCORPUS + ".txt");
        writeEvaluation(topResults, FILEPATH + "evaluation\\" + TESTCORPUS + "_evaluation.txt");
    }

    public static void main(String[] args) throws IOException {
        analyze();
    }





}
