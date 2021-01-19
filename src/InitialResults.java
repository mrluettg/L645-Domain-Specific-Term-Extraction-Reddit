import java.io.*;
import java.util.*;


import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Scanner;


import org.apache.commons.math3.stat.inference.ChiSquareTest;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.*;

import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.HasWord;
import edu.stanford.nlp.process.CoreLabelTokenFactory;
import edu.stanford.nlp.process.DocumentPreprocessor;
import edu.stanford.nlp.process.PTBTokenizer;

import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.CollinsHeadFinder;
import java.lang.Math;



public class InitialResults {
    public static String FILEPATH = "D:\\L645_corpora\\project_corpora\\";
    //reads in a file of the .json format
    //returns a HashMap<String, String> where the first is a subreddit name, second is the text of the entire subreddit
    //presumes from different subreddits. (small-reddit-corpus)
    public static HashMap<String, String> readJSON(String filePath) throws IOException, ParseException {
        System.out.println("reading " + filePath + "...");
        HashMap<String, String> subreddits = new HashMap<>();
        File file = new File(filePath);
        Scanner scanner = new Scanner(file);
        while(scanner.hasNextLine()) {
            String line = scanner.nextLine();
            Object obj = new JSONParser().parse(line);
            JSONArray ja = (JSONArray) obj;
            for(Object o : ja){
                JSONObject jo = (JSONObject) o;
                Object metaObj = jo.get("meta");
                JSONObject meta = (JSONObject) metaObj;
                String subredditName = (String) meta.get("subreddit");
                String content = (String) jo.get("text");
                content = content.replace('\n', ' ');
                content = content.replace('\t', ' ');
                content = content + "\n";
                if(! subreddits.containsKey(subredditName)){
                    subreddits.put(subredditName, content);
                } else {
                    subreddits.put(subredditName, subreddits.get(subredditName) + content);
                }
            }
        }
        return subreddits;
    }
    //returns a HashMap<String, String> where the first is a subreddit name, second is the text of the entire subreddit
    //presumes from different subreddits. (small-reddit-corpus)
    //presumes all from same subreddit
    public static HashMap<String, String> readJSONL(String filePath) throws IOException, ParseException {
        System.out.println("reading " + filePath + "...");
        File file = new File(filePath);
        Scanner scanner = new Scanner(file);
        HashMap<String, String> subreddits = new HashMap<>();
        while(scanner.hasNextLine()) {
            String line = scanner.nextLine();
            Object obj = new JSONParser().parse(line);
            JSONObject jo = (JSONObject) obj;
            Object metaObj = jo.get("meta");
            JSONObject meta = (JSONObject) metaObj;
            String subredditName = (String) meta.get("subreddit");
            String content = (String) jo.get("text");
            content = content.replace('\n', ' ');
            content = content.replace('\t', ' ');
            if(! subreddits.containsKey(subredditName)){
                subreddits.put(subredditName, content);
            } else {
                subreddits.put(subredditName, subreddits.get(subredditName) + content);
            }
        }
        return subreddits;
    }

    //helper functino for filter bigrams
    //for chi squared, finds the total freq for each word.
    public static int totalFreq(HashMap<String, Integer> freqs){
        int total = 0;
        String[] terms = freqs.keySet().toArray(new String[0]);
        for(String term: terms){
            total += freqs.get(term);
        }
        return total;
    }
    //reads corenlp_stopwords.txt and returns it as a linked list.

    public static ArrayList<String> getStopwords() throws FileNotFoundException {
        ArrayList<String> stopwords = new ArrayList<String>();
        Scanner scanner = new Scanner(new File(FILEPATH + "corenlp_stopwords.txt"));
        while(scanner.hasNextLine()){
            stopwords.add(scanner.nextLine());
        }
        return stopwords;
    }
    //generalized to do both bigrams and trigrams.
    //if a bigram is a, b
    //trigram is a_b, c
    //if bigram, aFrequencies and bFrequencies are both the unigram frequency.
    //if traigram, aFrequencies is bigram frequencies and bFrequencies is unigram frequencies.
    public static HashMap<String, Integer> filterGrams(ArrayList<String> stopwords, HashMap<String, Integer> aFrequencies, HashMap<String, Integer> bFrequencies, HashMap<String, HashMap<String, Integer>> unfilteredGrams){
        HashMap<String, Integer> filteredGrams = new HashMap<>();
        //do chi squared
        String[] firstWords = unfilteredGrams.keySet().toArray(new String[0]);
        double aN = totalFreq(aFrequencies);
        double bN = totalFreq(bFrequencies);
        ChiSquareTest cst = new ChiSquareTest();
        for(String aTerm: firstWords){
            HashMap<String, Integer> aTermFreqs = unfilteredGrams.get(aTerm);
            String[] secondWords = aTermFreqs.keySet().toArray(new String[0]);
            for(String bTerm: secondWords){
                double ATotal = aFrequencies.get(aTerm);
                double BTotal = bFrequencies.get(bTerm);
                double NotATotal = aN - ATotal;
                double NotBTotal = bN - BTotal;
                //System.out.println("Atotal " + ATotal + " BTotal " + BTotal + " NotATotal " + NotATotal + " NotBTotal " + NotBTotal);
                double AB_O = aTermFreqs.get(bTerm);
                double ANotB_O = ATotal - AB_O;
                double NotAB_O = BTotal - AB_O;
                double NotANotB_O = NotATotal - NotAB_O;
                //System.out.println("AB_O " + AB_O + " ANotB_O " + ANotB_O + " NotAB_O " + NotAB_O + " NotANotB_O " + NotANotB_O);
                double AB_E = ATotal/aN * BTotal/bN * bN;
                double ANotB_E = ATotal/aN * NotBTotal/bN * bN;
                double NotAB_E = NotATotal/aN * BTotal/bN * bN;
                double NotANotB_E = NotATotal/aN * NotBTotal/bN * bN;
                //System.out.println("AB_E " + AB_E + " ANotB_E " + ANotB_E + " NotAB_E " + NotAB_E + " NotANotB_E " + NotANotB_O);
                double[] expected = {AB_E, ANotB_E, NotAB_E, NotANotB_E};
                long[] observed = {(long)AB_O, (long)ANotB_O, (long)NotAB_O, (long)NotANotB_O};
                //check for stopwords
                //for trigrams, not cheacking if middle term in stopwords.
                //only testing the first term. Looked at the results, makes more sense this way.
                ArrayList<String> testTerms = new ArrayList<>();
                testTerms.add(bTerm);
                if(aTerm.contains("_") && aTerm.length() > 2){
                    String[] terms = aTerm.split("_");
                    testTerms.addAll(Arrays.asList(terms));
                } else{
                    testTerms.add(aTerm);
                }
                boolean hasStopwords = false;
                for(String term: testTerms){
                    if(stopwords.contains(term)){
                        hasStopwords = true;
                    }
                }
                if(cst.chiSquareTest(expected, observed,.01) && !hasStopwords){
                    String bigram = aTerm + "_" + bTerm;
                    filteredGrams.put(bigram, aTermFreqs.get(bTerm));
                }
            }
        }
        return filteredGrams;
    }
    //from text returns a HashMap<String, Integer> term->count
    //find bigrams here.
    public static HashMap<String, Integer> getFrequencies(ArrayList<String> stopwords, String text){
        HashMap<String, Integer> frequencies = new HashMap<>();
        HashMap<String, Integer> bigramFrequencies = new HashMap<>();
        String comments[] = text.split("\\n");
        //word1
        HashMap<String, HashMap<String, Integer>>  unfilteredBigrams =  new HashMap<>();
        HashMap<String, HashMap<String, Integer>> unfilteredTrigrams = new HashMap<>();
        for(String comment: comments){
            String previous = null;
            String preprevious = null; //for trigrams. Previous of previous.
            PTBTokenizer<CoreLabel> ptbt = new PTBTokenizer<CoreLabel>(new StringReader(comment),
                    new CoreLabelTokenFactory(), "");
            while (ptbt.hasNext()) {
                //unigram stuff
                String label = ptbt.next().value();
                label = label.toLowerCase();
                if(! frequencies.containsKey(label)){
                    frequencies.put(label, 1);
                }else{
                    frequencies.put(label, frequencies.get(label) + 1);
                }
                //bigram stuff;
                if(previous != null){
                    if (unfilteredBigrams.containsKey(previous)) {
                        HashMap<String, Integer> freqs = unfilteredBigrams.get(previous);
                        if(freqs.containsKey(label)){
                            freqs.put(label, freqs.get(label) + 1);
                            unfilteredBigrams.put(previous, freqs);
                        }else{
                            freqs.put(label, 1);
                            unfilteredBigrams.put(previous, freqs);
                        }
                    } else {
                        HashMap<String, Integer> freqs = new HashMap<>();
                        freqs.put(label , 1);
                        unfilteredBigrams.put(previous, freqs);
                    }
                    //bigram frequencies for trigrams.
                    String bigram = previous + "_" + label;
                    if(bigramFrequencies.containsKey(bigram)){
                        bigramFrequencies.put(bigram, bigramFrequencies.get(bigram) + 1);
                    }else{
                        bigramFrequencies.put(bigram, 1);
                    }

                }
                //trigram stuff.
                if(preprevious != null){
                    String previousBigram = preprevious + "_" + previous;
                    if (unfilteredTrigrams.containsKey(previousBigram)) {
                        HashMap<String, Integer> freqs = unfilteredTrigrams.get(previousBigram);
                        if(freqs.containsKey(label)){
                            freqs.put(label, freqs.get(label) + 1);
                            unfilteredTrigrams.put(previousBigram, freqs);
                        }else{
                            freqs.put(label, 1);
                            unfilteredTrigrams.put(previousBigram, freqs);
                        }
                    } else {
                        HashMap<String, Integer> freqs = new HashMap<>();
                        freqs.put(label , 1);
                        unfilteredTrigrams.put(previousBigram, freqs);
                    }

                }
                preprevious = previous;
                previous = label;
            }
        }

        HashMap<String, Integer> filteredBigrams = filterGrams(stopwords, frequencies, frequencies, unfilteredBigrams);
        HashMap<String, Integer> filteredTrigrams = filterGrams(stopwords, bigramFrequencies, frequencies, unfilteredTrigrams);
        String[] bigrams = filteredBigrams.keySet().toArray(new String[0]);
        for(String bigram: bigrams){
            //for the rare case where "_" is a common spelling convention of a bigram
            // like maybe "trec_eval" vs "trec eval" or something
            if(frequencies.containsKey(bigram)){
                frequencies.put(bigram, frequencies.get(bigram) + filteredBigrams.get(bigram));
            }else{
                frequencies.put(bigram, filteredBigrams.get(bigram));
            }
        }
        String[] trigrams = filteredTrigrams.keySet().toArray(new String[0]);
        for(String trigram: trigrams){
            //for the rare case where "_" is a common spelling convention of a bigram
            // like maybe "trec_eval" vs "trec eval" or something
            if(frequencies.containsKey(trigram)){
                frequencies.put(trigram, frequencies.get(trigram) + filteredTrigrams.get(trigram));
            }else{
                frequencies.put(trigram, filteredTrigrams.get(trigram));
            }
        }
        return frequencies;
    }
    //writes a frequency file for a subreddit.
    public static void writeFrequencyFile(ArrayList<String> stopwords, String text, String subredditName) throws IOException{
        System.out.println("writing " +  FILEPATH + "index\\" + subredditName + "_frequencies.txt...");
        FileWriter writer = new FileWriter(FILEPATH + "index\\" + subredditName + "_frequencies.txt");
        HashMap<String, Integer> freqs = getFrequencies(stopwords, text);
        Object[] keys = freqs.keySet().toArray();
        for(Object keyObj: keys){
            String key = (String) keyObj;
            writer.write(key + "\t" + freqs.get(key) + "\n");
        }
        writer.close();
        System.out.println("Done");
    }
    //reads a frequency file. returns HashMap<String, Integer> String->Integer
   public static HashMap<String, Integer> readFrequencyFile(String inputFilePath) throws FileNotFoundException {
        HashMap<String, Integer> freqs = new HashMap<>();
        File file = new File(inputFilePath);
        Scanner scanner = new Scanner(file);
        while(scanner.hasNext()){
            String line = scanner.nextLine();
            String[] freq = line.split("\\t");
            freqs.put(freq[0], Integer.parseInt(freq[1]));
        }
        return freqs;
   }


    //reads json files and creates an index, a frequency file for every subreddit in the corpus.
    public static void writeInvertedIndex(String basePath, String subredditPath, String subredditName) throws IOException, ParseException {
        ArrayList<String> stopwords = getStopwords();
        HashMap<String, String> docs = readJSON(basePath);
        HashMap<String, String> sr = readJSONL(subredditPath);
        String subredditText = sr.get(subredditName);
        writeFrequencyFile(stopwords, subredditText, subredditName);
        Object[] keys = docs.keySet().toArray();
        docs.put(subredditName, subredditText);
        String bigStr = "";
        for(Object keyObj: keys) {
            String key = (String) keyObj;
            bigStr += docs.get(key);
            writeFrequencyFile(stopwords, docs.get(key),key);
        }
        //writeFrequencyFile(stopwords, bigStr, "reddit-corpus-small");
    }
    public static ArrayList<HashMap<String, Integer>> readInvertedIndex(String indexPath) throws FileNotFoundException {
        ArrayList<HashMap<String, Integer>> index = new ArrayList<>();
        File folder = new File(indexPath);
        for (final File fileEntry : folder.listFiles()) {
                index.add(readFrequencyFile(fileEntry.getPath()));
        }
        return index;
    }
    public static HashMap<String, Integer> combineIndexes(ArrayList<HashMap<String, Integer>> index){
        HashMap<String, Integer> dictionary = new HashMap<>();
        for(HashMap<String, Integer> doc: index){
            Object[] keys = doc.keySet().toArray();
            for(Object key: keys){
                if(!dictionary.containsKey(key)){
                    dictionary.put((String)key, doc.get(key));
                }else{
                    dictionary.put((String)key, dictionary.get(key) + doc.get(key));
                }
            }
        }
        return dictionary;
    }
    //from a frequency hashmap, finds the length of a document.
    public static int getLength(HashMap<String, Integer> doc){
        int sum = 0;
        Object[] keys = doc.keySet().toArray();
        for(Object key: keys){
            sum += doc.get(key);
        }
        return sum;
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

    public static void analyze(String indexPath, String subredditName) throws IOException, ParseException {
        System.out.println("Analyzing...");
        System.out.println("\tReading InvertedIndex");
        ArrayList<HashMap<String, Integer>> index = readInvertedIndex(indexPath);
        System.out.println("\tReading subreddit file");
        HashMap<String, Integer> comparisonSubreddit = readFrequencyFile(FILEPATH + "index\\" + subredditName + "_frequencies.txt");
        Object[] keys = comparisonSubreddit.keySet().toArray();
        double n = index.size();
        ArrayList<TermResult> termResults = new ArrayList<>();
        System.out.println("Performing tf-idf on terms");
        double length = getLength(comparisonSubreddit);
        double mean = 0;
        double numTerm = comparisonSubreddit.size();
        for(Object key: keys){
            String t = (String) key;
            double k = 0;
            for(HashMap<String, Integer> doc: index){
                if(doc.containsKey(t)){
                    k += 1;
                }
            }
            double c = comparisonSubreddit.get(t);
            //TF-IDF
            double f = c/length + Math.log(1 + n/k);
            mean += f/numTerm;
            TermResult termResult = new TermResult(t, c/length, Math.log(1+n/k));
            termResults.add(termResult);
        }
        Collections.sort(termResults);
        Collections.reverse(termResults);
        //write all results to file
        FileWriter allWriter = new FileWriter(FILEPATH + subredditName + "_all_results.txt");
        int i = 0;
        for(TermResult termResult: termResults){
            allWriter.write((1 + i) + "\t" + termResult.getTerm() + "\t" + termResult.getF() + "\n");
            i++;
        }
        allWriter.close();
        double sd = findStandardDeviation(termResults, mean);
        i = 0;
        ArrayList<TermResult> termResultsTop = new ArrayList<>();
        double totalTop = 0;
        double NTop = 0;
        double prev = termResults.get(0).F; //just has to be big, tf-idf tends to be between 0 and 5.0
        System.out.println((prev - termResults.get(i).F));
        while(prev - termResults.get(i).F < .5){
            TermResult termResult = termResults.get(i);
            termResultsTop.add(termResult);
            totalTop += termResult.F;
            NTop += 1;
            i++;
            prev = termResult.F;
        }
        double meanTop = totalTop/NTop;
        double sdTop = findStandardDeviation(termResultsTop, meanTop);
        System.out.println("meanTop: " + meanTop + " sdTop: " + sdTop);
        i = 0;
        FileWriter topWriter = new FileWriter(FILEPATH + subredditName + "_top_results.txt");
        FileWriter evaluationWriter = new FileWriter(FILEPATH + subredditName + "_evaluation.txt");
        System.out.println("rank\tterm\tTF-IDF");
        topWriter.write("rank\tterm\tTF-IDF\n");
        while(termResultsTop.get(i).F > meanTop + 2 * sdTop && i + 1 < termResultsTop.size()){
            TermResult termResult = termResults.get(i);
            System.out.println((1 + i) + "\t" + termResult.getTerm() + "\t" + termResult.getF());
            topWriter.write((1 + i) + "\t" + termResult.getTerm() + "\t" + termResult.getF() + "\n");
            evaluationWriter.write((1 + i) + "\t" + termResult.getTerm() + "\t" + "\n");
            i++;
        }
        topWriter.close();
    }
    public static void main(String[] args) throws IOException, ParseException {
        

        String subreddit = "IndianaUniversity";
        writeInvertedIndex(FILEPATH + "reddit-corpus-small\\utterances.json", FILEPATH + subreddit + "\\utterances.jsonl", subreddit);
        analyze(FILEPATH + "index", subreddit);
    }
}
