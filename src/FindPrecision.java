import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;

public class FindPrecision {
    public static String FILEPATH = "D:\\L645_corpora\\project_corpora\\evaluation\\";
    public static void evaluatePrecision(String filePath) throws FileNotFoundException {
        Scanner scanner = new Scanner(new File(filePath));
        double correct = 0;
        double total = 0;
        while(scanner.hasNextLine()){
            String line = scanner.nextLine();
            if(line.equals("")){
                continue;
            }
            String[] lineLst = line.split("\t");
            int l = lineLst.length - 1;
            Integer answer = Integer.parseInt(lineLst[l]);
            if(answer == 1){
                correct ++;
            }
            total ++;
            if(total == 5){
                System.out.println("Precision@5: " + correct/total);
            }
            if(total == 25){
                System.out.println("Precision@25: " + correct/total);
            }
            if(total == 50){
                System.out.println("precision@50: " + correct/total);
            }
            if(total == 100){
                System.out.println("Precision@100: " + correct/total);
            }
            if(total == 150){
                System.out.println("Precision@150: " + correct/total);
            }
            if(total == 200){
                System.out.println("Precision@200: " + correct/total);
            }
        }
        System.out.println("Precision @ all: " + correct/total);

    }

    public static void main(String[] args) throws FileNotFoundException {
        System.out.println("r/LanguageTechnology");
        evaluatePrecision(FILEPATH + "LanguageTechnology_evaluation_save2.txt");
        System.out.println("r/StarWarsEU");
        evaluatePrecision(FILEPATH + "StarWarsEU_evaluation_save2.txt");
        System.out.println("r/IndianaUniversity");
        evaluatePrecision(FILEPATH + "IndianaUniversity_evaluation_save2.txt");
        System.out.println("r/IndianaUniversity + colleges");
        evaluatePrecision(FILEPATH + "IndianaUniversity_colleges_evaluation_save2.txt");

    }
}
