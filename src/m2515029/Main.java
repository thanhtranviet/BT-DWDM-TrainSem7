package m2515029;

import java.io.FileDescriptor;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.io.OutputStream;
import java.io.IOException;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class Main {

	/**
	 * @param args
	 * @throws Exception
	 */
	public static void main(String[] args) throws Exception {
		// input data
		String input_file = "/home/thanh/workspace/TrainSem7/TrainSem7.arff";
		// read data
		Instances datafull = DataSource.read(input_file);
		// class is the last attribute
		datafull.setClassIndex(datafull.numAttributes() - 1);
		
		//data full is too large so we will take the first 20% only
		int datasize = (int) Math.round(datafull.numInstances() * 20 / 100);
		Instances data = new Instances(datafull, 0, datasize);

		testWithSVM(data);
		testWithDT(data);
		testWithKNN(data);
	}

	private static void testWithDT(Instances data) throws Exception {
		double[] AUC_DT = new double[3];

		// cv 5-fold
		int folds = 5;
		// lap 3 lan, moi lan tao 1 bo train va set
		int round = 3;

		for (int counter = 0; counter < round; counter++) {
			System.out.println("=========== DT Round: " + counter +"==============================");
			Instances train = data.trainCV(round, counter);
			Instances test = data.testCV(round, counter);
			int seed = 1;
			String[] treeoptions = new String[2];
			treeoptions[0] = "-U"; // unpruned tree
			treeoptions[1] = "-i";
			Classifier tree = new J48(); // new instance of tree
			tree.setOptions(treeoptions);
			Evaluation evalDT = new Evaluation(train);
			evalDT.crossValidateModel(tree, train, folds, new Random(seed));
			evalDT = new Evaluation(data);
			tree.buildClassifier(train);
			evalDT.evaluateModel(tree, test);

			AUC_DT[counter] = evalDT.weightedAreaUnderROC();
			System.out.println("==============RESULT============================");
			System.out.println("AUC = " + AUC_DT[counter]);
		}

		double AUC_AVG_DT = 0;
		System.out.println("==============FINAL RESULT FOR DT==============");
		for (int counter = 0; counter < 3; counter++) {
			AUC_AVG_DT += AUC_DT[counter];
			System.out.println("AUC_KNN[" + counter + "]=" + AUC_DT[counter]);
		}
		System.out.println("AUC_AVG_KNN = " + AUC_AVG_DT/3);
	}
	private static void testWithKNN(Instances data) throws Exception {
		double[] AUC_KNN = new double[3];

		// cv 5-fold
		int folds = 5;
		// lap 3 lan, moi lan tao 1 bo train va set
		int round = 3;

		for (int counter = 0; counter < round; counter++) {
			System.out.println("=========== KNN Round: " + counter +"==============================");
			Instances train = data.trainCV(round, counter);
			Instances test = data.testCV(round, counter);

			int bestk = 1;
			double maxAUC = 0;
			int seed = 1;
			for (int k = 1; k < 20; k++) {
				seed = k;
				Evaluation eval = new Evaluation(train);
				IBk knn = new IBk(k);
				/*su dung ham crossValidateModel
				 * ham nay se lap folds lan, moi lan lap se lay 1/folds du lieu lam du 
				 * lieu test, va folds-1/folds du lieu lam du lieu train, moi lan lap se
				 * goi ham evaluateModel de danh gia mohinh train 
				 * sau khi lap duoc folds lan se lay gia tri trung binh cua folds lan lap
				 * lam gia tri cuoi cung de danh gia model do
				 */
				eval.crossValidateModel(knn, train, folds, new Random(seed));
				if (maxAUC < eval.weightedAreaUnderROC()) {
					maxAUC = eval.weightedAreaUnderROC();
					bestk = k;
				}
				
				System.out.println("k = " + k + "; AUC = " + eval.weightedAreaUnderROC());
			}
			Evaluation evalKNN = new Evaluation(data);
			IBk knn = new IBk(bestk);
			knn.buildClassifier(train);
			evalKNN.evaluateModel(knn, test);

			AUC_KNN[counter] = evalKNN.weightedAreaUnderROC();
			System.out.println("==============RESULT============================");
			System.out.println("Best K: " + bestk + "; AUC = " + AUC_KNN[counter]);
		}

		double AUC_AVG_KNN = 0;
		System.out.println("==============FINAL RESULT FOR KNN==============");
		for (int counter = 0; counter < 3; counter++) {
			AUC_AVG_KNN += AUC_KNN[counter];
			System.out.println("AUC_KNN[" + counter + "]=" + AUC_KNN[counter]);
		}
		System.out.println("AUC_AVG_KNN = " + AUC_AVG_KNN/3);
	}
	
	private static void testWithSVM(Instances data) throws Exception {
		double[] AUC_SVM = new double[3];

		// cv 5-fold
		int folds = 5;
		// lap 3 lan, moi lan tao 1 bo train va set
		int round = 3;

		for (int counter = 0; counter < round; counter++) {
			System.out.println("=========== SVM Round: " + counter +"==============================");
			Instances train = data.trainCV(round, counter);
			Instances test = data.testCV(round, counter);

			int typeOfKernelFunction = 2;
			double cNumber = 1;
			int bestTypeOfKernelFunction = 2;
			double bestCNumber = 0;
			double maxAUC = 0;
			int seed = 0;
			// cnumber from 0.1->2.0
			for (int i = 1; i < 20; i++) {
				cNumber = (double)i/10;
				// kernel type from 0-3
				for (int j = 2; j <= 3; j++) {
					System.setOut(new PrintStream(new OutputStream() {
		                @Override public void write(int b) throws IOException {}
		            }));
					typeOfKernelFunction = j;
					seed = i * j;
					Evaluation eval = new Evaluation(train);
					// option for SVM
					String[] svmoptions = new String[6];
					svmoptions[0] = "-K";
					svmoptions[1] = String.valueOf(typeOfKernelFunction);
					svmoptions[2] = "-C";
					svmoptions[3] = String.valueOf(cNumber);
					svmoptions[4] = "-H";
					svmoptions[5] = "0";
					LibSVM svm = new LibSVM();
					svm.setOptions(svmoptions);
					eval.crossValidateModel(svm, train, folds, new Random(seed));

					if (maxAUC < eval.weightedAreaUnderROC()) {
						maxAUC = eval.weightedAreaUnderROC();
						bestTypeOfKernelFunction = typeOfKernelFunction;
						bestCNumber = cNumber;
					}
					System.setOut(new PrintStream(new FileOutputStream(FileDescriptor.out)));
					System.out.println("C = " + cNumber +" K = " + typeOfKernelFunction + "; AUC = " + eval.weightedAreaUnderROC());
				}
			}
			String[] options = new String[6];
			options[0] = "-K";
			options[1] = String.valueOf(bestTypeOfKernelFunction);
			options[2] = "-C";
			options[3] = String.valueOf(bestCNumber);
			options[4] = "-H";
			options[5] = "0";

			LibSVM bestSVM = new LibSVM();
			bestSVM.setOptions(options);
			bestSVM.buildClassifier(train);
			Evaluation evalSVM = new Evaluation(data);
			evalSVM.evaluateModel(bestSVM, test);

			AUC_SVM[counter] = evalSVM.weightedAreaUnderROC();
			System.out.println("==============RESULT============================");
			System.out.println("C = " + bestCNumber +" K = " + bestTypeOfKernelFunction + "; AUC = " + AUC_SVM[counter]);
		}

		double AUC_AVG_SVM = 0;
		System.out.println("==============FINAL RESULT FOR SVM==============");
		for (int counter = 0; counter < 3; counter++) {
			AUC_AVG_SVM += AUC_SVM[counter];
			System.out.println("AUC_SVM[" + counter + "]=" + AUC_SVM[counter]);
		}
		System.out.println("AUC_AVG_SVM = " + AUC_AVG_SVM/3);
	}
}