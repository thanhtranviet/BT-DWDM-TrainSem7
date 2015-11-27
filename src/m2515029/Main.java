package m2515029;

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
		String input_file = "/home/thanh/workspace/TrainSem7/credit-g.arff";
		//String input_file = "D://credit-g.arff";
		// read data
		Instances data = DataSource.read(input_file);
		// class is the last attribute
		data.setClassIndex(data.numAttributes() - 1);

		// manually create train and test set
		int trainSize = (int) Math.round(data.numInstances() * 70 / 100);
		int testSize = data.numInstances() - trainSize;
		Instances data70 = new Instances(data, 0, trainSize);
		Instances data30 = new Instances(data, trainSize, testSize);

		double maxAUC = 0;
		// k-fold = 10
		int folds = 3;
		int seed = 1;
		// Try with DT
		String[] treeoptions = new String[2];
		treeoptions[0] = "-U"; // unpruned tree
		treeoptions[1] = "-i";
		Classifier tree = new J48(); // new instance of tree
		tree.setOptions(treeoptions);
		Evaluation evalDT = new Evaluation(data70);
		evalDT.crossValidateModel(tree, data70, folds, new Random(seed));
		evalDT = new Evaluation(data);
		tree.buildClassifier(data70);
		evalDT.evaluateModel(tree, data30);

		// try with KNN
		int bestk = 1;
		for (int k = 1; k < 20; k++) {
			seed = k;
			Evaluation eval = new Evaluation(data70);
			IBk knn = new IBk(k);
			eval.crossValidateModel(knn, data70, folds, new Random(seed));
			if (maxAUC < eval.weightedAreaUnderROC()) {
				maxAUC = eval.weightedAreaUnderROC();
				bestk = k;
			}
		}
		Evaluation evalKNN = new Evaluation(data);
		IBk knn = new IBk(bestk);
		knn.buildClassifier(data70);
		evalKNN.evaluateModel(knn, data30);
		
		// Try with SVM
		int typeOfKernelFunction = 2;
		double cNumber = 1;
		int bestTypeOfKernelFunction = 2;
		double bestCNumber = 1;
		maxAUC=0;

		// cnumber from 0.5->1.5
		for (int i = 0; i < 10; i++) {
			cNumber += i;
			// kernel type from 0-3
			for (int j = 2; j <= 3; j++) {
				typeOfKernelFunction = j;
				seed = i * j;
				Evaluation eval = new Evaluation(data70);
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
				eval.crossValidateModel(svm, data70, folds, new Random(seed));
				System.out.println(eval
						.toClassDetailsString("=== Class detail ==="));

				if (maxAUC < eval.weightedAreaUnderROC()) {
					maxAUC = eval.weightedAreaUnderROC();
					bestTypeOfKernelFunction = typeOfKernelFunction;
					bestCNumber = cNumber;
				}
			}
		}

		// Test with best value for SVM
		// option for SVM
		maxAUC=0;
		String[] options = new String[6];
		options[0] = "-K";
		options[1] = String.valueOf(bestTypeOfKernelFunction);
		options[2] = "-C";
		options[3] = String.valueOf(bestCNumber);
		options[4] = "-H";
		options[5] = "0";

		LibSVM bestSVM = new LibSVM();
		bestSVM.setOptions(options);
		bestSVM.buildClassifier(data70);
		Evaluation evalSVM = new Evaluation(data);
		evalSVM.evaluateModel(bestSVM, data30);

		System.out.println("==============SVM===============");
		System.out.println("Found kernel: " + bestTypeOfKernelFunction + " C: "
				+ bestCNumber);
		System.out
				.println(evalSVM.toClassDetailsString("=== Class detail ==="));
		System.out.println(evalSVM.toSummaryString("=== Summary ===", false));
		System.out.println("==============KNN================");
		System.out.println("Found best k: " + bestk);
		System.out
				.println(evalKNN.toClassDetailsString("=== Class detail ==="));
		System.out.println(evalKNN.toSummaryString("=== Summary ===", false));
		System.out.println("==============DT================");
		System.out.println(evalDT.toClassDetailsString("=== Class detail ==="));
		System.out.println(evalDT.toSummaryString("=== Summary ===", false));
	}
}
