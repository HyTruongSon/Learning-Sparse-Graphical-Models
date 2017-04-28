// Software: Training/Testing Neural Network (1 hidden layer)
// Author: Hy Truong Son
// Major: PhD. Computer Science
// Institution: Department of Computer Science, The University of Chicago
// Email: hytruongson@uchicago.edu
// Website: http://people.inf.elte.hu/hytruongson/
// Copyright 2017 (c) Hy Truong Son. All rights reserved. Only use for academic purposes.

import java.io.BufferedInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.File;
import java.io.PrintWriter;
import java.io.FileWriter;
import java.util.Scanner;
import java.util.ArrayList;
import MyLib.MLP;
import MyLib.MLP_ReLU;

public class NeuralNets {
	
	// +----------+
	// | Datasets |
	// +----------+

	static String TrainingInputFileName = "train_input.dat";
	static String TrainingOutputFileName = "train_target.dat";
	static String TestingInputFileName = "test_input.dat";
	static String TestingOutputFileName = "test_target.dat";

	// +-----------------------+
	// | Files to save results |
	// +-----------------------+

	static ArrayList<String> ModelFileNames;
	static String ReportFileName;
	
	// +--------------------+
	// | Constants of model |
	// +--------------------+
	
	static int nInput;
	static int nHidden;
	static int nOutput;
	
	// +----------------------------------------+
	// | MultiLayer Perceptron - Neural Network |
	// +----------------------------------------+

	static MLP_ReLU myNet;
	
	// +---------------------+
	// | Training parameters |
	// +---------------------+

	static int Epochs;
	static double LearningRate;
	static double Momentum = 0.9;      // Momentum for stochastic learning process
	
	// +--------------------------------------+
	// | Data structures to save the datasets |
	// +--------------------------------------+

	public static class Sample {
		public double input[];
		public double output[];
	}

	static ArrayList<Sample> trainData;
	static ArrayList<Sample> testData;

	static int nTraining;
	static int nTesting;
	
	static boolean printOutput;
	static ArrayList<Integer> performance;
	
	// +---------------------------------+
	// | Input the options from the user |
	// +---------------------------------+

	private static void inputParameters() {
		Scanner scanner = new Scanner(System.in);

		nHidden = 0;
		while (true) {
			System.out.print("Number of hidden neurons (for example, 128): ");
			nHidden = scanner.nextInt();
			if (nHidden <= 0) {
				System.out.println("The number of hidden neurons must be greater than 0");
			} else {
				break;
			}
		}

		Epochs = 0;
		while (true) {
			System.out.print("Epochs that is the number of times training the dataset (for example, 10): ");
			Epochs = scanner.nextInt();
			if (Epochs <= 0) {
				System.out.println("The number of epochs must be greater than 0");
			} else {
				break;
			}
		}

		LearningRate = 0.0;
		while (true) {
			System.out.print("Learning rate (for example, 0.01): ");
			LearningRate = scanner.nextDouble();
			if (LearningRate <= 0.0) {
				System.out.print("The learning rate must be greater than 0");
			} else {
				break;
			}
		}
		scanner.nextLine();
		
		String ModelFileName = "saves/model-ReLU-nHidden-" + Integer.toString(nHidden) + "-Epochs-" + Integer.toString(Epochs) + "-LearningRate-" + Double.toString(LearningRate);
		ModelFileNames = new ArrayList<>();
		ModelFileNames.add(ModelFileName + "-Layer-0.dat");
		ModelFileNames.add(ModelFileName + "-Layer-1.dat");

		System.out.println();
		System.out.println("Model file name: " + ModelFileNames.get(0) + ", " + ModelFileNames.get(1));

		ReportFileName = "saves/report-ReLU-nHidden-" + Integer.toString(nHidden) + "-Epochs-" + Integer.toString(Epochs) + "-LearningRate-" + Double.toString(LearningRate) + ".dat";
		System.out.println("Report file name: " + ReportFileName);
		System.out.println();

		printOutput = false;
		while (true) {
			System.out.print("Do you want to see the training for each example (yes/no)? ");
			String answer = scanner.nextLine();
			if (answer.equals("yes")) {
				printOutput = true;
				break;
			}
			if (answer.equals("no")) {
				printOutput = false;
				break;
			}
			System.out.println("Answer 'yes' or 'no' only!");
		}
	}

	// +---------------------------------------+
	// | Loading the datasets (inputs/targets) |
	// +---------------------------------------+

	private static void loadTrainingData() throws IOException {
		trainData = new ArrayList<>();

		Scanner input = new Scanner(new File(TrainingInputFileName));
		Scanner output = new Scanner(new File(TrainingOutputFileName));

		nTraining = input.nextInt();
		nInput = input.nextInt();

		output.nextInt();
		nOutput = output.nextInt();

		for (int i = 0; i < nTraining; ++i) {
			Sample sample = new Sample();
			sample.input = new double [nInput];
			sample.output = new double [nOutput];

			for (int j = 0; j < nInput; ++j) {
				sample.input[j] = input.nextDouble();
			}

			for (int j = 0; j < nOutput; ++j) {
				sample.output[j] = output.nextDouble();
			}

			trainData.add(sample);
		}

		input.close();
		output.close();
	}

	private static void loadTestingData() throws IOException {
		testData = new ArrayList<>();

		Scanner input = new Scanner(new File(TestingInputFileName));
		Scanner output = new Scanner(new File(TestingOutputFileName));

		nTesting = input.nextInt();
		nInput = input.nextInt();

		output.nextInt();
		nOutput = output.nextInt();

		for (int i = 0; i < nTesting; ++i) {
			Sample sample = new Sample();
			sample.input = new double [nInput];
			sample.output = new double [nOutput];

			for (int j = 0; j < nInput; ++j) {
				sample.input[j] = input.nextDouble();
			}

			for (int j = 0; j < nOutput; ++j) {
				sample.output[j] = output.nextDouble();
			}

			testData.add(sample);
		}

		input.close();
		output.close();
	}

	// +------------------------+
	// | Network initialization |
	// +------------------------+

	private static void networkInitialization() {
		myNet = new MLP_ReLU(nInput, nHidden, nOutput);
		myNet.setEpochs(10);
		myNet.setLearningRate(LearningRate);
		myNet.setMomentum(Momentum);
	}

	// +----------------------------+
	// | Training / Testing Process |
	// +----------------------------+

	private static void process() throws IOException {
		performance = new ArrayList<>();
		double input[] = new double [nInput];
		double output[] = new double [nOutput];

		double best_accuracy = 0.0;
		for (int epoch = 0; epoch < Epochs; ++epoch) {
			System.out.println();
			System.out.println("----------------------------------------------------------------------------------");
			System.out.println("Epoch " + Integer.toString(epoch));
			System.out.println("Training");

			for (int i = 0; i < trainData.size(); ++i) {
				Sample sample = trainData.get(i);
				if ((i + 1) % 1000 == 0) {
					System.out.println("Done training for " + Integer.toString(i + 1) + "/" + Integer.toString(trainData.size()) + " samples");
				}
			
				if (printOutput) {
					System.out.println();
					System.out.println("Training sample " + Integer.toString(i) + ":");
				}

				// Input
				for (int j = 0; j < nInput; ++j) {
					input[j] = sample.input[j];
				}

				// Output
				for (int j = 0; j < nOutput; ++j) {
					output[j] = sample.output[j];
				}

				// Neural Network Learning
				double loss = myNet.StochasticLearning(input, output);

				if (printOutput) {
					System.out.println("    Squared loss = " + Double.toString(loss));
				}
			}

			System.out.println("Testing");
			int nCorrect = 0;
			for (int i = 0; i < nTesting; ++i) {
				Sample sample = testData.get(i);
				for (int j = 0; j < nInput; ++j) {
					input[j] = sample.input[j];
				}

				// Neural network prediction
				myNet.Predict(input, output);

				for (int j = 0; j < nOutput; ++j) {
					if ((output[j] >= 0.5) && (sample.output[j] > 0.0)) {
						++nCorrect;
					}
					if ((output[j] < 0.5) && (sample.output[j] == 0)) {
						++nCorrect;
					}
				}
			}

			double accuracy = (double)(nCorrect) / (double)(nTesting * nOutput);
			System.out.println("Testing accuracy = " + Integer.toString(nCorrect) + "/" + Integer.toString(nTesting * nOutput) + " = " + Double.toString(accuracy));
			performance.add(nCorrect);

			if (accuracy < best_accuracy) {
				System.out.println("Early stopping!");
				break;
			} else {
				best_accuracy = accuracy;
				System.out.println("Save model to file");
				myNet.writeWeights(ModelFileNames);
			}
		}
	}

	// +--------------------+
	// | Making the summary |
	// +--------------------+

	private static void summary() throws IOException {
		System.out.println();
		System.out.println("----------------------------------------------------------------------------------");
		System.out.println("Summary:");
		int nTesting = testData.size();
		PrintWriter report = new PrintWriter(new FileWriter(ReportFileName));
		for (int epoch = 0; epoch < performance.size(); ++epoch) {
			double accuracy = (double)(performance.get(epoch)) / nTesting;
			System.out.println("Epoch " + Integer.toString(epoch) + ": Accuracy = " + Integer.toString(performance.get(epoch)) + "/" + Integer.toString(nTesting * nOutput) + " = " + Double.toString(accuracy));
			report.println("Epoch " + Integer.toString(epoch) + ": Accuracy = " + Integer.toString(performance.get(epoch)) + "/" + Integer.toString(nTesting * nOutput) + " = " + Double.toString(accuracy));
		}
		report.close();
	}

	// +--------------+
	// | Main Program |
	// +--------------+

	public static void main(String args[]) throws IOException {	
		// Input parameters (options) from the user
	    inputParameters();

	    // Loading the training dataset
	    loadTrainingData();

	    // Loading the testing dataset
	    loadTestingData();

	    // Neural network object creation
	    networkInitialization();

	    // Training / Testing process 
	    process();

	    // Making the summary
	    summary();
	}
	
}
