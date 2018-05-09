/*	CS6200  Che Shian Hung  3/12/2018
Programming Assignment 2
Purpose: This program uses backpropagation algorithm to train the network for three classes.
	Each sample is represented as a three dimensional vector, and there are 30 sample data
	randomly generated for each class everytime the program runs. The first half of data
	is used for training, and the second half of data is used for testing the network
	after training. There are three fixed sphere centers that are used to generate the data, and
	the distances between them have to be greater than 10. The data will then be randomly
	generated inside each sphere with radius 2. The user can modify the global variables at the 
	top to configure the network setting and the dataset. If the problem is configured to be
	non-separable, the last sample for both training and testing for each class will
	be swap to the next class. For instance, the samples for class 1 will now have the samples
	for class 3. In addition, we can also easily modify the variables such as 
	TEST_ITERATION and INPUT_NUM to assess the network performance with different setting.
Architecture: There are mainly three steps in the program: data generation, network training,
	and testing. The whole network is a class, which contains all functionalities for running 
	backpropagation algorithm. Each step has been encapsulated in few functions. Also, for testing purpose,
	each function has been designed for reusability. For instance, we can display any information
	related to the network. In the network, backpropagation algorithm consists of mainly two functions:
	forwardPass and backwardPass, and forwardPass function can be also used in testing phase. 
Data Structure: For the data set, the information is generated and stored in statically allocated arrays.
	For the nodes and weights in the network, I created two structs to handle the information used in
	backpropagation algorithm. Inside the BackpropNetwork class, the network is presented as a bunch of 
	weights arrays and nodes arrays that are dynamically allocated with the setting, so that we can 
	easily adjust layer height and number of layers to observe the changes. While training and testing,
	the network class requires to take a data set as a parameter to increase cohesion between dataset 
	and running network. 
*/

#define _USE_MATH_DEFINES

// Import libraries and constants
#include<iostream>
#include<stdlib.h>
#include<time.h>
#include<cmath>
#include<string>

#define TEST_ITERATION 30			// Number of testing with differnt generated dataset
#define MODIFY_DATA false			// Switch for making the problem linearly separable/ non-separable
#define SCALE_INDEX 1				// Number and switch to scale the dataset

#define INITIAL_WEIGHT_MIN -0.5		// Minimum initial weight for the network
#define INITIAL_WEIGHT_MAX 0.5		// Maximum initial weight for the network
#define INPUT_NUM 3					// Number of input feature for the network
#define HIDDEN_LAYER_NUM 1			// Number of hidden layer for the network
#define HIDDEN_LAYER_HEIGHT 8		// Number of node in each hidden layer for the network
#define LEARNING_BASE 1.5				// The greek symbol in gradient decent algorithm (zeta...???)
#define LEARNING_RATE 0.3			// Amount of increased learning rate after each layer for the network
#define OUTPUT_NUM 3				// Number of output for the network
#define ERROR 0.01					// The error threshold for training
#define MAX_ITERATION 30000			// Iteration limit for training with backpropagation network

#define RADIUS 2					// Radius for the class sphere
#define TEST_SIZE 15				// Number of testing data in each class
#define CLASS_NUM 3					// Number of classes
#define DIMENSION 3					// Dimensionality for each sample
#define SAMPLE_SIZE 30				// Number of sample in each class
#define MODIFY_INDEX 15				// The index of sample data that is chose to be non-separable while training

using namespace std;

struct edge{
	double weight;					// The current weight
	double oldWeight;				// Weight before modification (used in backward pass)
};

struct node {
	double value;					// Input/output value stored in the node
	double delta;					// Delta value for the current node (used in backward pass)
};

class BackpropNetwork {
private:
	int inputNum;									// Number of input features
	int outputNum;									// Number of output nodes
	int trainIteration;								// Number of iteration for current training
	int hiddenLayerNum;								// Number of hidden layers
	int hiddenLayerHeight;							// Number of hidden layers' height
	int** testResult;								// Test result (class number) of the current test
	double accuracy;								// Accuracy of the current test
	double* error;									// Error vector for each forward pass (Output - target)
	double* targetOutput;							// Target output for current training data
	node *biasNodes;								// Bias nodes for the network
	node **networkNodes;							// Input nodes + hidden nodes + output nodes for the network
	edge **biasWeights;								// Bias weight for each bias node
	edge ***nodeWeights;							// All other weights for the rest of the nodes

	void destroyNetwork();							// Delete and reset all pointers
	void initializePointers();						// Initialize all pointers
	void resetPointers();							// Reset the pointer values
	void pointersToNull();							// Set all pointers to null
	void forwardPass();								// Forward pass 
	void backwardPass();							// Backward pass
	void setInput(double* inputVector);				// Set input values in the input nodes (network nodes)
	void setTarget(double* targetVector);			// Set output values in the output nodes (network nodes)
	int getClass();									// Determine the maximum value stored in the output nodes and return class number
	double sigmoidFunction(double net);				// Sigmoid function with an input x
	double randomDouble(double min, double max);	// Return a random double between a minimum and a maximum value

public:
	BackpropNetwork();										// Constructor
	~BackpropNetwork();										// Destructor

	void trian(double dataSet[][SAMPLE_SIZE][DIMENSION]);	// Train the network with data set
	void test(double dataSet[][SAMPLE_SIZE][DIMENSION]);	// Test the network with data set
	void displayStats();									// Display none pointer information in the network
	void displayAllStats();									// Display all stats about the network
	void displayError();									// Display error
	void displayTargetOutput();								// Display targetOutput
	void displayBias();										// Display values stored in bias nodes
	void displayNetwork();									// Display values stored in network nodes
	void displayBiasWeight();								// Display values stored in bias weight
	void displayNodeWeight();								// Display values stored in nwteork nodes
	void displayAccuracy();									// Dispaly accuracy
	void displayTestReport();								// Display test report
	int getIteration() { return trainIteration; }			// Return train iteration for the last training
	double getAccuracy() { return accuracy; }				// Return accuracy for the last training
};

// Declare global constant variable
const double sphereCenter[CLASS_NUM][DIMENSION] = { { 0, 15, 0 },{ 7.5, 5, 0 },{ -7.5, 5, 0 } };	// Hard coded sphere centers

// Declare global variables
double classSamples[CLASS_NUM][SAMPLE_SIZE][DIMENSION];											// Sample for all three classes, including training data and testing data
int resultClass[CLASS_NUM][SAMPLE_SIZE - TEST_SIZE];											// Captures the testing result after testing with trained classifiers

// Define global fuctions
void generateAllSamples();																	// Generate samples randomly for all classes
void generateRandomSamples(double samples[SAMPLE_SIZE][DIMENSION], int classNum);			// Generate samples randomly for a specific class
void displayAllSamples();																	// Display samples for all classes
void displaySamples(int classNum);															// Display samples for a specific class
void modifyTrainedData(int modifyIndex);
void scaleAllSamples();

int main() {
	srand(time(NULL));

	double iterationTotal = 0;	
	double accuracyTotal = 0;
	double totalTime = 0;
	BackpropNetwork n;								// Create a new network

	for (int i = 0; i < TEST_ITERATION; i++) {
		generateAllSamples();						// Generate sample data and display for each class
		if (SCALE_INDEX != 1) scaleAllSamples();	// Scale the sample data if required
		if (MODIFY_DATA) {							// Swap specific samples to have non-separable problem
			modifyTrainedData(MODIFY_INDEX);		// Modify the 15th sample for each class
			modifyTrainedData(SAMPLE_SIZE);			// Modify the last sample for each class
		}
		//displayAllSamples();

		clock_t start = clock();
		n.trian(classSamples);						// Train the netwrok with dataset
		clock_t end = clock();
		totalTime += (double)(end - start) / CLOCKS_PER_SEC;

		n.test(classSamples);						// Test the network with dataset
		n.displayAccuracy();						// Display accuracy for testing result
		//n.displayAllStats();
		//n.displayTestReport();
		iterationTotal += n.getIteration();
		accuracyTotal += n.getAccuracy();
	}

	printf("Layer num: %d\nNode num: %d\n", HIDDEN_LAYER_NUM, HIDDEN_LAYER_HEIGHT);
	printf("Average iteration: %5.2f\n", iterationTotal / TEST_ITERATION);
	printf("Average accuracy: %5.2f\n", accuracyTotal / TEST_ITERATION);
	printf("Average training time: %5.2f\n\n", totalTime / TEST_ITERATION);
	iterationTotal = 0;
	accuracyTotal = 0;
	totalTime = 0;

	system("pause");
	return 0;
}

void generateAllSamples() {
	for (int i = 0; i < 3; i++) {
		generateRandomSamples(classSamples[i], i + 1);
	}
}

void generateRandomSamples(double sample[SAMPLE_SIZE][DIMENSION], int classNum) {
	for (int j = 0; j < SAMPLE_SIZE; j++) {
		double theta = rand() % 6282 / double(1000);
		double phi = rand() % 3141 / double(1000) - 1.5705;

		sample[j][0] = sphereCenter[classNum - 1][0] + RADIUS * cos(theta) * cos(phi);
		sample[j][1] = sphereCenter[classNum - 1][1] + RADIUS * sin(phi);
		sample[j][2] = sphereCenter[classNum - 1][2] + RADIUS * sin(theta) * cos(phi);
	}
}

void displayAllSamples() {
	printf("display all samples:\n");
	for (int i = 1; i < DIMENSION + 1; i++) displaySamples(i);
	printf("\n\n");
}

void displaySamples(int classNum) {
	printf("display class %d samples:\n", classNum);
	for (int i = 0; i < SAMPLE_SIZE; i++) {
		if ((i == MODIFY_INDEX - 1 && MODIFY_DATA) || (i == SAMPLE_SIZE - 1 && MODIFY_DATA)) printf("****");
		for (int j = 0; j < DIMENSION; j++) {
			if (j != 2)
				printf("%7.3f, ", classSamples[classNum - 1][i][j]);
			else
				printf("%7.3f\n", classSamples[classNum - 1][i][j]);
		}
	}
	printf("---------------------------------\n\n");
}

void modifyTrainedData(int modifyIndex) {
	modifyIndex--;
	double class12Sample[2][DIMENSION] = { { classSamples[0][modifyIndex][0], classSamples[0][modifyIndex][1], classSamples[0][modifyIndex][2] },{ classSamples[1][modifyIndex][0], classSamples[1][modifyIndex][1], classSamples[1][modifyIndex][2] } };
	for (int i = 0; i < CLASS_NUM; i++) {
		for (int j = 0; j < DIMENSION; j++) {
			if (i == 0) classSamples[i][modifyIndex][j] = classSamples[2][modifyIndex][j];
			else classSamples[i][modifyIndex][j] = class12Sample[i - 1][j];
		}
	}
}

void scaleAllSamples() {
	for (int i = 0; i < CLASS_NUM; i++) {
		for (int j = 0; j < SAMPLE_SIZE; j++) {
			for (int k = 0; k < DIMENSION; k++) {
				classSamples[i][j][k] *= SCALE_INDEX;
			}
		}
	}
}

BackpropNetwork::BackpropNetwork() {
	inputNum = INPUT_NUM;
	outputNum = OUTPUT_NUM;
	hiddenLayerNum = HIDDEN_LAYER_NUM;
	hiddenLayerHeight = HIDDEN_LAYER_HEIGHT;
	accuracy = -1;
	trainIteration = -1;
	pointersToNull();
	initializePointers();
	resetPointers();
}

BackpropNetwork::~BackpropNetwork() {
	destroyNetwork();
}

void BackpropNetwork::destroyNetwork() {
	for (int i = 0; i < CLASS_NUM; i++)
		delete[] testResult[i];
	delete[] testResult;
	delete[] error;
	delete[] targetOutput;
	delete[] biasNodes;
	for (int i = 0; i < hiddenLayerNum + 2; i++) 
		delete[] networkNodes[i];
	delete[] networkNodes;
	for (int i = 0; i < hiddenLayerNum + 1; i++) {
		delete[] biasWeights[i];
		int height = hiddenLayerHeight;
		if (i == 0) height = inputNum;
		for (int j = 0; j < height; j++)
			delete[] nodeWeights[i][j];
		delete[] nodeWeights[i];
	}
	delete[] biasWeights;
	delete[] nodeWeights;

	pointersToNull();
}

void BackpropNetwork::initializePointers() {
	testResult = new int*[CLASS_NUM];
	for (int i = 0; i < CLASS_NUM; i++) testResult[i] = new int[TEST_SIZE];
	error = new double[outputNum];
	targetOutput = new double[outputNum];
	biasNodes = new node[hiddenLayerNum + 1];
	networkNodes = new node*[hiddenLayerNum + 2];
	networkNodes[0] = new node[inputNum];
	networkNodes[hiddenLayerNum + 1] = new node[outputNum];
	biasWeights = new edge*[hiddenLayerNum + 1];
	nodeWeights = new edge**[hiddenLayerNum + 1];
	nodeWeights[0] = new edge*[inputNum];
	for (int i = 0; i < inputNum; i++) nodeWeights[0][i] = new edge[hiddenLayerHeight];
	for (int i = 0; i < hiddenLayerNum; i++) {
		networkNodes[i + 1] = new node[hiddenLayerHeight];
		biasWeights[i] = new edge[hiddenLayerHeight];
		nodeWeights[i + 1] = new edge*[hiddenLayerHeight];
		for (int j = 0; j < hiddenLayerHeight; j++)
			nodeWeights[i + 1][j] = new edge[hiddenLayerHeight];
	}
	for (int i = 0; i < hiddenLayerHeight; i++)
		nodeWeights[hiddenLayerNum][i] = new edge[outputNum];
	biasWeights[hiddenLayerNum] = new edge[outputNum];
}

void BackpropNetwork::resetPointers() {
	for (int i = 0; i < CLASS_NUM; i++)
		for (int j = 0; j < TEST_SIZE; j++) testResult[i][j] = 0;
	for (int i = 0; i < inputNum; i++) {
		networkNodes[0][i].value = 0;
		networkNodes[0][i].delta = 0;
	}
	for (int i = 0; i < outputNum; i++) {
		error[i] = 0;
		targetOutput[i] = 0;
		networkNodes[hiddenLayerNum + 1][i].value = 0;
		networkNodes[hiddenLayerNum + 1][i].delta = 0;
		biasWeights[hiddenLayerNum][i].oldWeight = 0;
		biasWeights[hiddenLayerNum][i].weight = randomDouble(INITIAL_WEIGHT_MIN, INITIAL_WEIGHT_MAX);
	}
	for (int i = 0; i < hiddenLayerNum + 1; i++) {
		biasNodes[i].value = 1;
		biasNodes[i].delta = 0;
	}
	for (int i = 0; i < hiddenLayerNum; i++) {
		for (int j = 0; j < hiddenLayerHeight; j++) {
			networkNodes[i + 1][j].value = 0;
			networkNodes[i + 1][j].delta = 0;
			biasWeights[i][j].oldWeight = 0;
			biasWeights[i][j].weight = randomDouble(INITIAL_WEIGHT_MIN, INITIAL_WEIGHT_MAX);
		}
	}
	for (int i = 0; i < hiddenLayerNum + 1; i++) {
		int fromNodeNum;
		if (i == 0) fromNodeNum = inputNum;
		else fromNodeNum = hiddenLayerHeight;
		for (int j = 0; j < fromNodeNum; j++) {
			int toNodeNum = hiddenLayerHeight;
			if (i == hiddenLayerNum) toNodeNum = outputNum;
			for (int k = 0; k < toNodeNum; k++) {
				nodeWeights[i][j][k].oldWeight = 0;
				nodeWeights[i][j][k].weight = randomDouble(INITIAL_WEIGHT_MIN, INITIAL_WEIGHT_MAX);
			}
		}
	}
}

void BackpropNetwork::pointersToNull() {
	testResult = NULL;
	error = NULL;
	targetOutput = NULL;
	biasNodes = NULL;
	networkNodes = NULL;
	biasWeights = NULL;
	nodeWeights = NULL;
}

void BackpropNetwork::forwardPass() {
	for (int i = 0; i < hiddenLayerNum + 1; i++) {
		int toNodeNum = hiddenLayerHeight;
		if (i == hiddenLayerNum) toNodeNum = outputNum;
		for (int j = 0; j < toNodeNum; j++) {
			int fromNodeNum = hiddenLayerHeight;
			if (i == 0) fromNodeNum = inputNum;
			double net = 0;
			for (int k = 0; k < fromNodeNum; k++) 
				net += networkNodes[i][k].value * nodeWeights[i][k][j].weight;
			net += biasNodes[i].value * biasWeights[i][j].weight;
			networkNodes[i + 1][j].value = sigmoidFunction(net);
		}
	}
	for (int i = 0; i < outputNum; i++) 
		error[i] = networkNodes[hiddenLayerNum + 1][i].value - targetOutput[i];
}

void BackpropNetwork::backwardPass() {
	for (int i = hiddenLayerNum; i >= 0; i--) {
		int toNodeNum = hiddenLayerHeight;
		if (i == hiddenLayerNum) toNodeNum = outputNum;
		double learningRate = (LEARNING_BASE + (LEARNING_RATE * (i - hiddenLayerNum)));
		for (int j = 0; j < toNodeNum; j++) {
			double outNet = networkNodes[i + 1][j].value * (1 - networkNodes[i + 1][j].value);
			double totalOut = 0;
			if (i == hiddenLayerNum) totalOut = error[j];
			else {
				int nextLayerNodeNum = hiddenLayerHeight;
				if (i + 1 == hiddenLayerNum) nextLayerNodeNum = outputNum;
				for (int l = 0; l < nextLayerNodeNum; l++) 
					totalOut += networkNodes[i + 2][l].delta * nodeWeights[i + 1][j][l].oldWeight;
			}
			networkNodes[i + 1][j].delta = totalOut * outNet;
			int fromNodeNum = hiddenLayerHeight;
			if (i == 0) fromNodeNum = inputNum;
			for (int k = 0; k < fromNodeNum; k++) {
				nodeWeights[i][k][j].oldWeight = nodeWeights[i][k][j].weight;
				nodeWeights[i][k][j].weight -= learningRate * networkNodes[i + 1][j].delta * networkNodes[i][k].value;
			}
			biasWeights[i][j].weight -= learningRate * networkNodes[i + 1][j].delta;
		}
	}
}

void BackpropNetwork::setInput(double* inputVector) {
	for (int i = 0; i < inputNum; i++)
		networkNodes[0][i].value = inputVector[i];
}

void BackpropNetwork::setTarget(double* targetVector) {
	for (int i = 0; i < outputNum; i++)
		targetOutput[i] = targetVector[i];
}

int BackpropNetwork::getClass() {
	int maxClass = -1;
	double max = -100;
	for (int i = 0; i < outputNum; i++) {
		if (max < networkNodes[hiddenLayerNum + 1][i].value) {
			max = networkNodes[hiddenLayerNum + 1][i].value;
			maxClass = i + 1;
		}
	}
	return maxClass;
}

double BackpropNetwork::sigmoidFunction(double net) {
	return 1 / (1 + exp(net * (-1)));
}

double BackpropNetwork::randomDouble(double min, double max) {
	return (double(rand()) / float(RAND_MAX)) * (max - min) + min;
}

void BackpropNetwork::trian(double dataSet[][SAMPLE_SIZE][DIMENSION]) {
	int iteration = 0;
	bool finished = false;

	resetPointers();
	while (!finished && iteration <= MAX_ITERATION) {
		finished = true;
		iteration++;
		for (int i = 0; i < CLASS_NUM; i++) {
			double targetOutput[3] = { 0, 0, 0 };
			targetOutput[i] = 1;
			setTarget(targetOutput);
			for (int j = 0; j < TEST_SIZE; j++) {
				setInput(dataSet[i][j]);
				forwardPass();
				backwardPass();
				for (int i = 0; i < outputNum; i++) {
					if (finished && abs(error[i]) >= ERROR) {
						finished = false;
					}
				}
			}
		}
	}
	printf("Iteration spend: %d\n\n", iteration);
	trainIteration = iteration;
}

void BackpropNetwork::test(double dataSet[][SAMPLE_SIZE][DIMENSION]) {
	double correctCounter = 0;
	for (int i = 0; i < CLASS_NUM; i++) {
		for (int j = TEST_SIZE; j < SAMPLE_SIZE; j++) {
			setInput(dataSet[i][j]);
			forwardPass();
			testResult[i][j - TEST_SIZE] = getClass();
			if (testResult[i][j - TEST_SIZE] == i + 1) correctCounter++;
		}
	}
	accuracy = correctCounter / (TEST_SIZE * CLASS_NUM) * 100;
}

void BackpropNetwork::displayStats() {
	printf("Print network stats:\n");
	printf("inputNum: %d\n", inputNum);
	printf("outputNum: %d\n", outputNum);
	printf("trainIteration: %d\n", trainIteration);
	printf("hiddenLayerNum: %d\n", hiddenLayerNum);
	printf("hiddenLayerHeight: %d\n", hiddenLayerHeight);
	printf("================================\n\n");
}

void BackpropNetwork::displayAllStats() {
	displayStats();
	displayError();
	displayTargetOutput();
	displayBias();
	displayNetwork();
	displayBiasWeight();
	displayNodeWeight();
	displayTestReport();
}

void BackpropNetwork::displayError() {
	printf("Display error:\n\n");
	for (int i = 0; i < outputNum; i++)
		printf("%6.3f  ", error[i]);
	printf("\n========================\n\n");
}

void BackpropNetwork::displayTargetOutput() {
	printf("Display targetOutput:\n\n");
	for (int i = 0; i < outputNum; i++)
		printf("%6.3f  ", targetOutput[i]);
	printf("\n========================\n\n");
}

void BackpropNetwork::displayBias() {
	printf("Display biasNodes:\n\n");
	for (int i = 0; i < hiddenLayerNum + 1; i++)
		printf("(%6.3f, %6.3f)  ", biasNodes[i].value, biasNodes[i].delta);
	printf("\n========================\n\n");
}

void BackpropNetwork::displayNetwork() {
	printf("Display networkNodes:\n\n");
	for (int i = 0; i < hiddenLayerNum + 2; i++) {
		int height = hiddenLayerHeight;
		if (i == 0) height = inputNum;
		else if (i == hiddenLayerNum + 1) height = outputNum;
		for (int j = 0; j < height; j++) 
			printf("(%6.3f, %6.3f)  ", networkNodes[i][j].value, networkNodes[i][j].delta);
		printf("\n");
	}
	printf("\n========================\n\n");
}

void BackpropNetwork::displayBiasWeight() {
	printf("Display biasWeights:\n\n");
	for (int i = 0; i < hiddenLayerNum; i++) {
		for (int j = 0; j < hiddenLayerHeight; j++) 
			printf("(%6.3f, %6.3f)  ", biasWeights[i][j].weight, biasWeights[i][j].oldWeight);
		printf("\n");
	}
	for (int i = 0; i < outputNum; i++)
		printf("(%6.3f, %6.3f)  ", biasWeights[hiddenLayerNum][i].weight, biasWeights[hiddenLayerNum][i].oldWeight);
	printf("\n========================\n\n");
}

void BackpropNetwork::displayNodeWeight() {
	printf("Display nodeWeights:\n\n");
	for (int i = 0; i < hiddenLayerNum + 1; i++) {
		printf("Display weights in layer %d:\n", i);
		int fromNodeNum = hiddenLayerHeight;
		if (i == 0) fromNodeNum = inputNum;
		for (int j = 0; j < fromNodeNum; j++) {
			int toNodeNum = hiddenLayerHeight;
			if (i == hiddenLayerNum) toNodeNum = outputNum;
			for (int k = 0; k < toNodeNum; k++) 
				printf("(%7.3f, %7.3f)  ", nodeWeights[i][j][k].weight, nodeWeights[i][j][k].oldWeight);
			printf("\n");
		}
		printf("\n");
	}
	printf("\n========================\n\n");
};

void BackpropNetwork::displayAccuracy() {
	printf("Accuracy: %.1f %% \n\n", accuracy);
}

void BackpropNetwork::displayTestReport() {
	printf("\nDisplay test result:\n\n");
	for (int i = 0; i < CLASS_NUM; i++) {
		printf("Class %d:\n", i + 1);
		for (int j = 0; j < TEST_SIZE; j++) {
			printf("%d\n", testResult[i][j]);
		}
		printf("\n\n");
	}
	printf("Accuracy: %.1f %% \n\n", accuracy);
	printf("================================\n\n");
}