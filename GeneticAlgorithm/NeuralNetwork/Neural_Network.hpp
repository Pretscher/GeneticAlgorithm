#pragma once
#include "MatrixMath.hpp"
class NeuralNetwork {
private:
	Matrix* inputNodes;
	Matrix* hiddenNodes;
	Matrix* outputNodes;

	Matrix* weightsHI;
	Matrix* weightsOH;

	Matrix* biasesH;
	Matrix* biasesO;

public:
	/** 
	A basic toy neural network with one hidden layer.
	Example code bit:
    NeuralNetwork* nn = new NeuralNetwork(2, 2, 1);
    float* input = new float[2];
    input[0] = 0.2f;
    input[1] = 1.0f;
    float* targets = new float[1];
    targets[0] = 1.0f;
    nn->train(input, 2, targets, 1, 0.05f, 100000);
    nn->feedForward(input, 2)->print();
	**/

	int inputNodeCount;
	int hiddenNodeCount;
	int outputNodeCount;
	NeuralNetwork(int inputNodeCount, int hiddenNodeCount, int outputNodeCount) {
		this->inputNodeCount = inputNodeCount;
		this->hiddenNodeCount = hiddenNodeCount;
		this->outputNodeCount = outputNodeCount;

		inputNodes = nullptr;
		hiddenNodes = nullptr;
		outputNodes = nullptr;

		//inputHidden Weight-Matrix: rows = hidden, cols = input
		weightsHI = new Matrix(hiddenNodeCount, inputNodeCount);
		//hiddenOutput Weight-Matrix: rows = output, cols = hidden
		weightsOH = new Matrix(outputNodeCount, hiddenNodeCount);
		//randomize between -1 and 1
		MatrixMath::randomizeInInterval(weightsHI, 0.0f, 1.0f);
		MatrixMath::randomizeInInterval(weightsOH, 0.0f, 1.0f);

		biasesH = new Matrix(hiddenNodeCount, 1);//biases to hidden nodes
		MatrixMath::randomizeInInterval(biasesH, 0.0f, 1.0f);
		biasesO = new Matrix(outputNodeCount, 1);//biases to output nodes
		MatrixMath::randomizeInInterval(biasesO, 0.0f, 1.0f);
	}
	bool firstIteration = true;
	Matrix* feedForward(float* imputArray, int inputSize) {
		if (firstIteration == false) {
			delete hiddenNodes;
			delete inputNodes;
			delete outputNodes;
		}
		else {
			firstIteration = false;
		}
		inputNodes = new Matrix(imputArray, inputSize);
		//multiply imputnodes with weights between inputs and hiddens to get hiddenNodes
		hiddenNodes = MatrixMath::dotProduct(weightsHI, inputNodes, false);
		//add hiddenBiases to hidden
		//hiddenNodes = MatrixMath::addMatrices(hiddenNodes, biasesH, true);
		hiddenNodes = MatrixMath::mapWithSigmoid(hiddenNodes, true);

		outputNodes = MatrixMath::dotProduct(weightsOH, hiddenNodes, false);
		//outputNodes = MatrixMath::addMatrices(outputNodes, biasesO, true);
		outputNodes = MatrixMath::mapWithSigmoid(outputNodes, true);
		return outputNodes;
	}

	void trainBackpropagation(float* inputs, int inputSize, float* traindata, int targetSize, float learningRate, int iterations) {
		Matrix* targets = MatrixMath::fromArray(traindata, targetSize);
		for (int i = 0; i < iterations; i++) {
			this->feedForward(inputs, inputSize);
			backpropagation(targets, learningRate);
		}
		delete targets;
	}


	Matrix* getIHWeights() {
		return this->weightsHI;
	}

	Matrix* getHOWeights() {
		return this->weightsOH;
	}

	void setIHWeights(Matrix* newWeights) {
		this->weightsHI = newWeights;
	}

	void setHOWeights(Matrix* newWeights) {
		this->weightsOH = newWeights;
	}

private:
	void backpropagation(Matrix* targets, float learningRate) {
		//output errors

		Matrix* oErrors = MatrixMath::substractMatrices(targets, outputNodes, false);

		Matrix* oGradients = MatrixMath::mapWithDsigmoid(outputNodes, false);
		oGradients = MatrixMath::elementWiseMult(oGradients, oErrors, true);
		MatrixMath::multiplyWithNumber(oGradients, learningRate);


		Matrix* hiddenTrans = MatrixMath::transpose(hiddenNodes, false);
		Matrix* ohDeltas = MatrixMath::dotProduct(oGradients, hiddenTrans, false);
		weightsOH = MatrixMath::addMatrices(weightsOH, ohDeltas, true);
		biasesO = MatrixMath::addMatrices(biasesO, oGradients, true);

		//hidden errors

		Matrix* wOHTrans = MatrixMath::transpose(weightsOH, false);
		Matrix* hErrors = MatrixMath::dotProduct(wOHTrans, oErrors, false);

		Matrix* hGradients = MatrixMath::mapWithDsigmoid(hiddenNodes, false);
		hGradients = MatrixMath::elementWiseMult(hGradients, hErrors, true);
		MatrixMath::multiplyWithNumber(hGradients, learningRate);

		Matrix* inputTrans = MatrixMath::transpose(inputNodes, false);
		Matrix* hiDeltas = MatrixMath::dotProduct(hGradients, inputTrans, false);
		weightsHI = MatrixMath::addMatrices(weightsHI, hiDeltas, true);
		biasesH = MatrixMath::addMatrices(biasesH, hGradients, true);


		delete oGradients;
		delete oErrors;
		delete hiddenTrans;
		delete ohDeltas;
		delete wOHTrans;
		delete hErrors;
		delete hGradients;
		delete inputTrans;
		delete hiDeltas;
	}
};