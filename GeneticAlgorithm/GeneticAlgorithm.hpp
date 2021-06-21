#pragma once
#include <iostream>
#define _USE_MATH_DEFINES
#include <math.h>
//#include "Renderer.hpp"
#include <time.h>

#include "NeuralNetwork/Neural_Network.hpp"

int popSize;
int dnaSize;
int outputSize;
float mutationRate;
float biasWeight;

NeuralNetwork* nn;
Matrix** ihWeights;
Matrix** hoWeights;
Matrix** hBiases;
Matrix** oBiases;

void selectAndClone(float* fitnesses);
void selectAndRecombinate(float* fitnesses);
void mutation(float iChance);
Matrix* tryModel(float* inputs, int inputSize, int index);
namespace GeneticAlgorithm {


	//init population
	void init(unsigned int iPopulationSize, float iMutationRate, unsigned int iOutputSize, unsigned int iInputSize, unsigned int iHiddenSize) {
		popSize = iPopulationSize;
		mutationRate = iMutationRate;
		outputSize = iOutputSize;
		//init generation randomly

		nn = new NeuralNetwork(iInputSize, iHiddenSize, iOutputSize);
		ihWeights = new Matrix* [popSize];
		hoWeights = new Matrix* [popSize];
		hBiases = new Matrix* [popSize];
		oBiases = new Matrix* [popSize];

		//randomize all weights
		for (int i = 0; i < popSize; i++) {
			ihWeights[i] = new Matrix(iHiddenSize, iInputSize);
			hoWeights[i] = new Matrix(iOutputSize, iHiddenSize);
			hBiases[i] = new Matrix(iHiddenSize, 1);
			oBiases[i] = new Matrix(iOutputSize, 1);

			MatrixMath::randomizeInInterval(ihWeights[i], 0.0f, 1.0f);
			MatrixMath::randomizeInInterval(hoWeights[i], 0.0f, 1.0f);
			MatrixMath::randomizeInInterval(hBiases[i], 0.0f, 1.0f);
			MatrixMath::randomizeInInterval(oBiases[i], 0.0f, 1.0f);
		}
	}

	/*
	//Biological Model (mutation, recombination, selection) with no supervision
	void randomBiologicalModelForGame(float* getCurrentInputs(), int inputSize, float iFitnessFunction(), bool* iIsFinished, void initGame(), void updateGame(float* iOutPuts, int outputSize), bool recombination) {
		//run test with every set of weights and calculate fitness for every member of the population
		float* fitnesses = new float[popSize];
		for (int i = 0; i < popSize; i++) {
			nn->setIHWeights(ihWeights[i]);
			nn->setHOWeights(hoWeights[i]);
			initGame();
			while (*iIsFinished == false) {
				Matrix* out = nn->feedForward(getCurrentInputs(), inputSize);
				float* currentOutputs = new float[outputSize];
				//give back output to game with update function so that it can apply output and go into nex
				for (int j = 0; j < out->rows; j++) {
					currentOutputs[j] = out->data[j][0];
				}
				updateGame(currentOutputs, outputSize);
				delete[] currentOutputs;
			}
			//game has finished, evaluate turn
			fitnesses[i] = iFitnessFunction();
		}
		if (recombination == true) {
			selectAndRecombinate(fitnesses);
		}
		else {
			selectAndClone(fitnesses);
		}
		mutation(mutationRate);
		delete[] fitnesses;
	}
	*/
	void randomBiologicalModel(float* currentInputs, int inputSize, float iFitnessFunction(float* outputs, int outputSize), bool recombination, float iBiasWeight) {
		biasWeight = iBiasWeight;
		//run test with every set of weights and calculate fitness for every member of the population
		float* fitnesses = new float[popSize];
		for (int i = 0; i < popSize; i++) {
			Matrix* out = tryModel(currentInputs, inputSize, i);
				
			float* currentOutputs = new float[outputSize];
			//give back output to game with update function so that it can apply output and go into nex
			for (int j = 0; j < out->rows; j++) {
				currentOutputs[j] = out->data[j][0];
			}
			//game has finished, evaluate turn
			fitnesses[i] = iFitnessFunction(currentOutputs, outputSize);
			delete[] currentOutputs;
		}
		if (recombination == true) {
			selectAndRecombinate(fitnesses);
		}
		else {
			selectAndClone(fitnesses);
		}
		mutation(mutationRate);
		delete[] fitnesses;
	}

	void test(float* inputs, int inputSize) {
		for (int i = 0; i < inputSize; i++) {
			std::cout << "inputs at " << i << ": " << inputs[i] << "\n";
		}
		for (int i = 0; i < popSize; i++) {
			Matrix* out = tryModel(inputs, inputSize, i);

			float* currentOutputs = new float[outputSize];
			//give back output to game with update function so that it can apply output and go into nex
			for (int j = 0; j < out->rows; j++) {
				std::cout << out->data[j][0] << "\n";
			}
		}
	}
}

Matrix* tryModel(float* inputs, int inputSize, int index) {
	nn->setIHWeights(ihWeights[index]);
	nn->setHOWeights(hoWeights[index]);
	Matrix* biasH = MatrixMath::multiplyWithNumber(hBiases[index], biasWeight, false);
	Matrix* biasO = MatrixMath::multiplyWithNumber(oBiases[index], biasWeight, false);
	nn->setBiasesH(biasH);
	nn->setBiasesO(biasO);
	Matrix* out = nn->feedForward(inputs, inputSize);

	delete hBiases[index];
	delete oBiases[index];
	hBiases[index] = nn->getHgradient();
	oBiases[index] = nn->getOgradient();
	delete biasH;
	delete biasO;
	return out;
}

void selectAndClone(float* fitnesses) {
	Matrix** newWeightsIH = new Matrix* [popSize];
	Matrix** newWeightsHO = new Matrix* [popSize];
	Matrix** newHbiases = new Matrix* [popSize];
	Matrix** newObiases = new Matrix* [popSize];

	float fitnessSum = 0.0f;
	//calculate fitness sum
	for (int i = 0; i < popSize; i++) {
		fitnessSum += fitnesses[i];
	}

	//for every space in the population, randomly (influenced by fitness) select parent of past generation and clone
	for (int i = 0; i < popSize; i++) {
		float runningSum = 0.0f;
		float randomNum = ((float(rand()) / float(RAND_MAX)) * (fitnessSum - 1.0f));//number between 0 and fitnessSum (-1 because of decimal conversion)
		for (int j = 0; j < popSize; j++) {
			float a = fitnesses[j];
			runningSum += fitnesses[j];
			if (runningSum >= randomNum) {
				newWeightsIH[i] = new Matrix(*ihWeights[j]); //copy matrix
				newWeightsHO[i] = new Matrix(*hoWeights[j]);
				newHbiases[i] = new Matrix(*hBiases[j]);
				newObiases[i] = new Matrix(*oBiases[j]);
				break;
			}
		}
	}
	for (int i = 0; i < popSize; i++) {
		delete ihWeights[i];
		delete hoWeights[i];
		delete hBiases[i];
		delete oBiases[i];
	}
	delete ihWeights;
	delete hoWeights;
	delete hBiases;
	delete oBiases;
	ihWeights = newWeightsIH;
	hoWeights = newWeightsHO;
	hBiases = newHbiases;
	oBiases = newObiases;
}

void selectAndRecombinate(float* fitnesses) {
	float fitnessSum = 0.0f;
	//calculate fitness sum
	for (int i = 0; i < popSize; i++) {
		fitnessSum += fitnesses[i];
	}

	//for every space in the population, randomly (influenced by fitness) select parent of past generation and recombinate
	for (int i = 0; i < popSize; i++) {
		//select parents for recombination
		int firstParent = -1;
		int secondParent = -1;
		for (int cParent = 0; cParent < 2; cParent++) {
			float runningSum = 0.0f;
			float randomNum = ((float(rand()) / float(RAND_MAX)) * (fitnessSum -1.0f));//number between 0 and fitnessSum
			for (int j = 0; j < popSize; j++) {
				runningSum += fitnesses[j];
				if (runningSum >= randomNum) {
					if (cParent == 0 && firstParent == -1) {
						firstParent = j;
					}
					if (cParent == 1 && secondParent == -1) {
						secondParent = j;
						break;
					}
				}
			}
		}

		//RECOMBINATION!!!---------------------------------------------------------------------

		for (int a = 0; a < ihWeights[0]->rows; a++) {
			for (int b = 0; b < ihWeights[0]->cols; b++) {
				ihWeights[i]->data[a][b] = (ihWeights[firstParent]->data[a][b] + ihWeights[secondParent]->data[a][b]) / 2;
			}
		}
		for (int a = 0; a < hoWeights[0]->rows; a++) {
			for (int b = 0; b < hoWeights[0]->cols; b++) {
				hoWeights[i]->data[a][b] = (hoWeights[firstParent]->data[a][b] + hoWeights[secondParent]->data[a][b]) / 2;
			}
		}
	}
}

void mutation(float iChance) {
	if (iChance < 0.0f || iChance > 1.0f) {
		std::cout << "Mutation rate is not between 0 and 1";
		std::exit(0);
	}
	for (int i = 0; i < popSize; i++) {
		float randomNum = (float(rand()) / float(RAND_MAX));//random number between 0 and 1
		for (int a = 0; a < ihWeights[0]->rows; a++) {
			for (int b = 0; b < ihWeights[0]->cols; b++) {
				if (randomNum < iChance) {
					//MatrixMath::randomizeInInterval(ihWeights[i], 0.0f, 1.0f);
					ihWeights[i]->data[a][b] += ((float(rand()) / float(RAND_MAX)) * 1.0f) - 0.5f;
					if (ihWeights[i]->data[a][b] > 1.0f) ihWeights[i]->data[a][b] = 1.0f;
					if (ihWeights[i]->data[a][b] < -1.0f) ihWeights[i]->data[a][b] = -1.0f;
				}
			}
		}
		randomNum = (float(rand()) / float(RAND_MAX));//random number between 0 and 1
		for (int a = 0; a < hoWeights[0]->rows; a++) {
			for (int b = 0; b < hoWeights[0]->cols; b++) {
				if (randomNum < iChance) {
					//MatrixMath::randomizeInInterval(hoWeights[i], 0.0f, 1.0f);
					hoWeights[i]->data[a][b] += ((float(rand()) / float(RAND_MAX)) * 1.0f) - 0.5f;
					if (hoWeights[i]->data[a][b] > 1.0f) hoWeights[i]->data[a][b] = 1.0f;
					if (hoWeights[i]->data[a][b] < -1.0f) hoWeights[i]->data[a][b] = -1.0f;
				}
			}
		}
	}
}