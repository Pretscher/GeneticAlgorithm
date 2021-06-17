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

NeuralNetwork* nn;
Matrix** ihWeights;
Matrix** hoWeights;


void selectAndClone(float* fitnesses);
void selectAndRecombinate(float* fitnesses);
void mutation(float iChance);
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
		//randomize all weights
		for (int i = 0; i < popSize; i++) {
			ihWeights[i] = new Matrix(iHiddenSize, iInputSize);
			hoWeights[i] = new Matrix(iOutputSize, iHiddenSize);
			MatrixMath::randomizeInInterval(ihWeights[i], 0.0f, 1.0f);
			MatrixMath::randomizeInInterval(hoWeights[i], 0.0f, 1.0f);
		}
	}


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

	void randomBiologicalModel(float* currentInputs, int inputSize, float iFitnessFunction(float* outputs, int outputSize), bool recombination) {
		//run test with every set of weights and calculate fitness for every member of the population
		float* fitnesses = new float[popSize];
		for (int i = 0; i < popSize; i++) {
			nn->setIHWeights(ihWeights[i]);
			nn->setHOWeights(hoWeights[i]);
			Matrix* out = nn->feedForward(currentInputs, inputSize);
				
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
			std::cout << "inputs at " << i << ": " << inputs[i];
		}
		for (int i = 0; i < popSize; i++) {
			nn->setIHWeights(ihWeights[i]);
			nn->setHOWeights(hoWeights[i]);
			Matrix* out = nn->feedForward(inputs, inputSize);

			float* currentOutputs = new float[outputSize];
			//give back output to game with update function so that it can apply output and go into nex
			for (int j = 0; j < out->rows; j++) {
				std::cout << out->data[j][0] << "\n";
			}
		}
	}
}

void selectAndClone(float* fitnesses) {
	Matrix** newWeightsIH = new Matrix * [popSize];
	Matrix** newWeightsHO = new Matrix * [popSize];
	float fitnessSum = 0.0f;
	//calculate fitness sum
	for (int i = 0; i < popSize; i++) {
		fitnessSum += fitnesses[i];
	}

	//for every space in the population, randomly (influenced by fitness) select parent of past generation and clone
	for (int i = 0; i < popSize; i++) {
		bool reached = false;
		float runningSum = 0.0f;
		float randomNum = ((float(rand()) / float(RAND_MAX)) * fitnessSum) - 1.0f;//number between 0 and fitnessSum
		for (int j = 0; j < popSize; j++) {
			runningSum += fitnesses[j];
			if (runningSum >= randomNum) {
				newWeightsIH[i] = ihWeights[j];
				newWeightsHO[i] = hoWeights[j];
				reached = true;
				break;
			}
		}
		if (reached == false) {
			std::cout << "Error in selectAndClone: runningsum couldnt reach randomnum";
		}
	}
	ihWeights = newWeightsIH;
	hoWeights = newWeightsHO;
}

void selectAndRecombinate(float* fitnesses) {
	Matrix** newWeightsIH = new Matrix * [popSize];
	Matrix** newWeightsHO = new Matrix * [popSize];
	float fitnessSum = 0.0f;
	//calculate fitness sum
	for (int i = 0; i < popSize; i++) {
		fitnessSum += fitnesses[i];
	}

	//for every space in the population, randomly (influenced by fitness) select parent of past generation and recombinate
	for (int i = 0; i < popSize; i++) {
		//select parents for recombination
		int firstParent;
		int secondParent;
		for (int cParent = 0; cParent < 2; cParent++) {
			float runningSum = 0.0f;
			float randomNum = (float(rand()) / float(RAND_MAX)) * fitnessSum;//number between 0 and fitnessSum
			for (int j = 0; j < popSize; j++) {
				runningSum += fitnesses[j];
				if (runningSum > randomNum) {
					if (i == 0) {
						firstParent = j;
					}
					if (i == 1) {
						secondParent = j;
					}
					break;
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
}
void mutation(float iChance) {
	if (iChance < 0.0f || iChance > 1.0f) {
		std::cout << "Mutation rate is not between 0 and 1";
		std::exit(0);
	}

	for (int i = 0; i < popSize; i++) {
		for (int a = 0; a < ihWeights[i]->rows; a++) {
			for (int b = 0; b < ihWeights[i]->cols; b++) {
				float randomNum = (float(rand()) / float(RAND_MAX));//random number between 0 and 1
				if (randomNum < iChance) {
					ihWeights[i]->data[a][b] = ((float(rand()) / float(RAND_MAX)) * 2.0f) - 1.0f;
				}
			}
		}
		for (int a = 0; a < hoWeights[i]->rows; a++) {
			for (int b = 0; b < hoWeights[i]->cols; b++) {
				float randomNum = (float(rand()) / float(RAND_MAX));//random number between 0 and 1
				if (randomNum < iChance) {
					hoWeights[i]->data[a][b] = ((float(rand()) / float(RAND_MAX)) * 2.0f) - 1.0f;
				}
			}
		}
	}
}