#pragma once
#include <iostream>
#define _USE_MATH_DEFINES
#include <math.h>

#include "GeneticAlgorithm.hpp"
#include "NeuralNetwork/Neural_Network.hpp"//for this we added $(ProjectDir) to additional includes for the file.

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
Matrix* tryModel(float* inputs, int inputSize, int index);

	//init population
	void GeneticAlgorithm::init(unsigned int iPopulationSize, float iMutationRate, unsigned int iOutputSize, unsigned int iInputSize, unsigned int iHiddenSize) {
		popSize = iPopulationSize;
		mutationRate = iMutationRate;
		outputSize = iOutputSize;
		//init generation randomly

		nn = new NeuralNetwork(iInputSize, iHiddenSize, iOutputSize);
		ihWeights = new Matrix * [popSize];
		hoWeights = new Matrix * [popSize];

		//randomize all weights
		for (int i = 0; i < popSize; i++) {
			ihWeights[i] = new Matrix(iHiddenSize, iInputSize);
			hoWeights[i] = new Matrix(iOutputSize, iHiddenSize);

			MatrixMath::randomizeInInterval(ihWeights[i], -1.0f, 1.0f);
			MatrixMath::randomizeInInterval(hoWeights[i], -1.0f, 1.0f);
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
	void GeneticAlgorithm::randomBiologicalModel(float* currentInputs, int inputSize, float iFitnessFunction(float* outputs, int outputSize), bool recombination) {
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

	void GeneticAlgorithm::test(float* inputs, int inputSize) {
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

	float GeneticAlgorithm::testAccuracy(float* inputs, int inputSize, float* targets) {
		float accuracy = 0.0f;
		for (int i = 0; i < popSize; i++) {
			Matrix* out = tryModel(inputs, inputSize, i);

			float* currentOutputs = new float[outputSize];
			//give back output to game with update function so that it can apply output and go into nex
			for (int j = 0; j < out->rows; j++) {

				float data = out->data[j][0];
				float target = targets[j];

				if ((target <= 0.0 && data >= 0.0) || (target >= 0.0 && data <= 0.0)) {
					target += 1.0;
					data += 1.0;
				}
				if ((target > data && target > 0 && data > 0) || (target < data && target < 0 && data < 0)) {
					accuracy += (data / target) / popSize;
				}
				else {
					accuracy += (target / data) / popSize;
				}
			}
			delete[] currentOutputs;
		}
		std::cout << "Accuracy: " << accuracy << "\n";
		return accuracy;
	}


Matrix* tryModel(float* inputs, int inputSize, int index) {
	nn->setIHWeights(ihWeights[index]);
	nn->setHOWeights(hoWeights[index]);

	return nn->feedForward(inputs, inputSize);;
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
		float runningSum = 0.0f;
		float randomNum = ((float(rand()) / float(RAND_MAX)) * (fitnessSum - 1.0f));//number between 0 and fitnessSum (-1 because of decimal conversion)
		for (int j = 0; j < popSize; j++) {
			float a = fitnesses[j];
			runningSum += fitnesses[j];
			if (runningSum >= randomNum) {
				newWeightsIH[i] = new Matrix(*ihWeights[j]); //copy matrix
				newWeightsHO[i] = new Matrix(*hoWeights[j]);
				break;
			}
		}
	}
	for (int i = 0; i < popSize; i++) {
		delete ihWeights[i];
		delete hoWeights[i];
	}
	delete ihWeights;
	delete hoWeights;
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
		int firstParent = -1;
		int secondParent = -1;
		for (int cParent = 0; cParent < 2; cParent++) {
			float runningSum = 0.0f;
			float randomNum = ((float(rand()) / float(RAND_MAX)) * (fitnessSum - 1.0f));//number between 0 and fitnessSum
			for (int j = 0; j < popSize; j++) {
				runningSum += fitnesses[j];
				if (runningSum >= randomNum) {
					if (cParent == 0 && firstParent == -1) {
						firstParent = j;
						break;
					}
					if (cParent == 1 && secondParent == -1) {
						secondParent = j;
						break;
					}
				}
			}
		}

		//RECOMBINATION!!!---------------------------------------------------------------------
		newWeightsIH[i] = new Matrix(ihWeights[0]->rows, ihWeights[0]->cols);
		newWeightsHO[i] = new Matrix(hoWeights[0]->rows, hoWeights[0]->cols);

		for (int a = 0; a < ihWeights[0]->rows; a++) {
			for (int b = 0; b < ihWeights[0]->cols; b++) {
				newWeightsIH[i]->data[a][b] = (ihWeights[firstParent]->data[a][b] + ihWeights[secondParent]->data[a][b]) / 2;
			}
		}
		for (int a = 0; a < hoWeights[0]->rows; a++) {
			for (int b = 0; b < hoWeights[0]->cols; b++) {
				newWeightsHO[i]->data[a][b] = (hoWeights[firstParent]->data[a][b] + hoWeights[secondParent]->data[a][b]) / 2;
			}
		}
	}

	for (int i = 0; i < popSize; i++) {
		delete ihWeights[i];
		delete hoWeights[i];
	}
	delete ihWeights;
	delete hoWeights;
	ihWeights = newWeightsIH;
	hoWeights = newWeightsHO;
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
					ihWeights[i]->data[a][b] += ((float(rand()) / float(RAND_MAX)) * 2.0f) - 1.0f;
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
					hoWeights[i]->data[a][b] += ((float(rand()) / float(RAND_MAX)) * 2.0f) - 1.0f;
					if (hoWeights[i]->data[a][b] > 1.0f) hoWeights[i]->data[a][b] = 1.0f;
					if (hoWeights[i]->data[a][b] < -1.0f) hoWeights[i]->data[a][b] = -1.0f;
				}
			}
		}
	}
}