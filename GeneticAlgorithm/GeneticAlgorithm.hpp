#pragma once
#include <iostream>
#define _USE_MATH_DEFINES
#include <math.h>
//#include "Renderer.hpp"
#include <time.h>

#include "NeuralNetwork/Neural_Network.hpp"

float** generation;
int popSize;
int dnaSize;
float mutationRate;

NeuralNetwork* nn;

namespace GeneticAlgorithm {


	//init population
	void init(unsigned int iPopulationSize, unsigned int iDnaSize, float iMutationRate, int iOutputSize) {
		popSize = iPopulationSize;
		dnaSize = iDnaSize;
		mutationRate = iMutationRate;

		//init generation randomly
		srand(time(NULL));//Set random seed with current time so its somewhere random
		for (int i = 0; i < popSize; i++) {
			generation[i] = new float[dnaSize];
			//randomize dna
			for (int j = 0; j < dnaSize; j++) {
				generation[i][j] = float(rand()) / float(RAND_MAX);//random num between 0.0 and 1.0
			}
		}

		nn = new NeuralNetwork(popSize, dnaSize, iOutputSize);

	}
	
	//Biological Model (mutation, recombination, selection) with no supervision
	void randomBiologicalModel(float iFitnessFunction()) {

	}
}