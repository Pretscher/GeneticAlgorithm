#include "GeneticAlgorithm/GeneticAlgorithm.hpp"
#include <iostream>
float* currentTarget;

float fitnessFunction(float* cOutputs, int outputSize) {
	float fitness = 0;
	for (int i = 0; i < outputSize; i++) {
		float target = currentTarget[i];
		float output = cOutputs[i];
		if (target > output) {
			fitness += target - output;
		}
		else {
			fitness += output - target;
		}
	}

	float out = 1 / fitness;

	return out;
}

int main() {
	srand(time(NULL));//Set random seed with current time so its somewhere random

//PREFERENCES-----------------------------------------------------------------------------------------------

	int iterations = 1000;
	int testingIterations = 25;
	
	int popSize = 100;
	float mutationRate = 0.02f;
	int hiddenNodes = 30;

	bool doRecombination = true;

//\PREFERENCES----------------------------------------------------------------------------------------------
	
	GeneticAlgorithm::init(popSize, mutationRate, 1, 2, hiddenNodes);// first int: targetSize, second int inputSize
	//print progress
	int progBars = 0;

	float avgAccuracy = 0.0f;
	for (int i = 0; i < iterations; i++) {
		float a = ((float(rand()) / float(RAND_MAX)) * 1.0f) - 0.5f;
		float b = ((float(rand()) / float(RAND_MAX)) * 1.0f) - 0.5f;

		float* input = new float[2];
		input[0] = a;
		input[1] = b;

		currentTarget = new float[1];
		currentTarget[0] = a + b;

		if (i < iterations - testingIterations) {
			GeneticAlgorithm::randomBiologicalModel(input, 2, &fitnessFunction, doRecombination);
		}
		else {
			avgAccuracy += GeneticAlgorithm::testAccuracy(input, 2, currentTarget) / testingIterations;
		}
		delete[] input;
		delete[] currentTarget;

		if (i % ((iterations + testingIterations) / 20) == 0) {
			std::cout << "Progress: [";
			for (int i = 0; i < 19; i++) {
				if (i < progBars) {
					std::cout << "=";
				}
				else {
					std::cout << " ";
				}
			}
			std::cout << "]\n";
			progBars++;
		}

	}
	std::cout << "\n-------------Average Accuracy: " << avgAccuracy << "--------------\n";
/*	float* input = new float[2];
	input[0] = -0.2f;
	input[1] = -0.3f;
	GeneticAlgorithm::test(input, 2);
	input[0] = 0.3f;
	input[1] = 0.4f;
	GeneticAlgorithm::test(input, 2);*/
	return 0;
}
