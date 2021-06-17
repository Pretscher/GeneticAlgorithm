#include "GeneticAlgorithm.hpp"

float* currentTarget;

float fitnessFunction(float* cOutputs, int outputSize) {
	float fitness = 0;
	for (int i = 0; i < outputSize; i++) {
		float target = abs(currentTarget[i]);
		float output = abs(cOutputs[i]);
		float potentialFitness = target - output;
		if (potentialFitness > 0) {
			fitness += potentialFitness;
		}
		else {
			fitness -= potentialFitness;
		}
	}
	fitness = 10000 / (fitness * 1000);//should be bigger the smaller the distance between values
	return fitness;
}

int main() {
	srand(time(NULL));//Set random seed with current time so its somewhere random

	GeneticAlgorithm::init(100, 0.01f, 1, 2, 20);
	for (int i = 0; i < 1000; i++) {
		float a = (float(rand()) / float(RAND_MAX)) * 0.5f;
		float b = (float(rand()) / float(RAND_MAX)) * 0.5f;
		float* input = new float[2];
		input[0] = a;
		input[1] = b;
		currentTarget = new float[1];
		currentTarget[0] = a + b;

		GeneticAlgorithm::randomBiologicalModel(input, 2, &fitnessFunction, false);
	}
	float a = (float(rand()) / float(RAND_MAX)) * 0.5f;
	float b = (float(rand()) / float(RAND_MAX)) * 0.5f;
	float* input = new float[2];
	input[0] = a;
	input[1] = b;
	GeneticAlgorithm::test(input, 2);

	return 0;
}
