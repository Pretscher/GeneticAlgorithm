#include "GeneticAlgorithm.hpp"

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

	GeneticAlgorithm::init(100, 0.01f, 1, 2, 10);
	for (int i = 0; i < 10000; i++) {
		//float a = (float(rand()) / float(RAND_MAX)) * 0.5f;
		//float b = (float(rand()) / float(RAND_MAX)) * 0.5f;
		float a = 0.1f;
		float b = 0.3f;
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
	input[0] = 0.0f;
	input[1] = 0.2f;
	GeneticAlgorithm::test(input, 2);
	input[0] = 0.3f;
	input[1] = 0.4f;
	GeneticAlgorithm::test(input, 2);
	return 0;
}
