#pragma once

namespace GeneticAlgorithm {
	void init(unsigned int iPopulationSize, float iMutationRate, unsigned int iOutputSize, unsigned int iInputSize, unsigned int iHiddenSize);
	void randomBiologicalModel(float* currentInputs, int inputSize, float iFitnessFunction(float* outputs, int outputSize), bool recombination);
	void test(float* inputs, int inputSize);
	float testAccuracy(float* inputs, int inputSize, float* targets);
}
