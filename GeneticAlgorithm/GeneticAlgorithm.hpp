#pragma once
#include "Utils.hpp"

namespace GeneticAlgorithm {
	float** generation;
	//init population
	void init(unsigned int iPopulationSize, unsigned int iModifiableNodesPerUnit, float iMutationRate) {
		for (int i = 0; i < iPopulationSize; i++) {
			generation[i] = new float[iModifiableNodesPerUnit];
			//randomize
			for (int j = 0; j < iModifiableNodesPerUnit; j++) {
				generation[i][j] = 0;
			}
		}
	}
	
	//Biological Model (mutation, recombination, selection)
	void biologicalModel(void* iFitnessFunction()) {

	}
}