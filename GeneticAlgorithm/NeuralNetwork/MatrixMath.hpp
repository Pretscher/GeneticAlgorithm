#pragma once
#include "Matrix.hpp"
#include "Utils.hpp"
namespace MatrixMath {
	//--------------------------------------------------------------
	//scalar functions

	void addNumber(Matrix* matrix, float n) {
		for (int i = 0; i < matrix->rows; i++) {
			for (int j = 0; j < matrix->cols; j++) {
				matrix->data[i][j] += n;
			}
		}
	}

	void multiplyWithNumber(Matrix* matrix, float n) {
		for (int i = 0; i < matrix->rows; i++) {
			for (int j = 0; j < matrix->cols; j++) {
				matrix->data[i][j] *= n;
			}
		}
	}

	void numberDividedByElements(Matrix* matrix, float n){
		for (int i = 0; i < matrix->rows; i++) {
			for (int j = 0; j < matrix->cols; j++) {
				float div = n / matrix->data[i][j];
				matrix->data[i][j] = div;
			}
		}
	}

	void limitToInterval(Matrix* matrix, float lowest, float heighest) {
		for (int i = 0; i < matrix->rows; i++) {
			for (int j = 0; j < matrix->cols; j++) {
				if (matrix->data[i][j] > heighest) {
					matrix->data[i][j] = heighest;
				}
				if (matrix->data[i][j] < lowest) {
					matrix->data[i][j] = lowest;
				}
			}
		}
	}
	//--------------------------------------------------------------
	//element-wise functions

	Matrix* addMatrices(Matrix* m1, Matrix* m2, bool reassignment) {

		//only works with matrices with same amount of rows/colss
		Matrix* result = new Matrix(m1->rows, m1->cols);
		if (m1->rows == m2->rows && m1->cols == m2->cols) {
			if (m1->rows == m2->rows && m1->cols == m2->cols) {
				for (int i = 0; i < result->rows; i++) {
					for (int j = 0; j < result->cols; j++) {
						result->data[i][j] = m1->data[i][j] + m2->data[i][j];
					}
				}
			}
		}
		else {
			std::cout << "Unequal rows/cols in add function";
			std::exit(0);
		}
		if (reassignment == true) {
			delete m1;
		}
		return result;
	}

	Matrix* substractMatrices(Matrix* m1, Matrix* m2, bool reassignment) {
		//only works with matrices with same amount of rows/colss
		Matrix* result = new Matrix(m1->rows, m1->cols);
		if (m1->rows == m2->rows && m1->cols == m2->cols) {
			for (int i = 0; i < result->rows; i++) {
				for (int j = 0; j < result->cols; j++) {
					result->data[i][j] = m1->data[i][j] - m2->data[i][j];
				}
			}
		}
		else {
			std::cout << "Unequal rows/cols in substract function";
			std::exit(0);
		}
		if (reassignment == true) {
			delete m1;
		}
		return result;
	}


	Matrix* dotProduct(Matrix* m1, Matrix* m2, bool reassignment) {
		Matrix* result = new Matrix(m1->rows, m2->cols);
		if (m1->cols == m2->rows) {
			//only works with matrices with same amount of rows/cols
			for (int i = 0; i < result->rows; i++) {
				for (int j = 0; j < result->cols; j++) {
					float sum = 0;
					for (int k = 0; k < m1->cols; k++) {
						sum += m1->data[i][k] * m2->data[k][j];
					}
					result->data[i][j] = sum;
				}
			}
		}
		else {
			std::cout << "Rows dont equal cols in dotproduct function";
			std::exit(0);
		}
		if (reassignment == true) {
			delete m1;
		}
		return result;
	}

	Matrix* elementWiseMult(Matrix* m1, Matrix* m2, bool reassignment) {
		Matrix* result = new Matrix(m1->rows, m1->cols);
		if (m1->rows == m2->rows && m1->cols == m2->cols) {
			for (int i = 0; i < m1->rows; i++) {
				for (int j = 0; j < m1->cols; j++) {
					result->data[i][j] = m1->data[i][j] * m2->data[i][j];
				}
			}
		}
		else {
			std::cout << "error in element wise multiplication function";
			std::exit(0);
		}
		if (reassignment == true) {
			delete m1;
		}
		return result;
	}

	//--------------------------------------------------------------
	//help functions


	void randomizeInInterval(Matrix* matrix, float lowest, float heighest) {
		for (int i = 0; i < matrix->rows; i++) {
			for (int j = 0; j < matrix->cols; j++) {
				float random = (float(rand()) / float(RAND_MAX)) * (heighest - lowest) + lowest;
				matrix->data[i][j] = random;
			}
		}
	}

	float sigmoidFunction(float input) {
		// 1/ (1 + e ^ -x)
		float result = 1 / (1 + exp(-input));
		return result;
	}

	/**@input only numbers that have been mapped with basic sigmoid function
	**/
	float dSigmoidAfterSigmoid(float input) {
		return input * (1 - input);
	}


	Matrix* mapWithDsigmoid(Matrix* m1, bool reassignment) {
		Matrix* results = new Matrix(m1->rows, m1->cols);
		for (int i = 0; i < results->rows; i++) {
			for (int j = 0; j < results->cols; j++) {
				results->data[i][j] = dSigmoidAfterSigmoid(m1->data[i][j]);
			}
		}
		if (reassignment == true) {
			delete m1;
		}
		return results;
	}

	Matrix* mapWithSigmoid(Matrix* m1, bool reassignment) {
		Matrix* results = new Matrix(m1->rows, m1->cols);
		for (int i = 0; i < results->rows; i++) {
			for (int j = 0; j < results->cols; j++) {
				results->data[i][j] = sigmoidFunction(m1->data[i][j]);
			}
		}
		if (reassignment == true) {
			delete m1;
		}
		return results;
	}

	Matrix* fromArray(float* arr, int size) {
		;
		Matrix* m = new Matrix(size, 1);
		for (int i = 0; i < size; i++) {
			m->data[i][0] = arr[i];
		}
		return m;
	}

	Matrix* transpose(Matrix* m1, bool reassignment) {
		Matrix* result = new Matrix(m1->cols, m1->rows);
		for (int i = 0; i < result->rows; i++) {
			for (int j = 0; j < result->cols; j++) {
				result->data[i][j] = m1->data[j][i];
			}
		}
		if (reassignment == true) {
			delete m1;
		}
		return result;
	}

	float* toArray(Matrix* m1) {
		float* outputArray = new float[m1->rows * m1->cols];
		for (int row = 0; row < m1->rows; row++) {
			for (int column = 0; column < m1->cols; column++) {
				outputArray[row + column] = m1->data[row][column];
			}
		}
		return outputArray;
	}
}