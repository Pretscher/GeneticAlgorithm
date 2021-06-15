#pragma once
#include <iostream>
class Matrix {
public:
	int rows;
	int cols;
	float** data;
	
	~Matrix() {
		for (int i = 0; i < rows; i++) {
			delete data[i];
		}
		delete data;
	}

	Matrix(int rows, int cols) {
		this->rows = rows;
		this->cols = cols;
		this->data = new float* [rows];
		for (int i = 0; i < rows; i++) {
			this->data[i] = new float[cols];
		}
	}
	//constructor from array
	Matrix(float* arr, int size) {
		this->rows = size;
		this->cols = 1;
		this->data = new float*[rows];
		for (int i = 0; i < rows; i++) {
			this->data[i] = new float[cols];
		}

		for (int i = 0; i < rows; i++) {
			this->data[i][0] = arr[i];
		}
	}

	void print() {
		for (int r = 0; r < this->rows; r++) {
			for (int c = 0; c < this->cols; c++) {
				std::cout << "data at r = " << r << " and c = " <<
					c << " : " << this->data[r][c];
			}
		}
	}
};