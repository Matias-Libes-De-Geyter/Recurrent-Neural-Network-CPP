#include <initializer_list>
#include <unordered_map>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cassert>
#include <random>
#include <string>
#include <vector>
#include <cmath>


#ifndef MATRIX_H
#define MATRIX_H

using d_vector = std::vector<double>;

class Matrix {
private:
	size_t _rows;
	size_t _cols;

	d_vector _matrix;

public:
	size_t rows() const { return _rows; };
	size_t cols() const { return _cols; };

	// Constructors
	inline Matrix() {};
	inline Matrix(size_t row, size_t columns) : _rows(row), _cols(columns), _matrix(d_vector(_rows* _cols, 0.0)) {};

	Matrix& operator=(std::initializer_list<std::initializer_list<double>>);
	Matrix operator*(const Matrix& B) const;
	Matrix operator*(double b) const;
	Matrix hadamard(const Matrix& B) const;

	Matrix operator+(const Matrix& B) const;
	Matrix& operator+=(const Matrix& B);
	Matrix operator-(const Matrix& B) const;

	Matrix T() const;
	Matrix addBias();
	Matrix addBias_then_T();
	Matrix removeBias();

	inline double& operator()(size_t i, size_t j) { return _matrix[i * _cols + j]; };
	inline const double& operator()(size_t i, size_t j) const { return _matrix[i * _cols + j]; };
};

#endif