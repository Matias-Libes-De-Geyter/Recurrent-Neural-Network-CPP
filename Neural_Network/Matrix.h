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

using dmatrix = std::vector<std::vector<double>>;
using dvector = std::vector<double>;


class Matrix : public std::vector<std::vector<double>> {
private:
	size_t _rows;
	size_t _cols;
public:
	Matrix();
	Matrix(size_t row, size_t columns);
	Matrix(dmatrix);
	Matrix(dvector);
	Matrix(double);
	Matrix(std::initializer_list<std::initializer_list<double>>);
	void operator=(std::initializer_list<std::initializer_list<double>>);
	Matrix operator*(const Matrix&) const;
	Matrix hadamard(const Matrix&) const;
	Matrix operator*(const double&);
	Matrix operator+(const Matrix&);
	Matrix& operator+=(const Matrix& B);
	Matrix operator-(const Matrix&);
	Matrix T();
	Matrix addBias();
	Matrix addBias_then_T();
	Matrix T_then_removeBias();
	Matrix removeBias();
	Matrix derivReLU();
	Matrix derivTanH();
	Matrix setMaxToOne();
	Matrix dropoutMask(const double&);
	Matrix convolution(const Matrix&, const int&, const int&);
	Matrix rotate180();
	Matrix dilate(int stride);
	double norm();
	double l2norm();
	size_t getRows() const;
	size_t getCols() const;
	Matrix& clip(double min_val, double max_val);
	dvector getParams();
};

#endif