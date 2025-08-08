#include "Matrix.h"

Matrix& Matrix::operator=(std::initializer_list<std::initializer_list<double>> init) {
	_rows = init.size();
	_cols = init.begin()->size();

	_matrix.clear();
	_matrix.reserve(_rows * _cols);

	for (auto& row : init)
		_matrix.insert(_matrix.end(), row.begin(), row.end());

	return *this;
}

Matrix Matrix::operator*(const Matrix& B) const {
	assert(_cols == B.rows());

	size_t new_cols = B.cols();
	Matrix C(_rows, new_cols);

	for (int i = 0; i < _rows; i++) {
		for (int j = 0; j < new_cols; j++) {
			for (int k = 0; k < _cols; k++) {
				C(i, j) += _matrix[i * _cols + k] * B(k, j);
			}
		}
	}

	return C;
}


Matrix Matrix::operator*(double b) const {

	Matrix C(_rows, _cols);

	for (int i = 0; i < _rows; i++)
		for (int j = 0; j < _cols; j++)
			C(i, j) = _matrix[i * _cols + j] * b;

	return C;
}

Matrix Matrix::hadamard(const Matrix& B) const {
	assert(_rows == B.rows());
	assert(_cols == B.cols());

	Matrix C(_rows, _cols);
	for (size_t i = 0; i < _rows; i++)
		for (size_t j = 0; j < _cols; j++)
			C(i, j) = _matrix[i * _cols + j] * B(i, j);

	return C;
}


Matrix Matrix::operator+(const Matrix& B) const {
	assert(_rows == B.rows());
	assert(_cols == B.cols());

	Matrix C(_rows, _cols);
	for (size_t i = 0; i < _rows; i++)
		for (size_t j = 0; j < _cols; j++)
			C(i, j) = _matrix[i * _cols + j] + B(i, j);

	return C;
}

Matrix& Matrix::operator+=(const Matrix& B) {
	assert(_rows == B.rows());
	assert(_cols == B.cols());

	Matrix C(_rows, _cols);
	for (size_t i = 0; i < _rows; i++)
		for (size_t j = 0; j < _cols; j++)
			_matrix[i * _cols + j] += B(i, j);

	return *this;
}

Matrix Matrix::operator-(const Matrix& B) const {
	assert(_rows == B.rows());
	assert(_cols == B.cols());

	Matrix C(_rows, _cols);
	for (size_t i = 0; i < _rows; i++)
		for (size_t j = 0; j < _cols; j++)
			C(i, j) = _matrix[i * _cols + j] - B(i, j);

	return C;
}



Matrix Matrix::T() const {
	Matrix C(_cols, _rows);
	for (size_t i = 0; i < _cols; i++)
		for (size_t j = 0; j < _rows; j++)
			C(i, j) = _matrix[j * _cols + i];

	return C;
}
Matrix Matrix::addBias() {

	Matrix C(_rows, _cols + 1);
	for (size_t i = 0; i < _rows; i++) {
		for (size_t j = 0; j < _cols; j++)
			C(i, j) = _matrix[i * _cols + j];
		C(i, _cols) = 1;
	}

	return C;
}
Matrix Matrix::addBias_then_T() {

	Matrix C(_cols + 1, _rows);
	for (size_t j = 0; j < _rows; j++) {
		for (size_t i = 0; i < _cols; i++)
			C(i, j) = _matrix[j * _cols + i];
		C(_cols, j) = 1;
	}

	return C;
}
Matrix Matrix::removeBias() {

	Matrix C(_rows - 1, _cols);
	for (size_t i = 0; i < _rows - 1; i++)
		for (size_t j = 0; j < _cols; j++)
			C(i, j) = _matrix[i * _cols + j];

	return C;
}