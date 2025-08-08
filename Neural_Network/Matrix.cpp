#include "Matrix.h"


Matrix::Matrix() {}
Matrix::Matrix(size_t row, size_t columns) : _rows(row), _cols(columns), dmatrix(row, dvector(columns, 0.0)) {}
Matrix::Matrix(std::initializer_list<std::initializer_list<double>> init) {
	*this = init;
}
Matrix::Matrix(dmatrix init) {
	this->clear();
	for (auto& row : init)
		this->emplace_back(row);

	_rows = this->size();
	_cols = (*this)[0].size();
}
Matrix::Matrix(dvector init) {
	this->clear();
	this->emplace_back(init);

	_rows = 1;
	_cols = init.size();
}
Matrix::Matrix(double init) {
	this->clear();
	
	this->emplace_back(dvector{ init });

	_rows = 1;
	_cols = 1;
}

void Matrix::operator=(std::initializer_list<std::initializer_list<double>> init) {
	this->clear();
	for (auto& row : init)
		this->emplace_back(row);

	_rows = this->size();
	_cols = (*this)[0].size();
}
Matrix Matrix::operator*(const Matrix& B) const {
	size_t n_columns = B.getCols();
	assert(B.getRows() == _cols);

	Matrix C(_rows, n_columns);
	for (size_t i = 0; i < _rows; i++)
		for (size_t j = 0; j < n_columns; j++)
			for (size_t k = 0; k < _cols; k++)
				C[i][j] += (*this)[i][k] * B[k][j];

	return C;
}
Matrix Matrix::hadamard(const Matrix& B) const {
	//assert(B.getRows() == _rows && B.getCols() == _cols);

	Matrix C(_rows, _cols);
	for (size_t i = 0; i < _rows; i++)
		for (size_t j = 0; j < _cols; j++)
			C[i][j] = (*this)[i][j] * B[i][j];

	return C;
}
Matrix Matrix::operator*(const double& a) {

	Matrix C(_rows, _cols);
	for (size_t i = 0; i < _rows; i++)
		for (size_t j = 0; j < _cols; j++)
			C[i][j] = (*this)[i][j] * a;

	return C;
}
Matrix Matrix::operator+(const Matrix& B) {
	//assert(B.getRows() == _rows && B.getCols() == _cols);

	Matrix C(_rows, _cols);
	for (size_t i = 0; i < _rows; i++)
		for (size_t j = 0; j < _cols; j++)
			C[i][j] = (*this)[i][j] + B[i][j];

	return C;
}
Matrix& Matrix::operator+=(const Matrix& B) {
	assert(B.getRows() == _rows && B.getCols() == _cols);

	for (size_t i = 0; i < _rows; i++)
		for (size_t j = 0; j < _cols; j++)
			(*this)[i][j] += B[i][j];

	return *this;
}
Matrix Matrix::operator-(const Matrix& B) {
	assert(B.getRows() == _rows && B.getCols() == _cols);

	Matrix C(_rows, _cols);
	for (size_t i = 0; i < _rows; i++)
		for (size_t j = 0; j < _cols; j++)
			C[i][j] = (*this)[i][j] - B[i][j];

	return C;
}

Matrix Matrix::T() {
	Matrix C(_cols, _rows);
	for (size_t i = 0; i < _cols; i++)
		for (size_t j = 0; j < _rows; j++)
			C[i][j] = (*this)[j][i];

	return C;
}
Matrix Matrix::addBias() {

	Matrix C(_rows, _cols + 1);
	for (size_t i = 0; i < _rows; i++) {
		for (size_t j = 0; j < _cols; j++)
			C[i][j] = (*this)[i][j];
		C[i][_cols] = 1;
	}

	return C;
}

// add biases and then transposes
Matrix Matrix::addBias_then_T() {

	Matrix C(_cols + 1, _rows);
	for (size_t j = 0; j < _rows; j++) {
		for (size_t i = 0; i < _cols; i++)
			C[i][j] = (*this)[j][i];
		C[_cols][j] = 1;
	}

	return C;
}

// transposes and then remove biases
Matrix Matrix::T_then_removeBias() {

	Matrix C(_cols, _rows - 1);
	for (size_t i = 0; i < _cols; i++)
		for (size_t j = 0; j < _rows - 1; j++)
			C[i][j] = (*this)[j][i];

	return C;
}
Matrix Matrix::removeBias() {

	Matrix C(_rows - 1, _cols);
	for (size_t i = 0; i < _rows - 1; i++)
		for (size_t j = 0; j < _cols; j++)
			C[i][j] = (*this)[i][j];

	return C;
}

Matrix Matrix::dropoutMask(const double& dropout) {
	double keep_prob = 1.0 - dropout;

	Matrix C(_rows, _cols);
	for (size_t i = 0; i < _rows; ++i)
		for (size_t j = 0; j < _cols; ++j)
			C[i][j] = ((double)rand() / RAND_MAX > keep_prob) ? 0.0 : ((*this)[i][j] / keep_prob);

	return C;
}
Matrix Matrix::derivReLU() {

	Matrix C(_rows, _cols);
	for (size_t i = 0; i < _rows; i++)
		for (size_t j = 0; j < _cols; j++)
			C[i][j] = ((*this)[i][j] > 0.0 ? 1.0 : 0);

	return C;
}
Matrix Matrix::derivTanH() {

	Matrix C(_rows, _cols);
	for (size_t i = 0; i < _rows; ++i) {
		for (size_t j = 0; j < _cols; ++j) {
			double v = std::tanh((*this)[i][j]);
			C[i][j] = 1 - v * v;
		}
	}

	return C;
}
Matrix Matrix::setMaxToOne() {

	Matrix C(_rows, _cols);
	for (size_t i = 0; i < _rows; i++) {
		auto maxElement = std::max_element((*this)[i].begin(), (*this)[i].end());
		size_t maxIndex = std::distance((*this)[i].begin(), maxElement);

		C[i][maxIndex] = 1;
	}

	return C;
}

Matrix Matrix::convolution(const Matrix& kernel, const int& padding, const int& stride) {
	assert(kernel.getRows() < _rows && kernel.getCols() < _cols);

	size_t nConv[2] = { size_t((_rows + 2 * padding - kernel.getRows()) / stride) + 1, size_t((_cols + 2 * padding - kernel.getCols()) / stride) + 1 };

	Matrix C(nConv[0], nConv[1]);
	for (size_t i = 0; i < nConv[0]; i++) {
		for (size_t j = 0; j < nConv[1]; j++) {
			double sum = 0;
			for (size_t kIdx_r = 0; kIdx_r < kernel.getRows(); kIdx_r++) {
				for (size_t kIdx_c = 0; kIdx_c < kernel.getCols(); kIdx_c++) {
					size_t inputRow = i * stride + kIdx_r - padding;
					size_t inputCol = j * stride + kIdx_c - padding;
					if (inputRow >= 0 && inputRow < _rows && inputCol >= 0 && inputCol < _cols) {
						sum += kernel[kIdx_r][kIdx_c] * (*this)[inputRow][inputCol];
					}
				}
			}
			C[i][j] = sum;
		}
	}

	return C;
}

Matrix Matrix::rotate180() {

	Matrix C(_rows, _cols);
	for (size_t i = 0; i < _rows; i++) {
		for (size_t j = 0; j < _cols; j++) {
			C[i][j] = (*this)[_rows - 1 - i][_cols - 1 - j];
		}
	}
	return C;
}

Matrix Matrix::dilate(int stride) {

	if (stride == 1) return (*this);

	size_t new_rows = (_rows - 1) * stride + 1;
	size_t new_cols = (_cols - 1) * stride + 1;

	Matrix dilated(new_rows, new_cols);

	for (size_t i = 0; i < _rows; i++) {
		for (size_t j = 0; j < _cols; j++) {
			dilated[i * stride][j * stride] = (*this)[i][j];
		}
	}

	return dilated;
}

double Matrix::norm() {
	double norm = 0;
	for (size_t i = 0; i < _rows; i++)
		for (size_t j = 0; j < _cols; j++)
			norm += (*this)[i][j];

	return norm / _rows / _cols;
}
double Matrix::l2norm() {
	double sum_sq = 0.0;
	for (size_t i = 0; i < _rows; i++)
		for (size_t j = 0; j < _cols; j++)
			sum_sq += (*this)[i][j] * (*this)[i][j];

	return std::sqrt(sum_sq);
}

size_t Matrix::getRows() const {
	return _rows;
}
size_t Matrix::getCols() const {
	return _cols;
}


Matrix& Matrix::clip(double min_val, double max_val) {
	for (size_t i = 0; i < _rows; ++i) {
		for (size_t j = 0; j < _cols; ++j) {
			if ((*this)[i][j] < min_val) (*this)[i][j] = min_val;
			else if ((*this)[i][j] > max_val) (*this)[i][j] = max_val;
		}
	}
	return *this;
}

dvector Matrix::getParams() {
	return dvector{ static_cast<double>(_rows), static_cast<double>(_cols) };
}