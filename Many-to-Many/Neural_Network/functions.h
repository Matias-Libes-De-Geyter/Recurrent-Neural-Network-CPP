#include "Matrix.h"

#ifndef FUNCTIONS_H
#define FUNCTIONS_H

// Hyperparameters
struct hyperparameters {
	int seq_len;
	int input_dimension;
	int hidden_dimension;
	int output_dimension;
	double learning_rate;
	int max_epochs;
	int n_batch;
	int batch_size;
	int test_size;
};

std::mt19937_64& get_rng();
double random(const double& min, const double& max); // Random function
int random_bit(); // Random bit between 0 and 1

double sequence_loss(const std::vector<Matrix>& y_pred, const std::vector<Matrix>& y_true); // Return the cross-entropy loss.

// Utility function used in TrainerClassifier.h
void writeFile(const d_vector& accuracies, const d_vector& trainLosses, const d_vector& testLosses, int nb_epochs, const std::string& filename);


// ===== PRINT FUNCTIONS =====
// Multiple print
template<typename... Args>
typename std::enable_if<(sizeof...(Args) > 1), void>::type
inline print(const Args&... args) { (std::cout << ... << args) << std::endl; }

// Single print
template<typename T>
typename std::enable_if<!std::is_same<T, Matrix>::value, void>::type
inline print(const T& arg) { std::cout << arg << std::endl; }

// Matrix print
inline void print(const Matrix& A) {
	const size_t rows = A.rows();
	const size_t cols = A.cols();

	std::ostringstream stream;
	stream << "[";
	for (size_t i = 0; i < rows; i++) {
		stream << "[";
		for (size_t j = 0; j < cols; j++) {
			stream << A(i, j);
			if (j < cols - 1) stream << ", ";
		}
		stream << "]";
		if (i < rows - 1) stream << ", " << std::endl;
	}
	stream << "]" << std::endl;

	std::cout << stream.str();
}

#endif