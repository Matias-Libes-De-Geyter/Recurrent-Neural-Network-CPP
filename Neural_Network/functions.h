#include "Matrix.h"

#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#define EULERS_NUMBER pow((1.0 + 1.0 / 10000000.0), 10000000.0)

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
};

std::mt19937_64& get_rng();
double random(const double& min, const double& max); // Random function
int random_bit(); // Random bit between 0 and 1

Matrix hotOne(const dvector& y, const int& nElements); // Returns the "hot one" matrix of a vector.
double CELossFunction(const Matrix& y_pred, const Matrix& y_true); // Return the cross-entropy loss.

// Flatten and unflatten functions
dvector flatten(const std::vector<Matrix>& A);
Matrix unFlatten(const dvector& A, const int& iFtMap, const int& rows, const int& cols);

// Utility function used in TrainerClassifier.h
Matrix flattenToMatrix(const std::vector<double>& flat_image, int rows, int cols);
void readMNIST(const std::string& imageFile, const std::string& labelFile, dmatrix& images, dvector& labels);
void writeFile(const dvector& accuracies, const dvector& trainLosses, const dvector& testLosses, int nb_epochs, const std::string& filename);


// ===== PRINT FUNCTIONS ===== //

// Print other stuff
template<typename... Args> inline void print(const Args&... args) { (std::cout << ... << args) << std::endl; }

// Print matrices & vectors
template<typename T>
inline void print(const T& container) {
	if constexpr (std::is_same_v<T, Matrix> || std::is_same_v<T, dvector>) {
		std::cout << "[";
		bool first = true;
		for (auto element : container) {
			if constexpr (!std::is_same_v<T, Matrix>) std::cout << (!first ? ", " : "") << element;
			else print(element);
			first = false;
		}
		std::cout << "]," << std::endl;
	}
	else print(container, "");
}

#endif