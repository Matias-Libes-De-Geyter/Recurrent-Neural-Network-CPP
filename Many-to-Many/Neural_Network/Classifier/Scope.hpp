#include "..\RNN/RNN.hpp"

#ifndef SCOPE_HPP
#define SCOPE_HPP

class Scope {
private:
	const hyperparameters& _hyper;

	// M, V matrices for Adam optimizer
	std::vector<Matrix> M, V;

	// Time index for Adam optimizer
	int t;

public:
	// Constructor method
	Scope(RNN&, const hyperparameters&);

	// Optimizers
	void Adam(Matrix& W, Matrix& dW, const int k);
	void SGD(Matrix& W, Matrix& dW);

	// 'Update parameters' method using the chosen optimizer.
	inline void step(RNN& model) {
		int k = 0;
		for (auto& [W, dW] : model.getParameters())
			Adam(*W, *dW, k++);

		t++;
	};

};

#endif