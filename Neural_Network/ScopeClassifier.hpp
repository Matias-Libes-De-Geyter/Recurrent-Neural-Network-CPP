#include "RNN.hpp"

#ifndef SCOPE_H
#define SCOPE_H

class ScopeClassifier {
private:
	const hyperparameters _hyper;

	std::vector<Matrix> M, V;

	int t;

public:
	ScopeClassifier(RNN&, const hyperparameters&);

	void Adam(Matrix& W, Matrix& dW, const int k);
	void SGD(Matrix& W, Matrix& dW);

	inline void step(RNN& model) {

		int k = 0;
		for (auto& [W, dW] : model.getParameters())
			Adam(*W, *dW, k++);

		t++;

	};

};

#endif