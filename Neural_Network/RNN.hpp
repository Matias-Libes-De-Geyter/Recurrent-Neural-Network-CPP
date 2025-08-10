#include "functions.h"

#ifndef RNN_HPP
#define RNN_HPP

class RNN {

private:
	const hyperparameters& _hyper;
	Matrix m_inWeights;
	Matrix m_hiddenWeights;
	Matrix m_outWeights;

	Matrix m_dU;
	Matrix m_dW;
	Matrix m_dWout;
	
	std::vector<Matrix> m_Z;
	std::vector<Matrix> m_hiddenStates;
	std::vector<Matrix> m_deltas;

	Matrix m_output;

public:
	RNN(const hyperparameters& hyper);

	void forward(const std::vector<Matrix>& input);

	void backpropagation(const std::vector<Matrix>& input, const Matrix& y_real);

	Matrix activate(Matrix& inputs);
	Matrix deriv_activate(Matrix& inputs);
	Matrix softmax_activation(Matrix& inputs);

	inline const Matrix& getOutput() const {
		return m_output;
	};
	inline std::vector<std::pair<Matrix*, Matrix*>> getParameters() {
		return {
		{ &m_inWeights, &m_dU, },
		{ &m_hiddenWeights, &m_dW, },
		{ &m_outWeights, &m_dWout, } };
	};

};

#endif