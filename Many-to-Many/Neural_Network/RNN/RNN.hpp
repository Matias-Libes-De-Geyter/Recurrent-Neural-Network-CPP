#include "..\Utilities/functions.hpp"

#ifndef RNN_HPP
#define RNN_HPP

class RNN {

private:
	const hyperparameters& _hyper;

	// Weights, values matrices and their gradient.
	Matrix m_inWeights;
	Matrix m_hiddenWeights;
	Matrix m_outWeights;

	Matrix m_dU;
	Matrix m_dW;
	Matrix m_dWout;

	std::vector<Matrix> m_Z;
	std::vector<Matrix> m_dZ;
	std::vector<Matrix> m_hiddenStates;
	std::vector<Matrix> m_Y;
	std::vector<Matrix> m_dY;

public:
	RNN(const hyperparameters& hyper);

	// Forward and Backprop
	void forward(const std::vector<Matrix>& input);
	void backpropagation(const std::vector<Matrix>& input, const std::vector<Matrix>& y_real);

	// Return the output vector
	inline const std::vector<Matrix>& getOutput() const {
		return m_Y;
	};

	// Return all the diff parameters and their gradient for later-on optimization (in Scope)
	inline std::vector<std::pair<Matrix*, Matrix*>> getParameters() {
		return {
		{ &m_inWeights, &m_dU, },
		{ &m_hiddenWeights, &m_dW, },
		{ &m_outWeights, &m_dWout, } };
	};

};

#endif