#include "RNN.hpp"


RNN::RNN(const hyperparameters& hyper) : _hyper(hyper), m_inWeights(_hyper.input_dimension+1, _hyper.hidden_dimension),
												  m_hiddenWeights(_hyper.hidden_dimension+1, _hyper.hidden_dimension),
												  m_outWeights(_hyper.hidden_dimension, _hyper.output_dimension) {

	m_dU = m_inWeights;
	m_dW = m_hiddenWeights;
	m_dWout = m_outWeights;

	// Arbitrary number. Else I would have to change the limit for each weight.
	double limit = 0.6;

	// inW and outW are initialised as already transposed, for better performance
	for (size_t i = 0; i < _hyper.hidden_dimension; i++)
		for (size_t j = 0; j < _hyper.input_dimension+1; j++)
			m_inWeights(j, i) = random(-limit, limit);

	for (size_t i = 0; i < _hyper.hidden_dimension+1; i++)
		for (size_t j = 0; j < _hyper.hidden_dimension; j++)
			m_hiddenWeights(i, j) = random(-limit, limit);

	for (size_t i = 0; i < _hyper.output_dimension; i++)
		for (size_t j = 0; j < _hyper.hidden_dimension; j++)
			m_outWeights(j, i) = random(-limit, limit);

	m_Z.clear();
	m_Z.resize(_hyper.seq_len);
	m_hiddenStates.clear();
	m_hiddenStates.resize(_hyper.seq_len + 1);
	m_dZ.clear();
	m_dZ.resize(_hyper.seq_len);

}

// Forward method
void RNN::forward(const std::vector<Matrix>& input) {

	m_hiddenStates[0] = Matrix(input[0].rows(), _hyper.hidden_dimension);

	// For each time step, Ht+1 = tanh(X*U + Ht*W) = tanh(Zt)
	for (int t = 0; t < _hyper.seq_len; ++t) {
		m_Z[t] = MATRIX_OPERATION::addbiases_then_mult(input[t], m_inWeights) + MATRIX_OPERATION::addbiases_then_mult(m_hiddenStates[t], m_hiddenWeights);
		m_hiddenStates[t + 1] = ACTIVATION::tanh_activation(m_Z[t]);
	}

	// Outputting Y = a(Hlast * Wout) 
	m_output = m_hiddenStates.back() * m_outWeights;
	m_output = ACTIVATION::softmax_activation(m_output);
}

// Backpropagation method. No optimization, just updating the gradients.
void RNN::backpropagation(const std::vector<Matrix>& input, const Matrix& y_real) {

	m_dU.fill(0.0);
	m_dW.fill(0.0);
	m_dWout.fill(0.0);

	Matrix delta_out = m_output - y_real;
	MATRIX_OPERATION::compute_out_weights(m_dWout, m_hiddenStates[_hyper.seq_len], delta_out);
	
	for (int t = _hyper.seq_len - 1; t >= 0; t--) {
		Matrix dH = delta_out * m_outWeights.T();
		if(t < _hyper.seq_len - 1)
			dH = m_dZ[t + 1] * m_hiddenWeights.removeBias().T();
		m_dZ[t] = dH.hadamard(ACTIVATION::deriv_tanh(m_Z[t]));

		MATRIX_OPERATION::compute_weigths(m_dU, input[t], m_dZ[t]);
		MATRIX_OPERATION::compute_weigths(m_dW, m_hiddenStates[t], m_dZ[t]);
	}

	// Normalizing the gradients
	double norm = 1.0 / _hyper.batch_size;
	m_dU *= norm;
	m_dW *= norm;
}