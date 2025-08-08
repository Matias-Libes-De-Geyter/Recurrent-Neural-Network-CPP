#include "RNN.hpp"


RNN::RNN(hyperparameters& hyper) : _hyper(hyper), m_inWeights(_hyper.input_dimension+1, _hyper.hidden_dimension),
												  m_hiddenWeights(_hyper.hidden_dimension+1, _hyper.hidden_dimension),
												  m_outWeights(_hyper.hidden_dimension, _hyper.output_dimension) {

	m_dU = m_inWeights;
	m_dW = m_hiddenWeights;
	m_dWout = m_outWeights;

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

	m_hiddenStates.clear();
	m_Z.clear();
	m_deltas.clear();

}

void RNN::forward(const std::vector<Matrix>& input) {

	m_Z.resize(_hyper.seq_len);
	m_hiddenStates.resize(_hyper.seq_len + 1);
	m_hiddenStates[0] = Matrix(_hyper.batch_size, _hyper.hidden_dimension);

	std::vector<Matrix> input_unconst = input;
	for (int t = 0; t < _hyper.seq_len; ++t) {
		// h = a(z) = a(x*U + h*W + b)
		m_Z[t] = input_unconst[t].addBias() * m_inWeights + m_hiddenStates[t].addBias() * m_hiddenWeights;
		m_hiddenStates[t + 1] = activate(m_Z[t]);
	}

	m_output = m_hiddenStates.back() * m_outWeights;
	m_output = softmax_activation(m_output);

}

void RNN::backpropagation(const std::vector<Matrix>& input, const Matrix& y_real) {

	// Remove the unconst later on, I want to change Matrix.h to have function that does everything directly.
	std::vector<Matrix> input_unconst = input;


	m_dU = Matrix(m_inWeights.rows(), m_inWeights.cols());
	m_dW = Matrix(m_hiddenWeights.rows(), m_hiddenWeights.cols());
	m_dWout = Matrix(m_outWeights.rows(), m_outWeights.cols());

	Matrix dy = m_output - y_real;
	m_dWout = m_hiddenStates.back().T() * dy;

	Matrix dh_next = Matrix(_hyper.batch_size, _hyper.hidden_dimension);
	for (int t = _hyper.seq_len - 1; t >= 0; --t) {
		// injecter dy_final uniquement pour t == T-1
		Matrix dh = dh_next;
		if (t == _hyper.seq_len - 1)
			dh += dy * m_outWeights.T();

		// dérivée de tanh(z)
		Matrix dz = dh.hadamard(deriv_activate(m_Z[t]));

		// accumuler
		m_dU += input_unconst[t].addBias_then_T() * dz;
		m_dW += m_hiddenStates[t].addBias().T() * dz;

		// préparer pour le pas précédent
		dh_next = dz * m_hiddenWeights.removeBias();
	}

}

Matrix RNN::activate(Matrix inputs) {

	for (size_t i = 0; i < inputs.rows(); i++)
		for (size_t j = 0; j < inputs.cols(); j++)
			inputs(i, j) = std::tanh(inputs(i, j));
	return inputs;
}
Matrix RNN::deriv_activate(Matrix inputs) {

	for (size_t i = 0; i < inputs.rows(); ++i) {
		for (size_t j = 0; j < inputs.cols(); ++j) {
			double v = std::tanh(inputs(i, j));
			inputs(i, j) = 1 - v * v;
		}
	}
	return inputs;
}

Matrix RNN::softmax_activation(Matrix inputs) {

	d_vector maxs(inputs.rows());
	for (size_t i = 0; i < inputs.rows(); i++) {
		maxs[i] = inputs(i, 0);
		for (size_t j = 0; j < inputs.cols(); j++)
			if (inputs(i, j) > maxs[i])
				maxs[i] = inputs(i, j);
	}

	Matrix expvalues = inputs;
	d_vector sum_of_exps(inputs.rows(), 0);
	for (size_t i = 0; i < inputs.rows(); i++) {
		for (size_t j = 0; j < inputs.cols(); j++) {
			expvalues(i, j) = pow(EULERS_NUMBER, inputs(i, j) - maxs[i]);
			sum_of_exps[i] += expvalues(i, j);
		}
	}

	for (size_t i = 0; i < inputs.rows(); i++)
		for (size_t j = 0; j < inputs.cols(); j++)
			inputs(i, j) = expvalues(i, j) / sum_of_exps[i]; // m_output = probabilities

	return inputs;

}
