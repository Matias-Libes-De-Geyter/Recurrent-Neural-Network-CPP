#include "TrainerClassifier.hpp"

TrainerClassifier::TrainerClassifier(RNN& model, const hyperparameters& hyper) : _model(model), _hyper(hyper) {
	_scope = nullptr;
	_xtrain = nullptr;
	_ytrain = nullptr;
}

void TrainerClassifier::set_scope(ScopeClassifier& scope) {
	_scope = &scope;
}

void TrainerClassifier::set_data(std::vector<std::vector<Matrix>>& x_train, std::vector<Matrix>& y_train) {
	_xtrain = &x_train;
	_ytrain = &y_train;
}

void TrainerClassifier::run() {

	const int n_batches = _hyper.n_batch / _hyper.batch_size;
	for (int epoch = 0; epoch < _hyper.max_epochs; epoch++) {

		double epoch_loss = 0;
		int correct = 0;

		for (int n = 0; n < n_batches; n++) {
			const std::vector<Matrix>& X = (*_xtrain)[n];
			const Matrix& Y = (*_ytrain)[n];

			_model.forward(X);
			_model.backpropagation(X, Y);

			_scope->step(_model);

			// Loss & accuracy
			const Matrix& ypred = _model.getOutput();
			epoch_loss += CELossFunction(ypred, Y);
			if ((Y(0, 0) < Y(0, 1)) == (ypred(0, 0) < ypred(0, 1)))
				correct++;
		}

		epoch_loss /= _hyper.n_batch;
		double accuracy = 100.0 * correct / _hyper.n_batch;

		print("[Epoch ", epoch+1, "/", _hyper.max_epochs, "] ",
			  "Loss = ", epoch_loss, " | ",
			  "acc = ", accuracy, " % ");
	}
}