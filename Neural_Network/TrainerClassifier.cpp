#include "TrainerClassifier.hpp"

TrainerClassifier::TrainerClassifier(RNN& model, const hyperparameters& hyper) : _model(model), _hyper(hyper) {
	_scope = nullptr;
	_xtrain = nullptr;
	_ytrain = nullptr;
	_xvalid = nullptr;
	_yvalid = nullptr;
}

void TrainerClassifier::set_scope(ScopeClassifier& scope) {
	_scope = &scope;
}

void TrainerClassifier::set_data(Dataset& train, Dataset& validation) {
	_xtrain = &train.x;
	_ytrain = &train.y;

	_xvalid = &validation.x;
	_yvalid = &validation.y;
}

void TrainerClassifier::run() {

	const int n_batches = _hyper.n_batch / _hyper.batch_size;
	for (int epoch = 0; epoch < _hyper.max_epochs; epoch++) {

		double epoch_loss = 0;
		int train_correct = 0;
		int val_correct = 0;

		// Test accuracy
		for (int n = 0; n < n_batches; n++) {
			const std::vector<Matrix>& X = (*_xtrain)[n];
			const Matrix& Y = (*_ytrain)[n];

			_model.forward(X);
			_model.backpropagation(X, Y);

			_scope->step(_model);

			// Loss & accuracy
			const Matrix& ypred = _model.getOutput();
			epoch_loss += CELossFunction(ypred, Y);
			for (int b = 0; b < _hyper.batch_size; b++)
				if ((Y(b, 0) < Y(b, 1)) == (ypred(b, 0) < ypred(b, 1)))
					train_correct++;
			
		}
		epoch_loss /= _hyper.n_batch;
		double train_accuracy = 100.0 * train_correct / _hyper.n_batch;

		// Validation accuracy
		for (int n = 0; n < _hyper.test_size; n++) {
			const std::vector<Matrix>& X = (*_xvalid)[n];
			const Matrix& Y = (*_yvalid)[n];

			_model.forward(X);
			
			const Matrix& ypred = _model.getOutput();
			if ((Y(0, 0) < Y(0, 1)) == (ypred(0, 0) < ypred(0, 1)))
				val_correct++;
		}
		double val_accuracy = 100.0 * val_correct / _hyper.test_size;

		// Printing the results
		print("[Epoch ", epoch+1, "/", _hyper.max_epochs, "] ",
			  "Loss = ", epoch_loss, " | ",
			  "train_acc = ", train_accuracy, " % | ",
			  "test_acc = ", val_accuracy, " %");

		// Early stopper (very primitive for now. Have to implement it in the ScopeClassifier)
		if (val_accuracy > 99 && train_accuracy > 99) break;
	}
}