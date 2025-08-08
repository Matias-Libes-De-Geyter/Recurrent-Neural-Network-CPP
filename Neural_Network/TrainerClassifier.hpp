/*trainer:



set_scope():
	mettre le scope

set_data():
	voilà

run():
	entrainer*/

#include "ScopeClassifier.hpp"


#ifndef TRAINER_H
#define TRAINER_H

class TrainerClassifier {
private:
	const hyperparameters _hyper;
	RNN& _model;

	ScopeClassifier* _scope;
	std::vector<std::vector<Matrix>>* _xtrain;
	std::vector<Matrix>* _ytrain;

public:
	TrainerClassifier(RNN&, const hyperparameters&);
	void set_scope(ScopeClassifier&);
	void set_data(std::vector<std::vector<Matrix>>&, std::vector<Matrix>&);
	void run();
};

#endif