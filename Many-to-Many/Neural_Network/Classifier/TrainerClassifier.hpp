#include "Scope.hpp"
#include "..\Dataset/Dataset.hpp"


#ifndef TRAINER_HPP
#define TRAINER_HPP

class TrainerClassifier {
private:
	const hyperparameters _hyper;
	RNN& _model;

	Scope* _scope;
	std::vector<std::vector<Matrix>>* _xtrain;
	std::vector<std::vector<Matrix>>* _ytrain;
	std::vector<std::vector<Matrix>>* _xvalid;
	std::vector<std::vector<Matrix>>* _yvalid;

public:
	TrainerClassifier(RNN&, const hyperparameters&);
	void set_scope(Scope&);
	void set_data(Dataset&, Dataset&);
	void run();
};

#endif