#include "RNN/RNN.hpp"
#include "Classifier/TrainerClassifier.hpp"
#include "Classifier/Scope.hpp"
#include "Dataset/Dataset.hpp"

hyperparameters hyper = {
    seq_len: 5,
    input_dimension : 1,
    hidden_dimension : 16,
    output_dimension : 2,
    learning_rate : 0.005,
    max_epochs : 100,
    n_batch : 1000,
    batch_size : 16,
    test_size : 100
};

/* FONCTIONNENT:
    seq_len: 5,
    input_dimension : 1,
    hidden_dimension : 8,
    output_dimension : 2,
    learning_rate : 0.005,
    max_epochs : 100,
    batch_size : 16,
    n_batch : 1000,
    batch_size : 1*/

/*
    seq_len: 3,
    input_dimension : 1,
    hidden_dimension : 32,
    output_dimension : 2,
    learning_rate : 0.001,
    max_epochs : 100,
    batch_size : 1,
    n_batch : 1000,
    batch_size : 1*/

const bool store = true;

int main() {

    RNN model(hyper);

    Scope scope(model, hyper);

    TrainerClassifier trainer(model, hyper);

    Dataset train = DataLoader(hyper, "train");
    Dataset validation = DataLoader(hyper, "test");

    trainer.set_scope(scope);
    trainer.set_data(train, validation);
    print("Data has been successfully imported");

    trainer.run(store);

    return 0;
}
