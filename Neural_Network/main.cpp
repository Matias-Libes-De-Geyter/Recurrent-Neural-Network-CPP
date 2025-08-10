#include "RNN.hpp"
#include "TrainerClassifier.hpp"
#include "ScopeClassifier.hpp"
#include "Dataset.hpp"

hyperparameters hyper = {
    seq_len: 5,
    input_dimension : 1,
    hidden_dimension : 8,
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

int main() {
    RNN model(hyper);

    ScopeClassifier scope(model, hyper);
    
    TrainerClassifier trainer(model, hyper);

    Dataset train = DataLoader(hyper, "train");
    Dataset validation = DataLoader(hyper, "test");

    trainer.set_scope(scope);
    trainer.set_data(train, validation);
    print("Data has been successfully imported");

    trainer.run();

    return 0;
}
