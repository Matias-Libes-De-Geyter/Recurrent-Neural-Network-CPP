#include "RNN.hpp"
#include "TrainerClassifier.hpp"
#include "ScopeClassifier.hpp"
#include "Dataset.hpp"

hyperparameters hyper = {
    seq_len: 5,      // input_number
    input_dimension : 1,     // time_steps (non utilisé ici, mais conservé)
    hidden_dimension : 8,     // hidden_dimension
    output_dimension : 2,      // output_dimension
    learning_rate : 0.005,   // learning_rate
    max_epochs : 100,
    n_batch : 1000,
    batch_size : 5,
    test_size : 100
};

/* FONCTIONNENT:
    seq_len : 5,      // input_number
    input_dimension : 1,     // time_steps (non utilisé ici, mais conservé)
    hidden_dimension : 8,     // hidden_dimension
    output_dimension : 2,      // output_dimension
    learning_rate : 0.005,   // learning_rate
    max_epochs : 100,
    n_batch : 1000,
    batch_size : 1*/

/*
    seq_len : 3,      // input_number
    input_dimension : 1,     // time_steps (non utilisé ici, mais conservé)
    hidden_dimension : 32,     // hidden_dimension
    output_dimension : 2,      // output_dimension
    learning_rate : 0.001,   // learning_rate
    max_epochs : 100,
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
