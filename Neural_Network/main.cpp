#include "RNN.hpp"
#include "TrainerClassifier.hpp"
#include "ScopeClassifier.hpp"
#include "Dataset.hpp"

hyperparameters hyper = {
    seq_len : 6,      // input_number
    input_dimension : 1,     // time_steps (non utilisé ici, mais conservé)
    hidden_dimension : 8,     // hidden_dimension
    output_dimension : 2,      // output_dimension
    learning_rate : 0.01,   // learning_rate
    max_epochs : 100,
    n_batch : 1000,
    batch_size : 1
};

/*
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
    Dataset train = DataLoader(hyper);

    TrainerClassifier trainer(model, hyper);



    trainer.set_scope(scope);
    trainer.set_data(train.x, train.y);
    print("Data has been successfully imported");

    trainer.run();

    Matrix X(1, 1);
    std::vector<Matrix> A;

    for(int i = 0; i < hyper.seq_len; i++) {
        std::cin >> X(0, 0);
        A.push_back(X);
    }

    
    model.forward(A);

    print(model.getOutput());


    Matrix Y(1, 2);
    int label = ((int)X(0, 0) + (int)X(0, 1) + (int)X(0, 2)) % 2;
    Y(0, 0) = (label == 0 ? 1.0 : 0.0);
    Y(0, 1) = (label == 1 ? 1.0 : 0.0);

    return 0;
}
