# Recurrent Neural Network (in CPP)
The aim of this project was to program a many-to-many Recurrent Neural Network from scratch in C++.

### Model
- **Model:** Recurrent Neural Network,
- **Optimizer:** Adam optimizer,
- **Regularization methods:** No regularization method.

## Introduction
The aim of this project was to create a simple Recurrent Neural Network (RNN) from scratch, in C++. Using Adam optimizer to update weight, we can get a perfect accuracy on predicting the binary nature of the sum of several binary numbers (up to 100 for now). The underlying aim of the project is to get a grasp on how this type of architecture works, in order to later on create a GRU (or LSTM). The final goal would be to handle words through tokenization.

### Why C++ ?
Firstly, I basically chose C++ because I'm much more familiar with it than Python. But more generally, the main reason to use C++ here was to fully exploit the capabilities of pointers and adresses, to get the most optimized program I could write in a given lapse of time. Indeed, a lot of time was spent to improve performance.

## Methodology
Firstly, I created the RNN class, which uses three different weights. U for the input weight, W for the hidden layer weight, and Wout for the output weight. These are used inside the forward method to generate the output, and their gradient are updated through the backpropagation method. The weights are then updated when the Scope is called, using Adam optimizer. A Dataset struct creates the Dataset, while the TrainerClassifier class gathers the training and validation process.

### Dataset

This dataset consists in synthetic data for a binary parity problem. Each example is a sequence of seq_len bits (0/1). At each time step t, the target is the cumulative parity sum(bits[0..t]) mod 2. The loader generates batches of size batch_size. The dataset is produced with random generation to ensure reproducibility.

### Hyperparameters

We have two slightly different architectures. A many-to-many, and a many-to-one. Since the many-to-many is the one that gives the best results (since a RNN is good at predicting the next step), we are going to focus on this one.
- I used a learning rate of $$5 \cdot 10^{-3}$$.
- The inputs and outputs are vectors with dimension $$1$$.
- The sequence length is $$100$$.
- The weights of the hidden layers has a dimension of $$16$$.
- In Adam optimizer, $$\beta_m = 0.9$$ and $$\beta_v = 0.999$$.
- I used batches of **16 bits**, and used it on the whole dataset.

## Results

### Observations
- Results on the binary nature of the sum of binary numbers database. When ran into the whole training database, the model gives the following results:
![Plots](Many-to-Many/img/latest_output.png)

Here, values are plotted after each epochs. The early stopper stopped 3 epochs before the end. We can see the training accuracy, validation accuracy and training loss for each epochs.

### Discussion
- A sequence with too much length is subject to the vanishing gradient issue.
- **Next steps:**
  - The next move would be to implement max-pooling layers and/or batch normalization, to lengthen the max sequence length we can have. It would help to tackle the vanishing gradient problem.
  - I didn't implement flooding. It could improve the model.
  - For performance, we could also try alternative optimizers like RMSprop, AdamW or Nadam. We could also experiment by implementing Dropout between fully connected layers.

---

## How to Use

- Run the ```RNN.bat``` file. It closes when it finished training (accuracy 100 for 3 epochs).
- To plot the output of the training, run the ```plot.py``` file from the main folder.

To change other hyperparameters, you must recompile everything for now. The command to compile is: ```mingw32-make -f MakeFile```.



## Requirements

- Mingw32 compiler version ```gcc-14.2.0-mingw-w64ucrt-12.0.0-r2```.
- Python 3.x .

---

## Repository Structure

```plaintext

NeuralNetwork/
├── Many-to-Many/
│   │
│   ├── executable/
│   │   ├── main.exe            # Main executable
│   │   ├── model_weights.txt   # Save of the weights. Used to run the program without having to train it everytime
│   │   └── xxx.dll             # C++ Dlls used in the main.exe file.
│   │
│   ├── img/
│   │   └── latest_output.png   # Output with 100 time steps
│   │
│   ├── Neural_Network/     # Main codes of the repository
│   │   ├── Classifier/
│   │   │   ├── Scope.cpp
│   │   │   ├── Scope.hpp
│   │   │   ├── TrainerClassifier.cpp
│   │   │   └── TrainerClassifier.hpp
│   │   ├── Dataset/
│   │   │   └── Dataset.hpp
│   │   ├── RNN/
│   │   │   ├── RNN.cpp
│   │   │   ├── RNN.hpp
│   │   ├── Utilities/
│   │   │   ├── functions.cpp
│   │   │   ├── functions.hpp
│   │   │   ├── Matrix.cpp
│   │   │   └── Matrix.hpp
│   │   │
│   │   ├── main.cpp        # Main code that initiate all variables
│   │   └── plot.py         # Run "py Neural_Network/plot.py" to get a plot of the result of the training
│   │
│   ├── MakeFile
│   ├── RNN.bat             # Execute this file to test the program
│   └── training_data.csv   # Output from the training process, to plot the loss and accuracy
├── Many-to-One/
├── └── Same
└── README.md

```

---