#include "functions.h"

#pragma once

struct Dataset {
	std::vector<std::vector<Matrix>> x;
	std::vector<Matrix> y;
};

Dataset DataLoader(hyperparameters& hyper) {
	Dataset data;
	data.x.reserve(hyper.n_batch / hyper.batch_size);
	data.y.reserve(hyper.n_batch / hyper.batch_size);

	// We get n_batch/batch_size
	for (int n = 0; n < hyper.n_batch / hyper.batch_size; n++) {

		std::vector<Matrix> x_batch;
		x_batch.reserve(hyper.seq_len);

		std::vector<int> sum(hyper.batch_size);

		for (int i = 0; i < hyper.seq_len; i++) {
			Matrix X(hyper.batch_size, 1);

			for (int b = 0; b < hyper.batch_size; b++) {
				X(b, 0) = random_bit();
				sum[b] += X(b, 0);
			}

			x_batch.emplace_back(std::move(X));
		}


		Matrix Y(hyper.batch_size, hyper.output_dimension);
		for (int b = 0; b < hyper.batch_size; b++) {

			Y(b, 0) = (sum[b] % 2 == 0 ? 1 : 0);
			Y(b, 1) = (sum[b] % 2 == 1 ? 1 : 0);
		}
		
		data.x.emplace_back(std::move(x_batch));
		data.y.emplace_back(std::move(Y));
	}

	return data;
}