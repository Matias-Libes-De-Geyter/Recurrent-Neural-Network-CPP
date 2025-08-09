#include "functions.h"


#pragma once

#ifndef DATASET_HPP
#define DATASET_HPP

struct Dataset {
	std::vector<std::vector<Matrix>> x;
	std::vector<Matrix> y;
};

inline Dataset DataLoader(const hyperparameters& hyper, const std::string& dataset_type) {

	int n_iter = 0;
	int batch_size = 0;
	if (dataset_type == "train") {
		n_iter = hyper.n_batch / hyper.batch_size;
		batch_size = hyper.batch_size;
	}
	else if (dataset_type == "test") {
		n_iter = hyper.test_size;
		batch_size = 1;
	}
	else
		print("Dataset type is wrong");


	Dataset data;
	data.x.reserve(n_iter);
	data.y.reserve(n_iter);

	for (int n = 0; n < n_iter; n++) {

		std::vector<Matrix> x_batch;
		x_batch.reserve(hyper.seq_len);

		std::vector<int> sum(batch_size);

		for (int i = 0; i < hyper.seq_len; i++) {
			Matrix X(batch_size, 1);

			for (int b = 0; b < batch_size; b++) {
				X(b, 0) = random_bit();
				sum[b] += X(b, 0);
			}

			x_batch.emplace_back(std::move(X));
		}


		Matrix Y(batch_size, hyper.output_dimension);
		for (int b = 0; b < batch_size; b++) {

			Y(b, 0) = (sum[b] % 2 == 0 ? 1 : 0);
			Y(b, 1) = (sum[b] % 2 == 1 ? 1 : 0);
		}

		data.x.emplace_back(std::move(x_batch));
		data.y.emplace_back(std::move(Y));
	}

	return data;
};

#endif