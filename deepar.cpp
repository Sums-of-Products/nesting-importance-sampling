// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          https://www.boost.org/LICENSE_1_0.txt)

// Based on https://github.com/hjuha/nesting/blob/main/deepar.cpp (Harviainen & Koivisto, 2023)
// Changed the input format and allowed including accuracy parameters

#include <bits/stdc++.h>
#include <boost/math/distributions/gamma.hpp>
#include "Breal.hpp"

#define MAX_N 150
#define MAX_J 16
#define EPSILON 0.000000001
#define EULER 2.71828182845904523536028747135266249L
#define BOUND 2 // 0 - Adapart, 1 - Huber-Law, 2 - Extended Huber
#define TASSA 1 // Ensure the matrix has total support (0/1) 
#define SHARPEN 2 // Try to sharpen the matrix: 0 - Do nothing, 1 - Make the matrix nearly doubly stochastic, 2 - Try column sharpening on both original and nearly doubly stochastic matrix

using namespace std;

template<typename T>
using Matrix = vector<vector<T>>;

long double dp[MAX_N][1<<MAX_J];
long double row_gammas[MAX_N];

long double gamma_precomputed[MAX_N + 1];

mt19937 gen;

long double time_since(chrono::steady_clock::time_point start) {
	return std::chrono::duration_cast<std::chrono::milliseconds>(chrono::steady_clock::now() - start).count() / 1000.;
}

long long find_k(long double epsilon, long double delta) {
	long long max_k = (int)(EPSILON + ceil(2 / epsilon / epsilon / (1. - (4. * epsilon / 3.)) * log(2 / delta)));
	long long step = max_k / 2;
	while (step >= 1) {
		while (max_k - step > 1) {
			long long K = max_k - step;
			boost::math::gamma_distribution<> gm(K, 1. / (K - 1.));
			long double cumulative = boost::math::cdf(gm, 1. / (1. - epsilon)) - boost::math::cdf(gm, 1. / (1. + epsilon));
			if (cumulative >= 1 - delta) {
				max_k -= step;
			} else {
				break;
			}
		}
		step /= 2;
	}
	return max_k;
}

inline long double log_sum_exp(const long double x1, const long double x2) {
	if (isnan(x1)) return x2;
	if (isnan(x2)) return x1;
	long double x = max(x1, x2);
	return x + log(exp(x1 - x) + exp(x2 - x));
}

inline long double log_sub_exp(const long double x1, const long double x2) {
	if (isnan(x1)) return x2;
	if (isnan(x2)) return x1;
	long double x = max(x1, x2);
	return x + log(exp(x1 - x) - exp(x2 - x));
}

template <typename T>
long double permanent(Matrix<T> v) {
	long long n = v.size();
	vector<long double> row_sums(n);
	long double subset_sum = 0;
	long long s = 1;
	while (s < (1LL<<n)) {
		// Gray code magic
		long long bit = s ^ (s>>1LL) ^ (s - 1LL) ^ ((s - 1LL)>>1LL);
		long long j = 0;
		while (!((1LL<<j) & bit)) {
			j += 1;
		}
		if ((s ^ (s>>1LL)) & bit) {
			for (int i = 0; i < n; i++) {
				row_sums[i] += v[i][j];
			}
		} else {
			for (int i = 0; i < n; i++) {
				row_sums[i] -= v[i][j];
			}
		}
		long double prod = 1;
		int c = 0;
		for (long long i = 0; i < n; i++) {
			prod *= row_sums[i];
			c += ((s ^ (s>>1LL)) & (1LL<<i)) > 0;
		}
		subset_sum += ((c % 2) ? -1 : 1) * prod;
		s += 1;
	}
	return ((n % 2) ? -1 : 1) * subset_sum;
}

long double huber_h(long double r) {
	if (r <= EPSILON) return 0;
	if (BOUND == 1) {
		if (r <= 1) return (1 + (EULER - 1) * r) / EULER;
		return (r + 0.5 * log(r) + EULER - 1)  / EULER;
	} else if (BOUND == 2) {
		return 1;
	}
}

long double mb_gamma(long double x) {
	if (x < EPSILON) return 0;
	return exp(lgamma(x + 1) / x);
}

long double row_bound(long double x) {
	if (x < EPSILON) return NAN;
	if (BOUND) return log(huber_h(x));
	return log(mb_gamma(x));
}

void precompute(int n) {
	gen.seed(time(0));
	srand(time(0));
	for (int i = 0; i <= n; i++) {
		gamma_precomputed[i] = exp(row_bound(i));
	}
	gamma_precomputed[0] = 0;
	if (BOUND == 2) {
		gamma_precomputed[1] = EULER;
		for (int i = 2; i <= n; i++) {
			gamma_precomputed[i] = gamma_precomputed[i - 1] + 1 + .5 / gamma_precomputed[i - 1] + .6 / gamma_precomputed[i - 1] / gamma_precomputed[i - 1];	
		}
		for (int i = 0; i <= n; i++) {
			gamma_precomputed[i] /= EULER;
		}
	}
}

int weighted_random(vector<long double> weights) {
	// gumbel max trick
	int n = weights.size();
	uniform_real_distribution<> uniform_dist(0, 1);
	int initial = 0;
	while (initial < n && isnan(weights[initial])) initial++;
	if (initial == n) return -1;
	int max_i = initial;
	long double max_weight = -log(-log(uniform_dist(gen))) + weights[initial];
	for (int i = initial + 1; i < n; i++) {
		if (isnan(weights[i])) continue;
		long double w = -log(-log(uniform_dist(gen))) + weights[i];
		if (w > max_weight) {
			max_i = i;
			max_weight = w;
		}
	}
	return max_i;
}


long double dec_table[MAX_N][MAX_N];

template<typename T>
long double precompute_dp(Matrix<T> matrix, int J) {
	int n = matrix.size();

	if (BOUND == 2) {
		for (int i = 0; i < n; i++) {
			vector<T> r;
			for (int j = n - 1; j >= 0; j--) {
				r.push_back(matrix[i][j]);
				sort(r.rbegin(), r.rend());
				dec_table[i][j] = 0;
				for (int k = 0; k < r.size(); k++) {
					dec_table[i][j] += r[k] * (gamma_precomputed[k + 1] - gamma_precomputed[k]);
				}
			}
		}
	}

	for (int i = 0; i < n; i++) {
		if (BOUND == 1) {
			long double row_sum = 0;
			for (int j = J; j < n; j++) {
				row_sum += matrix[i][j];
			}
			row_gammas[i] = row_bound(row_sum);
		} else if (BOUND == 0 || BOUND == 2) {
			row_gammas[i] = 0;
			vector<T> row = matrix[i];
			sort(row.begin() + J, row.end());
			reverse(row.begin() + J, row.end());
			for (int j = J; j < n; j++) {
				row_gammas[i] += row[j] * (gamma_precomputed[j - J + 1] - gamma_precomputed[j - J]);
			}
			row_gammas[i] = log(row_gammas[i]);
		}
	}

	if (J > n) {
		cout<<"ERROR: m > n on line "<<__LINE__<<endl;
		exit(0);
	}
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < (1<<J); j++) {
			dp[i][j] = -INFINITY;
		}
	}
	long double row_prod = 0;
	bool row_prod_zero = false;
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < (1<<J); j++) {
			if (i) dp[i][j] = dp[i - 1][j] + row_gammas[i];
			Breal value;
			bool first = true;
			for (int k = 0; k < J; k++) {
				if (!(j & (1<<k))) continue;
				if (matrix[i][k] < EPSILON) continue;
				if (j == (1<<k)) {
					if (row_prod_zero) continue;
					first = false;
					value.set_log(row_prod);
					value = value * (double)matrix[i][k];
				} else if (i) {
					if (isnan(dp[i - 1][j ^ (1<<k)]) || isinf(dp[i - 1][j ^ (1<<k)])) continue;
					Breal addition;
					addition.set_log(dp[i - 1][j ^ (1<<k)]);
					addition = addition * (double)matrix[i][k];
					if (first) {
						value = addition;
						first = false;
					} else {
						value = value + addition;
					}
				}
			}
			if (!first) {
				dp[i][j] = log_sum_exp(dp[i][j], value.get_log());
			}
		}
		row_prod = row_prod + row_gammas[i];
		if (isnan(row_prod) || isinf(row_prod)) {
			row_prod_zero = true;
		}
	}
	return dp[n - 1][(1<<J) - 1];
}

template<typename T>
long double minc_bregman(Matrix<T> matrix) {
	int n = matrix.size();
	int m = matrix[0].size();
	long double bound = 0;
	for (int i = 0; i < n; i++) {
		vector<T> row_sorted;
		for (int j = 0; j < m; j++) {
			row_sorted.push_back(matrix[i][j]);
		}
		sort(row_sorted.rbegin(), row_sorted.rend());
		long double row_sum = 0;
		for (int j = 0; j < n; j++) {
			row_sum += (gamma_precomputed[j + 1] - gamma_precomputed[j]) * row_sorted[j];
		}
		bound += log(row_sum);
	}
	return bound;
}

template<typename T>
long double huber_law_bound(Matrix<T> matrix) {
	int n = matrix.size();
	int m = matrix[0].size();
	long double bound = 0;
	for (int i = 0; i < n; i++) {
		long double row_sum = 0;
		for (int j = 0; j < n; j++) {
			row_sum += matrix[i][j];
		}
		bound += log(huber_h(row_sum));
	}
	return bound;
}

template<typename T>
long double hybrid_bound(Matrix<T> matrix, int J) {
	if (J <= 0) return BOUND == 1 ? huber_law_bound(matrix) : minc_bregman(matrix);
	int n = matrix.size();
	return dp[n - 1][(1<<J) - 1];
}

long double precomputed_bounds[MAX_N][MAX_N];
bool column_used[MAX_N];
bool row_used[MAX_N];
vector<pair<long double, int>> rows[MAX_N];

template<typename T>
void precompute_partitions(Matrix<T> matrix) {
	int n = matrix.size();
	int m = rows[0].size();
	int order[m][m];
	long double left_sums[n][n];
	long double right_sums[n][n];
	int row_ctr = 0;

	for (int i = 0; i < m; i++) {
		if (row_used[i]) continue;
		for (int j = 0; j < n; j++) {
			left_sums[row_ctr][j] = right_sums[row_ctr][j] = 0;
		}
		int inv_row[m];
		for (int j = 0; j < m; j++) {
			inv_row[j] = 0;
		}
		for (int j = 0; j < m; j++) {
			if (column_used[rows[i][j].second]) continue;
			inv_row[rows[i][j].second] = 1;
		}
		for (int j = 1; j < m; j++) {
			inv_row[j] += inv_row[j - 1];
		}

		int order_ctr = 0;
		for (int j = 0; j < m; j++) {
			if (column_used[rows[i][j].second]) continue;
			order[row_ctr][inv_row[rows[i][j].second] - 1] = order_ctr;
			order_ctr++;
		}
		int left_ctr = 0;
		for (int j = 0; j < m; j++) {
			if (column_used[rows[i][j].second]) continue;
			left_sums[row_ctr][left_ctr] = (gamma_precomputed[left_ctr + 1] - gamma_precomputed[left_ctr]) * rows[i][j].first;
			if (left_ctr) left_sums[row_ctr][left_ctr] += left_sums[row_ctr][left_ctr - 1];
			left_ctr++;
		}
		int right_ctr = 0;
		for (int j = 0; j < m - 1; j++) {
			if (column_used[rows[i][m - j - 1].second]) continue;
			right_sums[row_ctr][n - right_ctr - 1] = (gamma_precomputed[n - right_ctr - 1] - gamma_precomputed[n - right_ctr - 2]) * rows[i][m - j - 1].first;
			if (right_ctr) right_sums[row_ctr][n - right_ctr - 1] += right_sums[row_ctr][n - right_ctr];
			right_ctr++;
		}
		row_ctr++;
	}

	for (int column = 0; column < n; column++) {
		long double row_sums[n];
		int zero_rows = 0;
		long double product = 0;

		for (int row = 0; row < n; row++) {
			if (order[row][column] == 0) {
				row_sums[row] = right_sums[row][1];
			} else if (order[row][column] == n - 1) {
				row_sums[row] = left_sums[row][n - 2];
			} else {
				row_sums[row] = left_sums[row][order[row][column] - 1] + right_sums[row][order[row][column] + 1];
			}

			if (row_sums[row] < EPSILON) {
				zero_rows++;
			} else {
				product += log(row_sums[row]);
			}
		}
		for (int row = 0; row < n; row++) {
			if (row_sums[row] < EPSILON) {
				if (zero_rows == 1) {
					precomputed_bounds[row][column] = product;
				} else {
					precomputed_bounds[row][column] = NAN;
				}
			} else {
				if (zero_rows) {
					precomputed_bounds[row][column] = NAN;
				} else {
					precomputed_bounds[row][column] = product - log(row_sums[row]);
				}
			}
		}
	}
}

template<typename T>
Matrix<T> reduce(Matrix<T> matrix, int row, int col) {
	int n = matrix.size();
	Matrix<T> reduced;
	for (int i = 0; i < n; i++) {
		if (i == row) continue;
		reduced.push_back(vector<T>());
		for (int j = 0; j < n; j++) {
			if (j == col) continue;
			reduced.back().push_back(matrix[i][j]);
		}
	}
	return reduced;
}

template<typename T>
bool rejection_sample(Matrix<T> matrix, int J) {
	int n = matrix.size();
	long double upper_bound = hybrid_bound(matrix, J) + EPSILON;

	// Exact sampling part O(Jn)
	vector<int> chosen_rows;
	int subset = (1<<J) - 1;
	for (int i = n - 1; i >= 0; i--) {
		if (!subset) break;
		if (i) {
			if (weighted_random({dp[i - 1][subset] + row_gammas[i], log_sub_exp(dp[i][subset], dp[i - 1][subset] + row_gammas[i])})) {
				chosen_rows.push_back(i);
				vector<long double> cols;
				for (int j = 0; j < J; j++) {
					if (subset == (1<<j)) cols.push_back(dp[i][subset]);
					else if (subset & (1<<j)) cols.push_back(log(matrix[i][j]) + dp[i - 1][subset ^ (1<<j)]);
					else cols.push_back(NAN);
				}
				int shift = weighted_random(cols);
				if (shift < 0) return false;
				subset ^= 1<<shift;
			}
		} else {
			if (subset != (subset & -subset)) { // at most one 1 bit remaining
				cout<<"Exact sampling failed: "<<subset<<endl;
				exit(0);
			}
			chosen_rows.push_back(0);
		}
	}
	set<int> remaining_rows_s;
	for (int i = 0; i < n; i++) {
		remaining_rows_s.insert(i);
	}

	for (int i : chosen_rows) {
		if (BOUND != 2) {
			matrix.erase(matrix.begin() + i);
		} else {
			remaining_rows_s.erase(remaining_rows_s.lower_bound(i));
		}
	}
	vector<int> remaining_rows;
	for (int i : remaining_rows_s) {
		remaining_rows.push_back(i);
	}
	
	n -= J;
	
	for (int i = 0; i < matrix.size(); i++) {
		vector<T> v;
		for (int j = J; j < matrix[i].size(); j++) {
			v.push_back(matrix[i][j]);
		}
		matrix[i] = v;
	}
	if (!n) return true;

	if (!BOUND) {
		// AdaPart O(n^3)
		upper_bound = minc_bregman(matrix) + EPSILON;
		for (int i = 0; i < n; i++) {
			rows[i].clear();
			column_used[i] = false;
			row_used[i] = false;
			for (int j = 0; j < n; j++) {
 				rows[i].push_back({matrix[i][j], j});
			}
			sort(rows[i].rbegin(), rows[i].rend());
		}
		while (n > 1) {
			vector<Matrix<T>> chosen_matrices = {matrix}; // matrices that have been partitioned
			vector<pair<int, int>> reduction_positions = {{-1, -1}}; // helper arrays for computing the partitions efficiently
			vector<int> reduction_indices = {0};
			vector<vector<int>> used_columns_list = {{}};
			vector<vector<int>> used_rows_list = {{}};
			vector<long double> matrix_weights = {upper_bound - EPSILON};
			vector<long double> weight_multipliers = {0};
			long double weight_sum = matrix_weights[0]; // sum of the weights of the matrices in the partition
			do {
				// pick a random matrix from the partition
				int matrix_index = rand() % reduction_positions.size();
				pair<int, int> reduction_pos = reduction_positions[matrix_index];
				int reduction_index = reduction_indices[matrix_index];
				vector<int> used_columns = used_columns_list[matrix_index];
				vector<int> used_rows = used_rows_list[matrix_index];
				if (reduction_pos.first == -1) {
					matrix = chosen_matrices[reduction_index];
				} else {
					matrix = reduce(chosen_matrices[reduction_index], reduction_pos.first, reduction_pos.second);
				}
				n = matrix.size();
				while (n == 1) {
					matrix_index = rand() % reduction_positions.size();
					reduction_pos = reduction_positions[matrix_index];
					reduction_index = reduction_indices[matrix_index];
					used_columns = used_columns_list[matrix_index];
					used_rows = used_rows_list[matrix_index];
					if (reduction_pos.first == -1) {
						matrix = chosen_matrices[reduction_index];
					} else {
						matrix = reduce(chosen_matrices[reduction_index], reduction_pos.first, reduction_pos.second);
					}
					n = matrix.size();
				}
				chosen_matrices.push_back(matrix);
				long double weight_multiplier = weight_multipliers[matrix_index];
				
				reduction_indices.erase(reduction_indices.begin() + matrix_index);
				reduction_positions.erase(reduction_positions.begin() + matrix_index);
				matrix_weights.erase(matrix_weights.begin() + matrix_index);
				used_columns_list.erase(used_columns_list.begin() + matrix_index);
				used_rows_list.erase(used_rows_list.begin() + matrix_index);
				weight_multipliers.erase(weight_multipliers.begin() + matrix_index);

				for (int col : used_columns) {
					column_used[col] = true;
				}
				for (int rw : used_rows) {
					row_used[rw] = true;
				}

				int col_inv[n];
				int row_inv[n];
				int col_ctr = 0;
				int row_ctr = 0;
				for (int i = 0; i < rows[0].size(); i++) {
					if (!column_used[i]) {
						col_inv[col_ctr] = i;
						col_ctr++;
					}
					if (!row_used[i]) {
						row_inv[row_ctr] = i;
						row_ctr++;
					}
				}

				// find the column whose partition has the smallest total weight
				long double min_row_sum = -INFINITY;
				int min_index = -1;
				precompute_partitions(matrix); // O(n^2)
				for (int column = 0; column < n; column++) {
					long double row_sum = NAN;
					for (int j = 0; j < n; j++) {
						row_sum = log_sum_exp(row_sum, weight_multiplier + log(matrix[j][column]) + precomputed_bounds[j][column]);
					}
					if (min_index == -1 || row_sum < min_row_sum) {
						min_index = column;
						min_row_sum = row_sum;
					}
				}

				// add the partition to our lists
				used_columns.push_back(col_inv[min_index]);
				for (int j = 0; j < n; j++) {
					reduction_indices.push_back(chosen_matrices.size() - 1);
					reduction_positions.push_back({j, min_index});
					used_columns_list.push_back(used_columns);
					used_rows.push_back(row_inv[j]);
					used_rows_list.push_back(used_rows);
					used_rows.pop_back();
					matrix_weights.push_back(weight_multiplier + log(matrix[j][min_index]) + precomputed_bounds[j][min_index]);
					weight_multipliers.push_back(weight_multiplier + log(matrix[j][min_index]));
				}

				// compute the sum of weights again
				weight_sum = NAN;
				for (long double ld : matrix_weights) {
					weight_sum = log_sum_exp(weight_sum, ld);
				}

				for (int col : used_columns) {
					column_used[col] = false;
				}
				for (int rw : used_rows) {
					row_used[rw] = false;
				}
			} while (weight_sum > upper_bound);
			
			// choose a random matrix from a weighted distribution or return slack
			matrix_weights.push_back(log_sub_exp(upper_bound, weight_sum));
			int choice = weighted_random(matrix_weights);
			if (choice == reduction_positions.size()) return false;
			matrix = reduce(chosen_matrices[reduction_indices[choice]], reduction_positions[choice].first, reduction_positions[choice].second);
			upper_bound = matrix_weights[choice] - weight_multipliers[choice] + EPSILON;
			vector<int> used_columns = used_columns_list[choice];
			vector<int> used_rows = used_rows_list[choice];
			for (int col : used_columns) {
				column_used[col] = true;
			}
			for (int rw : used_rows) {
				row_used[rw] = true;
			}
			n = matrix.size();
		}
	} else if (BOUND == 1) {
		// Huber-type self-reducible sampling O(n^2)
		upper_bound = 0;
		long double row_sums[n];
		for (int i = 0; i < n; i++) {
			row_sums[i] = 0;
			row_used[i] = column_used[i] = false;
			for (int j = 0; j < n; j++) {
				row_sums[i] += matrix[i][j];
			}
		}
		for (int j = 0; j < n; j++) {
			upper_bound += log(huber_h(row_sums[j]));
		}
		upper_bound += EPSILON;

		for (int i = 0; i < n; i++) {
			vector<long double> partition(n + 1);
			long double prod = 0;
			long double remaining = upper_bound;
			int zeros = 0;

			for (int j = 0; j < n; j++) {
				if (row_used[j]) continue;
				if (row_sums[j] - matrix[j][i] < EPSILON) {
					zeros++;
				} else {
					prod += log(huber_h(row_sums[j] - matrix[j][i]));
				}
			}
			for (int j = 0; j < n; j++) {
				if (row_used[j]) {
					partition[j] = NAN;
					continue;
				}
				if (row_sums[j] - matrix[j][i] < EPSILON) {
					partition[j] = log(matrix[j][i]) + prod + log(max(2 - zeros, 0));
				} else {
					partition[j] = log(matrix[j][i]) + prod - log(huber_h(row_sums[j] - matrix[j][i])) + log(max(1 - zeros, 0));
				}
				remaining = log_sub_exp(remaining, partition[j]);
			}
			partition[n] = remaining;
			int index = weighted_random(partition);
			if (index == n) return false;
			row_used[index] = true;
			for (int j = 0; j < n; j++) {
				row_sums[j] -= matrix[j][i];
			}
			upper_bound = partition[index] - log(matrix[index][i]);
		}
	} else if (BOUND == 2) {
		// Huber-type self-reducible sampling O(n^2)
		upper_bound = 0;
		for (int i = 0; i < n; i++) {
			upper_bound += log(dec_table[remaining_rows[i]][J]);
		}
		upper_bound += EPSILON;

		for (int i = 0; i < n - 1; i++) {
			vector<long double> partition(n - i + 1, NAN);
			long double prod = 0;
			long double remaining = upper_bound;
			int zeros = 0;

			for (int j = 0; j < n - i; j++) {
				if (dec_table[remaining_rows[j]][J + i + 1] < EPSILON) {
					zeros++;
				} else {
					prod += log(dec_table[remaining_rows[j]][J + i + 1]);
				}
			}
			for (int j = 0; j < n - i; j++) {
				if (dec_table[remaining_rows[j]][J + i + 1] < EPSILON) {
					partition[j] = log(matrix[remaining_rows[j]][i]) + prod;
				} else if (!zeros) {
					partition[j] = log(matrix[remaining_rows[j]][i]) + prod - log(dec_table[remaining_rows[j]][J + i + 1]);
				}
				remaining = log_sub_exp(remaining, partition[j]);
			}
			partition[n - i] = remaining;
			int index = weighted_random(partition);
			if (index == n - i) return false;
			upper_bound = partition[index] - log(matrix[remaining_rows[index]][i]);
			remaining_rows.erase(remaining_rows.begin() + index);
		}
		return matrix[remaining_rows[0]].back() >= EPSILON;
	}
	return true;
}

template <typename T>
vector<int> hopcroft_karp(Matrix<T> matrix) {
	int n = matrix.size();
	vector<int> bipartite_edges[n];
	int matching[n];
	int inv[n];
	for (int i = 0; i < n; i++) {
		matching[i] = inv[i] = -1;
		bipartite_edges[i].clear();
		for (int j = 0; j < n; j++) {
			if (matrix[i][j] >= EPSILON) {
				bipartite_edges[i].push_back(j + n);
			}
		}
	}

	vector<int> dag[2 * n];
	bool modified = false;
	do {
		modified = false;
		queue<int> q;
		int seen[2 * n];
		for (int i = 0; i < n; i++) {
			seen[i] = seen[i + n] = 0;
			dag[i].clear();
			dag[i + n].clear();
			if (matching[i] == -1) {
				q.push(i);
			}
		}
		// construct graph
		bool stop = false;
		while (!q.empty()) {
			int i = q.front();
			q.pop();
			if (seen[i]) continue;
			seen[i] = 1;
			if (i >= n) {
				if (inv[i - n] == -1) {
					stop = true;
				} else if (!seen[inv[i - n]] && !stop) {
					q.push(inv[i - n]);
					dag[i].push_back(inv[i - n]);
				}
			} else {
				for (int j : bipartite_edges[i]) {
					if (!seen[j] && !stop && matching[i] != j) {
						q.push(j);
						dag[i].push_back(j);
					}
				}
			}
		}
		bool handled[2 * n];
		for (int i = 0; i < 2 * n; i++) {
			seen[i] = -1;
			handled[i] = false;
		}

		stack<int> s;
		for (int i = 0; i < n; i++) {
			if (matching[i] == -1) {
				s.push(i);
				seen[i] = -2;
			}
			while (!s.empty()) {
				int i = s.top();
				s.pop();
				if (handled[i]) continue;
				handled[i] = true;
				if (i >= n && inv[i - n] == -1) {
					while (!s.empty()) {
						s.pop();
					}
					modified = true;
					int j = i;
					while (j != -2) {
						matching[seen[j]] = j;
						inv[j - n] = seen[j];
						j = seen[seen[j]];
					}
					break;
				} else {
					for (int j : dag[i]) {
						if (handled[j]) continue;
						seen[j] = i;
						s.push(j);
					}
				}
			}
		}
	} while (modified);

	vector<int> perm(n);
	for (int i = 0; i < n; i++) {
		perm[i] = matching[i] - n;
	}
	return perm;
}

template<typename T>
vector<int> scc(Matrix<T> m) {
	int n = m.size();
	vector<int> v[n];
	vector<int> r[n];
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			if (m[i][j] > EPSILON) {
				v[i].push_back(j);
				r[j].push_back(i);
			}
		}
	}
	vector<int> state(n);
	vector<int> order;

	for (int i = 0; i < n; i++) {
		if (state[i]) continue;
		stack<int> s;
		s.push(i);
		state[i] = 1;
		while (!s.empty()) {
			int i = s.top();
			if (!v[i].empty()) {
				int j = v[i].back();
				v[i].pop_back();
				if (!state[j]) {
					s.push(j);
					state[j] = 1;
				}
			} else {
				s.pop();
				order.push_back(i);
			}
		}
	}
	vector<int> components(n);
	int component_id = 1;
	reverse(order.begin(), order.end());
	for (int i : order) {
		if (components[i]) continue;
		stack<int> s;
		s.push(i);
		while (!s.empty()) {
			int i = s.top();
			s.pop();
			if (components[i]) {
				continue;
			}
			components[i] = component_id;
			for (int j : r[i]) {
				s.push(j);
			}
		}
		component_id += 1;
	}

	return components;
}

template<typename T>
Matrix<T> tassa(Matrix<T> matrix) {
	int n = matrix.size();
	vector<int> matching = hopcroft_karp(matrix);
	int p[n];
	int pr[n];
	for (int i = 0; i < n; i++) {
		if (matching[i] < 0) return {};
		p[i] = matching[i];
		pr[matching[i]] = i;
		matching[i] = i;
	}
	{
		Matrix<T> permuted;
		for (int i = 0; i < n; i++) {
			permuted.push_back(matrix[pr[i]]);
		}
		matrix = permuted;
	}
	vector<long double> cs(n);
	for (int i = 0; i < n; i++) {
		cs[i] = matrix[i][i];
		matrix[i][i] = 0;
	}
	vector<int> components = scc(matrix);
	Matrix<T> support = matrix;
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			support[i][j] = 0;
		}
	}
	for (int i = 0; i < n; i++) {
		support[i][i] = cs[i];
	}
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			if (matrix[i][j] > EPSILON && components[i] == components[j]) {
				support[i][j] = matrix[i][j];
			}
		}
	}
	{
		Matrix<T> permuted;
		for (int i = 0; i < n; i++) {
			permuted.push_back(support[p[i]]);
		}
		support = permuted;
	}
	return support;
}

/*
INPUT FORMAT
============
epsilon delta J time_limit
n
A_11  ..  ..  A_1n
 ..   ..       ..
 ..       ..   ..
A_n1  ..  ..  A_nn
*/
int main(int argc, char *argv[]) {
	long double epsilon = 0.01;
	long double delta = 0.05;
	if (argc == 3) {
		epsilon = atof(argv[1]);
		delta = atof(argv[2]);
	}
	long double time_limit;
	int n;
	cin>>n>>time_limit;
	
	int J = min(16, n);
	// Find necessary number of accepted samples for desired accuracy (Huber, 2017)
	long long K = find_k(epsilon, delta); // find_k(0.1, 0.05) = 385

	if (J > MAX_J) {
		cout<<"J can't be greater than "<<MAX_J<<endl;
		return 0;
	}
	if (J > n) {
		cout<<"J can't be greater than "<<n<<endl;
		return 0;
	}
	Matrix<long double> matrix;
	for (int i = 0; i < n; i++) {
		matrix.push_back(vector<long double>(n));
		for (int j = 0; j < n; j++) {
			cin>>matrix[i][j];
		}
	}

	chrono::steady_clock::time_point start_time = chrono::steady_clock::now();

	if (TASSA) { // remove entries that don't contribute to the permanent (Tassa, 2012)
		matrix = tassa(matrix);
		if (matrix.empty()) {
			cout<<"zero permanent"<<endl;
			return 0;
		}
	}

	
	long double coef = 0;
	precompute(n);
	if (SHARPEN) {
		if (!TASSA) {
			cout<<"SHARPEN cannot be 1 if TASSA is 0"<<endl;
			return 0;
		}
		{
			// test that the matrix uses floats
			auto x = matrix[0][0];
			matrix[0][0] = 0.5;
			if (abs(matrix[0][0] - 0.5) >= EPSILON) {
				cout<<"The type of the matrix must be some floating point type"<<endl;
			}
			matrix[0][0] = x;
		}

		#if SHARPEN == 2
		uniform_real_distribution<> uniform_dist(-1, 1);
		long double best = minc_bregman(matrix);
		cerr<<"Default bound: "<<best<<endl;
		for (int t = 0; t < n * n; t++) {
			int j = rand() % n;
			double x = pow(2, uniform_dist(gen)); // multiply by a number between [0.5, 2]
			for (int i = 0; i < n; i++) {
				matrix[i][j] *= x;
			}
			coef -= log(x);
			long double b = minc_bregman(matrix);
			if (b + coef < best) {
				best = b + coef;
			} else {
				coef += log(x);
				for (int i = 0; i < n; i++) {
					matrix[i][j] /= x;
				}
			}
		}
		cerr<<"Sharpened bound: "<<(minc_bregman(matrix) + coef)<<endl;
		auto m_copy = matrix;
		long double sharpened_coef = coef;
		#endif

		// "As a practical matter, doing n^2 iterations of row and column balancing, as described above, is usually more
		// than enough to get convergence to a reasonable for the purposes of computing the permanent" -Sullivan & Beichl
		for (int t = 0; t < n * n; t++) {
			for (int i = 0; i < n; i++) {
				long double sum = 0;
				for (int j = 0; j < n; j++) {
					sum += matrix[j][i];
				}
				for (int j = 0; j < n; j++) {
					matrix[j][i] /= sum;
				}
				coef += log(sum);
			}
			for (int i = 0; i < n; i++) {
				long double sum = 0;
				for (int j = 0; j < n; j++) {
					sum += matrix[i][j];
				}
				for (int j = 0; j < n; j++) {
					matrix[i][j] /= sum;
				}
				coef += log(sum);
			}
		}
		for (int i = 0; i < n; i++) {
			long double mx = 0;
			for (int j = 0; j < n; j++) {
				mx = max(mx, matrix[i][j]);
			}
			for (int j = 0; j < n; j++) {
				matrix[i][j] /= mx;
			}
			coef += log(mx);
		}

		#if SHARPEN == 2
		cerr<<"Doubly stochastic: "<<(minc_bregman(matrix) + coef)<<endl;

		best = minc_bregman(matrix) + coef;
		for (int t = 0; t < n * n; t++) {
			int j = rand() % n;
			double x = pow(2, uniform_dist(gen)); // multiply by a number between [0.5, 2]
			for (int i = 0; i < n; i++) {
				matrix[i][j] *= x;
			}
			coef -= log(x);
			long double b = minc_bregman(matrix);
			if (b + coef < best) {
				best = b + coef;
			} else {
				coef += log(x);
				for (int i = 0; i < n; i++) {
					matrix[i][j] /= x;
				}
			}
		}
		cerr<<"Doubly stochastic sharpened: "<<(minc_bregman(matrix) + coef)<<endl;
		if (minc_bregman(matrix) + coef > minc_bregman(m_copy) + sharpened_coef) {
			matrix = m_copy;
			coef = sharpened_coef;
		}
		#endif
	}
	precompute_dp(matrix, J);

	long double ub = hybrid_bound(matrix, J);
	long double succ = 0;
	// long double exact = log(permanent(matrix)); // permanent of ENZYMES_g192: 713143040
	cout<<setprecision(10)<<fixed;

	long double preprocessing_time = time_since(start_time);

	// huber 2017
	long long S = 0;
	long double R = 0;
	default_random_engine generator;
	generator.seed(time(0));
	exponential_distribution<double> distribution(1);
	while (S != K) {
		R += distribution(generator);
		if (rejection_sample(matrix, J)) {
			S++;
		}
		if (time_since(start_time) > time_limit) {
			cout<<0<<" "<<time_limit<<endl;
			return 0;
		}
	}
	
	long double p = (K - 1) / R;
	
	// estimated lower bound: coef + ub + log(lc); estimated upper bound: coef + ub + log(uc)
	long double lc = p / (1 + epsilon);
	long double uc = p / (1 - epsilon);

	long double sampling_time = time_since(start_time) - preprocessing_time;

	// -19.8152856433
	// OUTPUT: estimate preprocessing_time sampling_time total_time
	cout<<(coef + ub + log(p))<<" "<<time_since(start_time)<<endl;
}
