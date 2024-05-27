// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          https://www.boost.org/LICENSE_1_0.txt)

// Heavily modified from https://github.com/hjuha/nesting/blob/main/deepar.cpp (Harviainen & Koivisto, 2023)

#include <bits/stdc++.h>
#include "Breal.hpp"

#define EPSILON 0.000000001
#define EULER 2.71828182845904523536028747135266249L

using namespace std;

template<typename T>
using Matrix = vector<vector<T>>;
int n; // matrix dimension
long double coef = 0;

long double *row_gammas;
long double *gamma_precomputed;
long double *dp;

long double *weight;
long double ub;

long double time_limit;
chrono::steady_clock::time_point start_time;

#define MAX_J 16
int J = MAX_J;

const int D_INDEX(const int row, const int mask) { // map 2d to 1d
	return row * (1<<J) + mask;
}

mt19937 gen;

template<typename T>
long double deep_sample(Matrix<T> matrix);

long double time_since(chrono::steady_clock::time_point start) {
	return std::chrono::duration_cast<std::chrono::milliseconds>(chrono::steady_clock::now() - start).count() / 1000.;
}

inline void check_timeout() {
	if (time_since(start_time) > time_limit) {
		cout<<0<<" "<<time_limit<<endl;
		exit(0);
	}
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

template<typename T>
long double bernoulli(Matrix<T> &matrix) {
	uniform_real_distribution<> uniform_dist(0, 1);
	long double u = uniform_dist(gen);
	long double x = deep_sample(matrix);
	if (log(u) < x) return 1;
	return 0;
}

template<typename T>
long double var_bernoulli(Matrix<T> &matrix) {
	uniform_real_distribution<> uniform_dist(0, 1);
	long double b = uniform_dist(gen) < 0.5;
	if (!b) return 0;
	long double u = uniform_dist(gen);
	long double x1 = deep_sample(matrix);
	long double x2 = deep_sample(matrix);
	if (log(u) <= 2 * log_sub_exp(max(x1, x2), min(x1, x2))) return 1;
	return 0;
}

template<typename T>
long double gbas(long long k, Matrix<T> &matrix) {
	long long N = 1;
	long double R = 0;
	long long cnt = 0;
	exponential_distribution<long double> distribution(1);

	while (true) {
		if (bernoulli(matrix)) {
			cnt += 1;
		}
		R += distribution(gen);
		if (cnt == k) break;
		N++;
		if (N % 10 == 0) check_timeout();
	}
	return (k + 2) / R;
}

inline long double psi(long double s) {
	if (s >= 0) return log(1 + s + s * s / 2.);
	else return -log(1 - s + s * s / 2.);
}

template<typename T>
long double ltsa(long long n, long double cc, long double mu0, long double epsilon, long double epsilon0, Matrix<T> &matrix) {
	long double mu = 0;
	long double mu0t = mu0 / (1 - epsilon0 * epsilon0);
	long double alpha = epsilon / (cc * mu0t);
	for (long long i = 0; i < n; i++) {
		long double X = exp(deep_sample(matrix));
		long double W = mu0t + 1. / alpha * psi(alpha * (X - mu0t));
		mu += W / n;
		if (i % 10 == 0) check_timeout();
	}
	return mu;
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

template<typename T>
long double estimate(long double epsilon, long double delta, Matrix<T> &matrix) {
	long long k = (long long)(0.1 + ceil(2 * pow(epsilon, -2./3.) * log(6. / delta)));
	cerr<<"GBAS... "<<k<<endl;
	long double mu0 = gbas(k, matrix);
	cerr<<"Bad estimate: "<<(coef + log(mu0) + ub)<<endl;

	poisson_distribution<long long> poisson(2. * log(3. / delta) / (epsilon * mu0));
	long long N = poisson(gen);
	long double A = 0;
	cerr<<"Bern(Var)... "<<N<<endl;
	for (long long i = 1; i <= N; i++) {
		A += var_bernoulli(matrix);
		if (i % 10 == 0) check_timeout();
	}
	long double c1 = 2 * log(3. / delta);
	long double cc = (A / c1 + 0.5 + sqrt(A / c1 + 0.25)) * (1. + pow(epsilon, 1./3.)) * (1. + pow(epsilon, 1./3.)) * epsilon / mu0;
	long long n = (long long)(0.1 + ceil(2. / epsilon / epsilon * log(6. / delta) * cc / (1. - pow(epsilon, 1./3.))));

	cerr<<"LTSA... "<<n<<endl;

	return ltsa(n, cc, mu0, epsilon, pow(epsilon, 1. / 3.), matrix);
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

void precompute() {
	gen.seed(time(0));
	srand(time(0));

	gamma_precomputed[0] = 0;
	gamma_precomputed[1] = EULER;
	for (int i = 2; i <= n; i++) {
		gamma_precomputed[i] = gamma_precomputed[i - 1] + 1 + .5 / gamma_precomputed[i - 1] + .6 / gamma_precomputed[i - 1] / gamma_precomputed[i - 1];	
	}
	for (int i = 0; i <= n; i++) {
		gamma_precomputed[i] /= EULER;
	}
}

template<typename T>
void precompute_dp(Matrix<T> matrix) {
	ub = 0;
	for (int j = 0; j < n; j++) {
		for (int i = 0; i < n; i++) {
			vector<T> vals;
			for (int k = j + 1; k < n; k++) {
				vals.push_back(matrix[i][k]);
			}
			sort(vals.rbegin(), vals.rend());
			long double sum = 0;
			for (int k = 0; k < n - j - 1; k++) {
				sum += (gamma_precomputed[k + 1] - gamma_precomputed[k]) * vals[k];
			}

			if (matrix[i][j] < EPSILON) weight[i + n * j] = 0;
			else if (sum < EPSILON) weight[i + n * j] = 1 / EPSILON; 
			else weight[i + n * j] = matrix[i][j] / sum;
		}
	}

	for (int i = 0; i < n; i++) {
		row_gammas[i] = 0;
		vector<T> row = matrix[i];
		sort(row.begin() + J, row.end());
		reverse(row.begin() + J, row.end());
		for (int j = J; j < n; j++) {
			row_gammas[i] += row[j] * (gamma_precomputed[j - J + 1] - gamma_precomputed[j - J]);
		}
		row_gammas[i] = log(row_gammas[i]);
	}

	for (int i = 0; i < n; i++) {
		for (int j = 0; j < (1<<J); j++) {
			dp[D_INDEX(i, j)] = -INFINITY;
		}
	}
	long double row_prod = 0;
	bool row_prod_zero = false;
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < (1<<J); j++) {
			if (i) dp[D_INDEX(i, j)] = dp[D_INDEX(i - 1, j)] + row_gammas[i];
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
					if (!isfinite(dp[D_INDEX(i - 1, j ^ (1<<k))])) continue;
					Breal addition;
					addition.set_log(dp[D_INDEX(i - 1, j ^ (1<<k))]);
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
				dp[D_INDEX(i, j)] = log_sum_exp(dp[D_INDEX(i, j)], value.get_log());
			}
		}
		row_prod = row_prod + row_gammas[i];
		if (isnan(row_prod) || isinf(row_prod)) {
			row_prod_zero = true;
		}
	}
	ub = dp[D_INDEX(n - 1, (1<<J) - 1)];
}

template<typename T>
long double deep_sample(Matrix<T> matrix) {
	uniform_real_distribution<> uniform_dist(0, 1);
	int n = matrix[0].size();
	long double upper_bound = ub + EPSILON;
	long double dlog = ub;

	bool used[n];
	for (int j = 0; j < n; j++) {
		used[j] = 0;
	}

	// Exact sampling part O(Jn)
	vector<int> chosen_rows;
	int subset = (1<<J) - 1;
	for (int i = n - 1; i >= 0; i--) {
		if (!subset) {
			dlog -= row_gammas[i];
		} else {
			if (i) {
				if (log(uniform_dist(gen)) > dp[D_INDEX(i - 1, subset)] + row_gammas[i] - dp[D_INDEX(i, subset)]) {
					used[i] = 1;
					int max_j = -1;
					long double max_weight = -INFINITY;
					for (int j = 0; j < J; j++) {
						if ((subset & (1<<j)) && matrix[i][j] > EPSILON) {
							long double weight = -log(-log(uniform_dist(gen))) + log(matrix[i][j]) + dp[D_INDEX(i - 1, subset ^ (1<<j))];
							if (max_j == -1 || weight > max_weight) {
								max_j = j;
								max_weight = weight;
							}
						}
					}
					if (max_j < 0) return -INFINITY;
					subset ^= 1<<max_j;
				} else {
					dlog -= row_gammas[i];
				}
			} else {
				if (subset != (subset & -subset)) { // at most one 1 bit remaining
					cout<<"Exact sampling failed: "<<subset<<endl;
					exit(0);
				}
				used[i] = 1;
			}
		}
	}
	
	if (n - J == 0) return 0;

	for (int j = J; j < n; j++) {
		long double norm = 0;
		for (int i = 0; i < n; i++) {
			if (!used[i]) {
				norm += weight[i + n * j];
			}
		}
		long double choice = uniform_dist(gen) * norm;
		if (norm < EPSILON) return -INFINITY;
		for (int i = 0; i < n; i++) {
			if (used[i]) continue;
			if (choice < weight[i + n * j]) {
				dlog += log(matrix[i][j]) - log(weight[i + n * j] / norm);
				used[i] = 1;
				break;
			} else {
				choice -= weight[i + n * j];
			}
		}
	}

	return dlog - ub;
}


/*
5 100000
1 1 0 0 0 
1 1 1 0 0
0 1 1 1 0
0 0 1 1 1
0 0 0 1 1

Output: log(8) ~ 2.0794415416798357
*/

/*
INPUT FORMAT
============
n time_limit
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
	cin>>n>>time_limit;
	J = min(MAX_J, n);
	row_gammas = new long double[n + 1];
	gamma_precomputed = new long double[n + 1];
	weight = new long double[n * n];
	dp = new long double[(n + 1) * (1<<J)];

	Matrix<long double> matrix;
	for (int i = 0; i < n; i++) {
		matrix.push_back(vector<long double>(n));
		for (int j = 0; j < n; j++) {
			cin>>matrix[i][j];
		}
	}

	start_time = chrono::steady_clock::now();

	cerr<<"Preprocessing..."<<endl;
	
	matrix = tassa(matrix);
	if (matrix.empty()) {
		cout<<"zero permanent"<<endl;
		return 0;
	}

	precompute();
	{
		for (int t = 0; t < 100; t++) {
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
	}

	precompute_dp(matrix);
	
	cerr<<"Sampling..."<<endl;

	cout<<setprecision(8)<<fixed;

	long double est = estimate(epsilon, delta, matrix);
	cout<<(coef + ub + log(est))<<" "<<time_since(start_time)<<endl;

	delete[] row_gammas;
	delete[] gamma_precomputed;
	delete[] weight;
	delete[] dp;
}