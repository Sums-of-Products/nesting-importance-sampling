// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          https://www.boost.org/LICENSE_1_0.txt)

// Code for evaluating the permanent from https://github.com/hjuha/deepar/blob/main/DeepAR/deepar.cpp

#include <bits/stdc++.h>

using namespace std;

template<typename T>
using Matrix = vector<vector<T>>;
int n;

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
	long double time_limit;
	cin>>n>>time_limit;
	Matrix<long double> matrix;
	for (int i = 0; i < n; i++) {
		matrix.push_back(vector<long double>(n));
		for (int j = 0; j < n; j++) {
			cin>>matrix[i][j];
		}
	}
	long double out = 0;
	for (int i = 0; i < n; i += 5) {
		int l = min(5, n - i);
		Matrix<long double> sub(l);
		for (int ii = i; ii < i + l; ii++) {
			for (int j = i; j < i + l; j++) {
				sub[ii - i].push_back(matrix[ii][j]);
			}
		}
		out += log(permanent(sub));
	}
	cout<<setprecision(10)<<fixed<<out<<endl;
}