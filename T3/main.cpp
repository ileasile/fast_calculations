#include <iostream>
#include <fstream>
#include <mpi.h>
#include <utility>
#include <algorithm>
#include <cstring>
#include <random>
#include <chrono>
#include <cmath>
#include "Timer.h"

const double pi = 3.1415926;
const double c_1_2pi = 1./2./pi;
const double c_m1_2pi = - c_1_2pi;

void f(int n, int k, int sz, double * arg, double * res, double * omega){
	double * x = arg;
	double * y = arg + n;
	double * rx = res;
	double * ry = res + n;

	int cluster_size = n / sz + ((n % sz) != 0);
	int cluster_beg = std::min(n - 1, k * cluster_size);
	int cluster_end = std::min(n, (k + 1) * cluster_size);

	for(int i = 0; i < n; ++i){
		for(int j = cluster_beg; j < cluster_end; ++j){
			if(i == j)
				continue;

			double dx = x[i] - x[j];
			double dy = y[i] - y[j];
			double m = omega[j]/(dx*dx + dy*dy);

			rx[i] += dy * m;
			ry[i] += dx * m;
		}

		rx[i] *= c_m1_2pi;
		ry[i] *= c_1_2pi;
	}

	MPI_Allreduce(MPI_IN_PLACE, res, 2*n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
}

void rk(int n, int k, int sz,
		int n_iter, double h,
		double * p, double * w, double ** res_p){
	double h_2 = h * 0.5;
	double h_6 = h / 6.;
	int n2 = n * 2;
	size_t vsz = n2 * sizeof(double);

	*res_p = new double[n2 * n_iter];
	double * r = *res_p;
	double * k1 = new double[n2 * 4];
	double * k2 = k1 + n2;
	double * k3 = k2 + n2;
	double * k4 = k3 + n2;
	double * temp = new double[n2];

	memset(r, 0, vsz * n_iter);
	memcpy(r, p, vsz);

	for(int i = 1; i < n_iter; ++i){
		memset(k1, 0, vsz * 4);

		double * r_cur = r + i * n2;
		double * r_prev = r + (i - 1) * n2;

		f(n, k, sz, r_prev, k1, w);

		for(int j = 0; j < n2; ++j)
			temp[j] = r_prev[j] + h_2 * k1[j];
		f(n, k, sz, temp, k2, w);

		for(int j = 0; j < n2; ++j)
			temp[j] = r_prev[j] + h_2 * k2[j];
		f(n, k, sz, temp, k3, w);

		for(int j = 0; j < n2; ++j)
			temp[j] = r_prev[j] + h * k3[j];
		f(n, k, sz, temp, k4, w);

		for(int j = 0; j < n2; ++j)
			r_cur[j] = r_prev[j] + h_6 * (k1[j] + k4[j] + 2 * (k2[j] + k3[j]));
	}

	delete[] k1;
	delete[] temp;
}

void gen(int n, double * p, double * w, bool inside_square = true){
	// Generate inside a circle or square

	double r = 2;
	double r2 = r * r;

	typedef std::chrono::high_resolution_clock  hr_clock;
	hr_clock::time_point now = hr_clock::now();
	auto now_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(now);
	auto value = now_ms.time_since_epoch();
	long long seed = value.count();
	std::mt19937_64 gen(seed);
	std::uniform_real_distribution<double> dist(-r, r);
	// std::normal_distribution<double> dist2(0., 1.);
	std::uniform_real_distribution<double> dist2(-2.5, 2.5);

	int i = 0;
	while(i < n){
		double x = dist(gen);
		double y = dist(gen);
		if(inside_square || x * x + y * y < r2 ){
			p[i] = x;
			p[i + n] = y;
			w[i] = dist2(gen);
			++i;
		}
	}
}

void gen2(int n, double * p, double * w){
	// Generate on a circle
	double r = 2;

	typedef std::chrono::high_resolution_clock  hr_clock;
	hr_clock::time_point now = hr_clock::now();
	auto now_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(now);
	auto value = now_ms.time_since_epoch();
	long long seed = value.count();
	std::mt19937_64 gen(seed);
	std::uniform_real_distribution<double> dist(-pi, pi);

	int i = 0;
	while(i < n){
		double phi = dist(gen);
		double x = r * cos(phi);
		double y = r * sin(phi);
		p[i] = x;
		p[i + n] = y;
		w[i] = 1;
		++i;
	}
}

void gen3(int n, double * p, double * w){
	// Generate on a circle
	double r = 2;

	for(int i = 0; i < n; ++i){
		double phi = i * 2 * pi / n;
		double x = r * cos(phi);
		double y = r * sin(phi);
		p[i] = x;
		p[i + n] = y;
		w[i] = 1;
	}
}

void gen4(int n, double * p, double * w){
	// Generate on a circle
	double r = 2;

	for(int i = 0; i < n; ++i){
		double phi = i * 2 * pi / n;
		double rd = r * (1 - sin(phi));
		double x = rd * cos(phi);
		double y = rd * sin(phi);
		p[i] = x;
		p[i + n] = y;
		w[i] = 1;
	}
}

void go(std::ostream & out, int rank, int size, int n, int n_iter, double h){
	Timer tt;
	double *p, *w, *r;
	p = new double[2 * n];
	w = new double[n];
	if(rank == 0)
		gen4(n, p, w);
	MPI_Bcast(p, 2 * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(w, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	rk(n, rank, size, n_iter, h, p, w, &r);
	auto tm = tt.stop_reset();

	if(rank == 0) {
		out << (double) tm * 1e-6 << std::endl;
		out << n << " " << n_iter << " " << h << "\n";
		for (int i = 0; i < n; ++i) {
			out << w[i] << " ";
		}
		out << "\n";
		for (int i = 0; i < n_iter; ++i) {
			for (int j = 0; j < n; ++j)
				out << r[i * 2 * n + j] << " ";
			out << "\n";
			for (int j = n; j < 2 * n; ++j)
				out << r[i * 2 * n + j] << " ";
			out << "\n";
		}
	}
}

int main(int argc, char ** argv) {
	MPI_Init(NULL, NULL);
	int size, rank;
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

#ifdef ILEASILE
	std::ofstream out;
	if(rank == 0)
		//out.open("out_"+std::to_string(rank)+".txt");
		out.open("out.txt");
#else
	std::ostream & out = std::cout;
#endif

	go(out, rank, size,
	   std::stoi(argv[1]),
	   std::stoi(argv[2]),
	   std::stod(argv[3]));

#ifdef ILEASILE
	if(rank == 0)
		out.close();
#endif

	MPI_Finalize();
	return 0;
}