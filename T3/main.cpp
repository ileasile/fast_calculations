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

const double G = 10;

void f(int n, int k, int sz, double * arg, double * res, double * omega){
	double * x = arg;
	double * y = x + n;
	double * vx = y + n;
	double * vy = vx + n;

	double * rx = res;
	double * ry = res + n;
	double * rvx = ry + n;
	double * rvy = rvx + n;

	int cluster_size = n / sz + ((n % sz) != 0);
	int cluster_beg = std::min(n - 1, k * cluster_size);
	int cluster_end = std::min(n, (k + 1) * cluster_size);

	for(int i = 0; i < n; ++i){
		rx[i] = vx[i];
		ry[i] = vy[i];

		for(int j = cluster_beg; j < cluster_end; ++j){
			if(i == j)
				continue;

			double dx = x[j] - x[i];
			double dy = y[j] - y[i];
			double r_2 = (dx*dx + dy*dy);
			double r_1 = sqrt(r_2);

			double m = G * omega[j]/(r_2 * r_1);

			rvx[i] += dx * m;
			rvy[i] += dy * m;
		}
	}

	MPI_Allreduce(MPI_IN_PLACE, res, 4*n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
}

void rk(int n, int k, int sz,
		int n_iter, double h,
		double * p, double * w, double ** res_p){
	double h_2 = h * 0.5;
	double h_6 = h / 6.;
	int n2 = n * 4;
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

void gen(int n, double * p, double * w){
	double * x = p;
	double * y = x + n;
	double * vx = y + n;
	double * vy = vx + n;

	w[0] = 100000.;
	w[1] = 1.;

	x[0] = 0;
	y[0] = 0;
	vx[0] = 0;
	vy[0] = 0;

	x[1] = 0;
	y[1] = 5;
	vx[1] = 300;
	vy[1] = 0;
}

void go(std::ostream & out, int rank, int size, int n, int n_iter, double h){
	Timer tt;
	double *p, *w, *r;
	p = new double[4 * n];
	w = new double[n];
	if(rank == 0)
		gen(n, p, w);
	MPI_Bcast(p, 4 * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
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
				out << r[i * 4 * n + j] << " ";
			out << "\n";
			for (int j = n; j < 2 * n; ++j)
				out << r[i * 4 * n + j] << " ";
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