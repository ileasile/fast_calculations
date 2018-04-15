#include <iostream>
#include <fstream>
#include <ctime>
#include <cmath>
#include <tuple>
#include <vector>
#include <algorithm>
#include <sys/time.h>
#include <omp.h>
#include <chrono>
#include <random>
#include <mpi.h>

class Timer{
	struct timeval tv;
	unsigned long long tm;
public:
	Timer(){
		gettimeofday(&tv, NULL);
		tm = (unsigned long long)tv.tv_sec * 1000000 + tv.tv_usec;
	}

	unsigned long long stop_reset(){
		gettimeofday(&tv, NULL);
		auto tm_ = (unsigned long long)tv.tv_sec * 1000000 + tv.tv_usec;
		auto res = tm_ - tm;
		tm = tm_;
		return res;
	}
};

double cube_v(double R, int dim){
	double d = 2*R;
	double v = 1;
	for(int i=0; i<dim; ++i){
		v *= d;
	}
	return v;
}

double vr(int n, double R){
	if(n == 0){
		return 1;
	}
	else if(n == 1){
		return 2;
	}
	else{
		return vr(n - 2, R)*2*M_PI*R*R/n;
	}
}

double do_(int dim, int n, int size, int k, double R, MPI_Comm & comm) {
	Timer tm;

	int n_per_p = n/size;
	int overall = n_per_p * size;
	double r2 = R*R;
	int hits = 0;

	typedef std::chrono::high_resolution_clock  hrclock;
	hrclock::time_point now = hrclock::now();
	auto now_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(now);
	auto value = now_ms.time_since_epoch();
	long long seed = value.count();

	int h = 0;

	long long sd = seed + k * 10;
	std::mt19937_64 gen(sd);
	double rnd =  (2. * R) / (1. + gen.max()) ;

	for(int i=0; i < n_per_p; ++i){
		double s = 0, p = 0;
		for(int j=0; j < dim; ++j){
			p = gen()*rnd - R;
			s += p*p;
		}
		if(s < r2)
			++h;
	}

	MPI_Allreduce(&h, &hits, 1, MPI_INT, MPI_SUM, comm);

	double t_val = tm.stop_reset() * 1e-6;

	if(k == 0){
		double v = cube_v(R, dim) * hits/overall;
		double v_real = vr(dim, R);

		std::cout
				<< "Threads number: " << size
				<< ", calculated volume:" << v
				<< ", real volume:" << v_real
				<< ", n = " << n
				<< ", dim = " << dim
				<< ", time(sec) = " << t_val << "\n";
	}

	return t_val;
}

int main(int argn, char ** argv) {
	MPI_Init(NULL, NULL);
	int size, rank;
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	std::vector<int> sizes = {(int)1e7,
							  (int)2e7,
							  (int)5e7,
							  (int)7e7,
							  (int)1e8,
							  (int)1.3e8,
							  (int)1.6e8};

	int dim = 9;

	std::vector<int> processors;
	for(int i = 1; i < argn; ++i){
		processors.push_back(atoi(argv[i]));
	}

	std::ofstream csv;
	std::string delim = ",";

	if(rank == 0){
		csv.open("out_d_"+std::to_string(dim)+".csv");
		csv << "N / processors";
		for(int p: processors)
			csv << delim << p ;
		csv << std::endl;
	}

	for(int s: sizes){
		if(rank == 0)
			csv << s;
		for(int p: processors){
			MPI_Comm comm;
			MPI_Comm_split(MPI_COMM_WORLD, (rank < p)? 0 : 1, rank, &comm);

			if(rank < p){
				double sec_t = do_(dim, s, p, rank, 1, comm);
				double secmax;
				MPI_Reduce(&sec_t, &secmax, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
				if(rank == 0)
					csv << delim << secmax ;
			}

			MPI_Comm_free(&comm);
			MPI_Barrier(MPI_COMM_WORLD);

		}
		if(rank == 0)
			csv << std::endl;
	}

	MPI_Finalize();
	return 0;
}
