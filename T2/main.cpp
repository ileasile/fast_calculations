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

double do_(int dim, int n, int k, double R) {

	Timer tm;

	int n_per_p = n/k;
	int overall = n_per_p * k;
	double r2 = R*R;
	int hits = 0;

	typedef std::chrono::high_resolution_clock  hrclock;
	hrclock::time_point now = hrclock::now();
	auto now_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(now);

	auto value = now_ms.time_since_epoch();
	long long seed = value.count();

	#pragma omp parallel num_threads(k)
	{
		//int T = omp_get_num_threads();
		int nT = omp_get_thread_num();
		int h = 0;

		long long sd = seed + nT * 10;
		//std::mt19937 gen(sd);
		//std::minstd_rand gen(sd);
		std::mt19937_64 gen(sd);
		double rnd =  (2. * R) / (1. + gen.max()) ;

		for(int i=0; i < n_per_p; ++i){
			double s = 0, p = 0;
			for(int j=0; j < dim; ++j){
				//p = rand()/rnd - R;
				p = gen()*rnd - R;
				s += p*p;
			}
			if(s < r2)
				++h;
		}
		//std::cout << "\nproc " << nT;
		//std::cout << nT << "\n";
		#pragma omp critical
		{
			hits += h;
	    }
	}

	double v = cube_v(R, dim) * hits/overall;

	double t_val = tm.stop_reset() * 1e-6;

	double v_real = vr(dim, R);

	std::cout
			<< "Threads number: " << k
			<< ", calculated volume:" << v
			<< ", real volume:" << v_real
			<< ", n = " << n
			<< ", dim = " << dim
			<< ", time(sec) = " << t_val << "\n";

	return t_val;
}

int main(int argn, char ** argv) {
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

	std::ofstream csv("out_d_"+std::to_string(dim)+".csv");
	std::string delim = ",";

	csv << "N / processors";
	for(int p: processors)
		csv << delim << p ;
	csv << std::endl;

	for(int s: sizes){
		csv << s;
		for(int p: processors){
			double sec_t = do_(dim, s, p, 1);
			csv << delim << sec_t ;
		}
		csv << std::endl;
	}

	return 0;
}
