#include <iostream>
#include <fstream>
#include <ctime>
#include <cmath>
#include <tuple>
#include <vector>
#include <algorithm>
#include <sys/time.h>
#include <omp.h>

void generate(int * a, int n){
	for(int * c = a; c != (a + n); ++c){
		 *c = rand();
	}
}

void sort(int * a, int * aend){
	std::sort(a, aend);
}

bool comp(std::tuple<int, int *, int> a, std::tuple<int, int *, int> b){
	return std::get<0>(a) > std::get<0>(b);
}

void merge(int ** a, int ** p, int ** pend, int k, int n){
	int * b = new int[n];
	int * bp = b;
	std::tuple<int, int *, int> * hp = new std::tuple<int, int *, int> [k];
	for(int i = 0; i < k; ++i){
		hp[i] = std::make_tuple(*(p[i]), p[i], i);
	}
	std::make_heap(hp, hp + k, comp);
	int sz = k;
	while (sz > 0){
		std::pop_heap(hp, hp + sz, comp);
		int val;
		int * ptr;
		int idx;
		std::tie(val, ptr, idx) = hp[sz-1];
		*(bp ++) = val;
		++ptr;
		if(ptr != pend[idx]){
			hp[sz-1] = std::make_tuple(*ptr, ptr, idx);
			std::push_heap(hp, hp + sz, comp);
		}
		else{
			--sz;
		}
	}

	delete[] (*a);
	delete[] hp;
	*a = b;
}

void merge_s(int ** a, int ** p, int ** pend, int k, int n){
	int * b = new int[n];
	int * bp = b;

	while (1){
		int j = -1;
		for(int i = 0; i < k; ++i){
			if(p[i] != pend[i] && (j == -1 || *(p[i]) < *(p[j])) ){
				j = i;
			}
		}
		if(j == -1)
			break;

		*(bp ++) = *(p[j] ++);
	}

	delete[] (*a);
	*a = b;
}

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

bool verify(int * a, int n){
	for(int i=0; i < n-1; ++i){
		if(a[i] > a[i+1])
			return false;
	}
	return true;
}

void print_array(int * a, int n){
	for(int i = 0; i < n; ++i){
		std::cout << a[i] << " ";
	}
	std::cout << std::endl;
}

double do_(int n, int k) {
	int * a = new int[n];
	int ** p = new int * [k];
	int ** pend = new int * [k];

	srand((unsigned)time(0));
	generate(a, n);

	Timer tm;

	#pragma omp parallel num_threads(k)
	{
		int T = omp_get_num_threads();
		int nT = omp_get_thread_num();
		//std::cout << nT << "\n";
		int h = n / T + 1;

		int * beg = a + std::min(n, nT * h);
		int * end = a + std::min(n, (nT + 1) * h);
		p[nT] = beg;
		pend[nT] = end;

		sort(beg, end);
	}

	merge_s(&a, p, pend, k, n);

	double t_val = tm.stop_reset() * 1e-6;

	if (verify(a, n)){
		std::cout << "Sorting OK, ";
	}
	else{
		std::cout << "Sorting failed, ";
	}

	//print_array(a, 20);
	delete[] a;
	delete[] p;
	delete[] pend;
	std::cout
			<< "Threads number: " << k
			<< ", n = " << n
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

	std::vector<int> processors;
	for(int i = 1; i < argn; ++i){
		processors.push_back(atoi(argv[i]));
	}

	std::ofstream csv("out.csv");
	std::string delim = ",";

	csv << "N / processors";
	for(int p: processors)
		csv << delim << p ;
	csv << std::endl;

	for(int s: sizes){
		csv << s;
		for(int p: processors){
			double sec_t = do_(s, p);
			csv << delim << sec_t ;
		}
		csv << std::endl;
	}

	return 0;
}
