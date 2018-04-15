#include <iostream>
#include <fstream>
#include <ctime>
#include <cmath>
#include <tuple>
#include <vector>
#include <algorithm>
#include <sys/time.h>
#include <mpi.h>
#include "Timer.h"

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

double do_(int n, MPI_Comm & comm, int rank, int size) {
	int * a;

	Timer tm;
	int h = n / size + 1;
	MPI_Status st;

	if(rank == 0) {
		a = new int[n];
		int ** p = new int * [size];
		int ** pend = new int * [size];

		srand((unsigned)time(0));
		generate(a, n);

		int * counts = new int[size];
		int * displs = new int[size];
		for(int i = 0; i < size; ++i){
			displs[i] = std::min(n, i * h);
			counts[i] = std::min(n, (i + 1) * h) - displs[i];

			p[i] = a + displs[i];
			pend[i] = p[i] + counts[i];
			if(i != 0)
				MPI_Send(a + displs[i], counts[i], MPI_INT, i, 0, comm);
		}

		sort(p[0], pend[0]);

		for(int i = 1; i < size; ++i){
			MPI_Recv(a + displs[i], counts[i], MPI_INT, i, 0, comm, &st);
		}

		merge_s(&a, p, pend, size, n);

		delete[] p;
		delete[] pend;
	}
	else{
		auto cnt = std::min(n, (rank + 1) * h) - std::min(n, rank * h);
		a = new int[cnt];


		MPI_Recv(a, cnt, MPI_INT, 0, 0, comm, &st);
		sort(a, a+cnt);
		MPI_Send(a, cnt, MPI_INT, 0, 0, comm);
	}

	// std::cout << "\nsdsdsds" << rank;

	double t_val = tm.stop_reset() * 1e-6;

	if(rank == 0) {
		if (verify(a, n)) {
			std::cout << "Sorting OK, ";
		} else {
			std::cout << "Sorting failed, ";
		}
	}

	delete[] a;
	if(rank == 0) {
		std::cout
				<< "Threads number: " << size
				<< ", n = " << n
				<< ", time(sec) = " << t_val << "\n";
	}

	MPI_Barrier(comm);
	return t_val;
}

int main(int argn, char ** argv) {
	MPI_Init(NULL, NULL);
	int size, rank;
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	std::vector<int> sizes = {
			(int)1e7,
			(int)2e7,
			(int)5e7,
			(int)7e7,
			(int)1e8,
			(int)1.3e8,
			(int)1.6e8
	};

	std::vector<int> processors;
	for(int i = 1; i < argn; ++i){
		processors.push_back(atoi(argv[i]));
	}

	std::string delim = ",";
	std::ofstream csv;

	if(rank == 0) {
		csv.open("out.csv");
		csv << "N / processors";
		for (int p: processors)
			csv << delim << p;
		csv << std::endl;
	}

	for(int s: sizes){
		if(rank == 0)
			csv << s;
		for(int p: processors){
			MPI_Comm comm;
			MPI_Comm_split(MPI_COMM_WORLD, (rank < p)? 0 : 1, rank, &comm);

			double secmax = 0;
			if(rank < p) {
				double sec_t = do_(s, comm, rank, p);
				MPI_Reduce(&sec_t, &secmax, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
			}
			MPI_Barrier(MPI_COMM_WORLD);

			MPI_Comm_free(&comm);

			if(rank == 0)
				csv << delim << secmax ;
		}
		if(rank == 0)
			csv << std::endl;
	}

	MPI_Finalize();
	return 0;
}
