#ifndef T3_TIMER_H
#define T3_TIMER_H

#include <sys/time.h>
#include <cstdlib>

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

#endif //T3_TIMER_H
