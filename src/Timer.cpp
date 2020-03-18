#include "Timer.h"

Timer::Timer() {
	gettimeofday(&_tic, 0);
}

void Timer::reset() {
	gettimeofday(&_tic, 0);
}

float Timer::getTimeMs() {
	gettimeofday(&_tac, 0);
	int sec  = _tac.tv_sec  - _tic.tv_sec;
	int usec = _tac.tv_usec - _tic.tv_usec;
	float t = 1.0e3 * sec + 1.0e-3 * usec;
	return t;
}
  	

