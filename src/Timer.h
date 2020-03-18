#pragma once
#include <sys/time.h>

class Timer 
{
  private: 
	struct timeval _tic;
	struct timeval _tac;
  public:
	Timer();
	void reset();
	float getTimeMs();
};
