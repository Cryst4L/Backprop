#pragma once
#include "Log.h"

class Tensor1D 
{
	friend class Tensor2D;
	friend class Tensor3D;

  private:
	int _size;
	float * _data;

  public:
	Tensor1D(int size);
	Tensor1D(const Tensor1D& t);
	~Tensor1D();

	Tensor1D operator+(const Tensor1D& t);
	Tensor1D operator-(const Tensor1D& t);

	Tensor1D& operator+=(const Tensor1D& t);
	Tensor1D& operator-=(const Tensor1D& t);

	float& operator()(int n);
	Tensor1D& operator=(const Tensor1D& t);

	int size() const;
	void setValue(float v);

	Tensor1D convolve(const Tensor1D& kernel);

	friend Tensor1D operator*(const Tensor1D& t, float lambda);
	friend Tensor1D operator*(float lambda, const Tensor1D& t);

	friend float dot(Tensor1D& lhs, Tensor1D& rhs);
//	friend Tensor2D outer(Tensor1D& lhs, Tensor1D& rhs);

	friend std::ostream& operator<<(std::ostream& os, const Tensor1D& t);
};



