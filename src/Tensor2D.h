#pragma once
#include "Tensor1D.h"
#include "Log.h"

class Tensor2D 
{
	friend class Tensor3D;

  private:
	int _rows;
	int _cols;
	float ** _data;

 public:
	Tensor2D(int rows, int cols);
	Tensor2D(const Tensor2D& t);
	~Tensor2D();

	Tensor2D operator+(const Tensor2D& t);
	Tensor2D operator-(const Tensor2D& t);

	Tensor2D operator*(const Tensor2D& t);
	Tensor1D operator*(const Tensor1D& t);

	Tensor2D& operator+=(const Tensor2D& t);
	Tensor2D& operator-=(const Tensor2D& t);

	Tensor2D& operator=(const Tensor2D& t);

	int rows() const;
	int cols() const;

	void setValue(float value);
	float& operator()(int i, int j);
	
	Tensor1D row(const int n);
	Tensor1D col(const int n);

	Tensor2D convolve(const Tensor2D& kernel);

	Tensor1D flat();

	friend Tensor2D operator*(const Tensor2D& t, float lambda);
	friend Tensor2D operator*(float lambda, const Tensor2D& t);

	friend Tensor1D operator*(Tensor2D& lhs, Tensor1D& rhs);
	friend Tensor1D operator*(Tensor1D& lhs, Tensor2D& rhs);

	friend std::ostream& operator<<(std::ostream& os, const Tensor2D& t);
};

Tensor2D outer(Tensor1D& lhs, Tensor1D& rhs);  	

