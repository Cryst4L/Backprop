#pragma once
#include "Log.h"
#include "Tensor2D.h"

class Tensor3D 
{
  private:
	int _depth;
	int _rows;
	int _cols;
	float *** _data;

  public:
	Tensor3D(int depth, int rows, int cols);
	Tensor3D(const Tensor3D& t);
	~Tensor3D();

	Tensor3D& operator=(const Tensor3D& t);

	Tensor3D operator+(const Tensor3D& t);
	Tensor3D operator-(const Tensor3D& t);

	Tensor3D& operator+=(const Tensor3D& t);
	Tensor3D& operator-=(const Tensor3D& t);

	int depth() const;	
	int rows() const;
	int cols() const;

	void setValue(float value);
	float& operator()(int d, int i, int j);
	
	Tensor2D slice(const int n);

	Tensor1D flat();

	Tensor3D convolve(const Tensor3D& kernel);

	friend Tensor3D operator*(const Tensor3D& t, float lambda);
	friend Tensor3D operator*(float lambda, const Tensor3D& t); 

	friend std::ostream& operator<<(std::ostream& os, const Tensor3D& t);	
};


