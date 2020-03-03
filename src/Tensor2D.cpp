#include "Log.h"
#include "Tensor1D.h"
#include "Tensor2D.h"

Tensor2D::Tensor2D(int rows, int cols)
  : _rows(rows), _cols(cols)
{
	_data = new float*[rows];
	for (int i=0; i<rows; i++)
		_data[i] = new float[cols];

//	delete [] _data;

}

int Tensor2D::rows() {
	return _rows;
}

int Tensor2D::cols() {
	return _cols;
}

void Tensor2D::set(float value) {
	for (int i=0; i<_rows; i++)
		for (int j=0; j<_cols; j++)
			_data[i][j] = value;
}

float& Tensor2D::operator()(int i, int j)
{
	return _data[i][j];
}

Tensor2D::~Tensor2D() {
	for (int i=0; i<_rows; i++)
		delete [] _data[i];
	delete [] _data;
}

