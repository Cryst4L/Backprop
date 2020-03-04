#include "Log.h"
#include "Tensor1D.h"
#include "Tensor2D.h"


Tensor2D::Tensor2D(int rows, int cols)
  : _rows(rows), _cols(cols) 
{
	_data = new float*[rows];
	for (int i = 0; i < rows; i++)
		_data[i] = new float[cols];
}

Tensor2D::Tensor2D(const Tensor2D& t) 
	: _rows(t._rows), _cols(t._cols) 
{
	_data = new float*[_rows];
	for (int i = 0; i < _rows; i++)
		_data[i] = new float[_cols];

	for (int i = 0; i < _rows; i++)
		for (int j = 0; j < _cols; j++)
			_data[i][j] = t._data[i][j];
}

Tensor2D::~Tensor2D() {
	for (int i = 0; i < _rows; i++)
		delete [] _data[i];
	delete [] _data;
}

Tensor2D Tensor2D::operator+(const Tensor2D& t) {
	Tensor2D r(_rows,_cols);
	if (_rows == t._rows && _cols == t._cols) {
		for (int i = 0; i < _rows; i++)
			for (int j = 0; j < _cols; j++)
				r._data[i][j] = _data[i][j] + t._data[i][j];
	} else {
		std::cerr << " Tensor size mismatch !" << std::endl;
	}
	return r;
}

Tensor2D Tensor2D::operator-(const Tensor2D& t) {
	Tensor2D r(_rows,_cols);
	if (_rows == t._rows && _cols == t._cols) {
		for (int i = 0; i < _rows; i++)
			for (int j = 0; j < _cols; j++)
				r._data[i][j] = _data[i][j] - t._data[i][j];
	} else {
		std::cerr << " Tensor size mismatch !" << std::endl;
	}
	return r;
}

Tensor2D Tensor2D::operator*(const Tensor2D& t) {
	Tensor2D r(_rows,t._cols);
	if (_cols == t._rows) {
		for (int i = 0; i < _rows; i++)
			for (int j = 0; j < t._cols; j++) {
				float acc = 0;
				for (int k = 0; k < _cols; k++)
					acc += _data[i][k] * t._data[k][j];
				r._data[i][j] = acc;
			}
	} else {
		std::cerr << " Tensor size mismatch !";
	}
	return r;
}
	
Tensor2D& Tensor2D::operator=(const Tensor2D& t) {

	if (this == &t) return *this;

	if (_data != 0) delete [] _data;

	_rows = t._rows;
	_cols = t._cols;
 
	_data = new float*[_rows];
	for (int i = 0; i < _rows; i++)
		_data[i] = new float[_cols];

	for (int i = 0; i < _rows; i++)
		for (int j = 0; j < _cols; j++)
			_data[i][j] = t._data[i][j];

	return *this;
}


int Tensor2D::rows() {
	return _rows;
}

int Tensor2D::cols() {
	return _cols;
}

void Tensor2D::setValue(float value) {
	for (int i = 0; i < _rows; i++)
		for (int j = 0; j < _cols; j++)
			_data[i][j] = value;
}

float& Tensor2D::operator()(int i, int j) {
	return _data[i][j];
}

Tensor2D outer(Tensor1D& lhs, Tensor1D& rhs)
{
	Tensor2D r = Tensor2D(lhs.size(), rhs.size());
	for (int i = 0; i < lhs.size(); i++)
		for (int j = 0; j < rhs.size(); j++)
			r(i,j) = lhs(i) * rhs(j);
	return r; 	
}

