#include "Log.h"
#include "Tensor3D.h"

Tensor3D::Tensor3D(int depth, int rows, int cols)
  : _depth(depth), _rows(rows), _cols(cols)
{
	_data = new float**[_depth];

	for (int d = 0; d < _depth; d++)
		_data[d] = new float*[_rows];

	for (int d = 0; d < _depth; d++)
		for (int i = 0; i < _rows; i++)
			_data[d][i] = new float[_cols]; 

}

Tensor3D::Tensor3D(const Tensor3D& t)
  : _depth(t._depth), _rows(t._rows), _cols(t._cols)
{
	_data = new float**[_depth];

	for (int d = 0; d < _depth; d++)
		_data[d] = new float*[_rows];

	for (int d = 0; d < _depth; d++)
		for (int i = 0; i < _rows; i++)
			_data[d][i] = new float[_cols]; 

	for (int d = 0; d <_depth; d++)
		for (int i = 0; i < _rows ; i++)
			for (int j = 0; j < _cols; j++)
				_data[d][i][j] = t._data[d][i][j]; 
}

Tensor3D::~Tensor3D() 
{
	for (int d = 0; d < _depth; d++)
		for (int i = 0; i < _rows; i++)
			delete [] _data[d][i]; 

	for (int d = 0; d < _depth; d++)
		delete [] _data[d];
		
	delete [] _data;
}

Tensor3D Tensor3D::operator+(const Tensor3D& t)
{
	Tensor3D r(_depth, _rows, _cols);
	if (_depth == t._depth && _rows == t._rows && _cols == t._cols) {
		for (int d = 0; d < _depth; d++)
			for (int i = 0; i < _rows; i++)
				for (int j = 0; j < _cols; j++)
					r._data[d][i][j] = _data[d][i][j] + t._data[d][i][j];
	}
	return r;
}

Tensor3D Tensor3D::operator-(const Tensor3D& t)
{
	Tensor3D r(_depth, _rows, _cols);
	if (_depth == t._depth && _rows == t._rows && _cols == t._cols) {
		for (int d = 0; d < _depth; d++)
			for (int i = 0; i < _rows; i++)
				for (int j = 0; j < _cols; j++)
					r._data[d][i][j] = _data[d][i][j] - t._data[d][i][j];
	}
	return r;
}

Tensor3D& Tensor3D::operator+=(const Tensor3D& t)
{
	if (_depth == t._depth && _rows == t._rows && _cols == t._cols) {
		for (int d = 0; d < _depth; d++)
			for (int i = 0; i < _rows; i++)
				for (int j = 0; j < _cols; j++)
					_data[d][i][j] += t._data[d][i][j];
	}
	return *this;
}

Tensor3D& Tensor3D::operator-=(const Tensor3D& t)
{
	if (_depth == t._depth && _rows == t._rows && _cols == t._cols) {
		for (int d = 0; d < _depth; d++)
			for (int i = 0; i < _rows; i++)
				for (int j = 0; j < _cols; j++)
					_data[d][i][j] -= t._data[d][i][j];
	}
	return *this;
}


Tensor3D& Tensor3D::operator=(const Tensor3D& t)
{
	if (this == &t) 
		return *this;

	// Delete data
	for (int d = 0; d < _depth; d++)
		for (int i = 0; i < _rows; i++)
			delete [] _data[d][i]; 

	for (int d = 0; d < _depth; d++)
		delete [] _data[d];
		
	delete [] _data;

	// Copy attributes
	_depth = t._depth;
	_rows  = t._rows;
	_cols  = t._cols;

	// Allocate memory
	_data = new float**[_depth];

	for (int d = 0; d < _depth; d++)
		_data[d] = new float*[_rows];

	for (int d = 0; d < _depth; d++)
		for (int i = 0; i < _rows; i++)
			_data[d][i] = new float[_cols]; 

	// Copy data
	for (int d = 0; d <_depth; d++)
		for (int i = 0; i < _rows ; i++)
			for (int j = 0; j < _cols; j++)
				_data[d][i][j] = t._data[d][i][j];

	// Return
	return *this;
}

int Tensor3D::depth() const {
	return _depth;
}

int Tensor3D::rows() const {
	return _rows;
}

int Tensor3D::cols() const {
	return _cols;
}

void Tensor3D::setValue(float value) {
	for (int d = 0; d < _depth; d++)
		for (int i = 0; i < _rows; i++)
			for (int j = 0; j < _cols; j++)
				_data[d][i][j] = value;
}

float& Tensor3D::operator()(int d, int i, int j) {
	return _data[d][i][j];
}

Tensor2D Tensor3D::slice(const int n) {
	Tensor2D c(_rows, _cols);	
	for (int i = 0; i < _rows; i++)
		for (int j = 0; j < _cols; j++)
			c._data[i][j] = _data[n][i][j];
	return c; 
}

Tensor1D Tensor3D::flat() {
	Tensor1D t(_depth * _rows * _cols);
	int idx = 0;
	for (int i = 0; i < _depth; i++)
		for (int j = 0; j < _rows; j++)
			for (int k = 0; k < _cols; k++) {
				t._data[idx] = _data[i][j][k];
				idx++;
			}
	return t;
}

std::ostream& operator<<(std::ostream& os, const Tensor3D& t)
{
	for (int d = 0; d < t._depth; d++) {
		for (int i = 0; i < t._rows; i++) {
			for (int j = 0; j < t._cols; j++) {
				os << std::setw(OSTREAM_WIDTH);
				os << t._data[d][i][j] << ' ';
			}
			os << '\n';
		}
		for (int k = 0; k < ((OSTREAM_WIDTH+1) * t._cols)-1; k++)
			os << '-';
		os << '\n';
	}
	return os;
}

Tensor3D Tensor3D::convolve(const Tensor3D& kernel) {
	Tensor3D r(_depth - kernel._depth + 1,
	           _rows  - kernel._rows  + 1,
	           _cols  - kernel._cols  + 1);
	for (int i = 0; i < r._depth; i++)
		for (int j = 0; j < r._rows; j++)
			for (int k = 0; k < r._rows; j++)
				for (int di = 0; di < kernel._cols; di++)
					for (int dj = 0; dj < kernel._rows; dj++)
						for (int dk = 0; dj < kernel._rows; dj++)
							r._data[i][j][k] = 
								_data[i + di][j + dj][k + dk] * kernel._data[di][dj][dk];
	return r;		
}

Tensor3D operator*(const Tensor3D& t, float lambda) {
	Tensor3D r(t._depth, t._rows, t._cols);
	for (int i = 0; i < t._depth; i++)
		for (int j = 0; j < t._rows; j++)
			for (int k = 0; k < t._cols; k++)
				r._data[i][j][k] = lambda * t._data[i][j][k];
	return r;
}

Tensor3D operator*(float lambda, const Tensor3D& t) {
	Tensor3D r(t._depth, t._rows, t._cols);
	for (int i = 0; i < t._depth; i++)
		for (int j = 0; j < t._rows; j++)
			for (int k = 0; k < t._cols; k++)
				r._data[i][j][k] = lambda * t._data[i][j][k];
	return r;
}

//	Tensor3D conv(Tensor3D& t, Tensor3D& k);

