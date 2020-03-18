#include "Log.h"
#include "Tensor1D.h"

Tensor1D::Tensor1D(int size) 
  : _size(size) 
{
	_data = new float[size];
	for (int i = 0; i < _size; i++)
		_data[i] = 0;

}

Tensor1D::Tensor1D(const Tensor1D& t)
  : _size(t._size)
{
	_data = new float[_size];
	for (int i = 0; i < _size; i++)	
		_data[i] = t._data[i];
}

Tensor1D Tensor1D::operator+(const Tensor1D& t) {
	Tensor1D r(_size);
	if (_size == t._size) {
		for (int i = 0; i < _size; i++)
			r._data[i] = _data[i] + t._data[i];
	} else {
		std::cerr << " Tensor size mismatch !\n";
	}	
	return r;
}

Tensor1D Tensor1D::operator-(const Tensor1D& t) {
	Tensor1D r(_size);
	if (_size == t._size) {
		for (int i = 0; i < _size; i++)
			r._data[i] = _data[i] - t._data[i];
	} else {
		std::cerr << " Tensor size mismatch !\n";
	}
	return r;
}

Tensor1D& Tensor1D::operator+=(const Tensor1D& t) {
	if (_size == t._size) {
		for (int i = 0; i < _size; i++)
			_data[i] += t._data[i];
	} else {
		std::cerr << " Tensor size mismatch !\n";
	}
	return *this;
}

Tensor1D& Tensor1D::operator-=(const Tensor1D& t) {
	if (_size == t._size) {
		for (int i = 0; i < _size; i++)
			_data[i] -= t._data[i];
	} else {
		std::cerr << " Tensor size mismatch !\n";
	}
	return *this;
}


float& Tensor1D::operator()(int n) {
	return _data[n];
}

Tensor1D& Tensor1D::operator=(const Tensor1D& t) {

	if (this == &t) 
		return *this;

//	if (_data != 0) 
	delete [] _data;

	_size = t._size; 
	_data = new float [_size];
	
	for (int i = 0; i < _size; i++)
		_data[i] = t._data[i];

	return *this;
}

int Tensor1D::size() const {
	return _size;
}

void Tensor1D::setValue(float v) {
	for (int i = 0; i < _size; i++)
		_data[i] = v;
}

Tensor1D::~Tensor1D() {
	if (_data != 0)
		delete [] _data;
}


Tensor1D Tensor1D::convolve(const Tensor1D& kernel) {
	Tensor1D r(_size - kernel._size +1);
	for (int i = 0; i < r._size; i++)
		for (int di = 0; di < kernel._size; di++)
			r._data[i] = _data[i + di] * kernel._data[i + di];
	return r;
}

Tensor1D operator*(const Tensor1D& t, float lambda) {
	Tensor1D r(t._size);
	for (int i = 0; i < t._size; i++)
		r._data[i] = lambda * t._data[i];
	return r;
}  

Tensor1D operator*(float lambda, const Tensor1D& t) {
	Tensor1D r(t._size);
	for (int i = 0; i < t._size; i++)
		r._data[i] = lambda * t._data[i];
	return r;
}  

float dot(Tensor1D& lhs, Tensor1D& rhs) {
	float acc = 0;
	if (lhs._size == rhs._size) {
		for (int i = 0; i < lhs._size; i++)
			acc += lhs._data[i] * rhs._data[i];
	} else {
		std::cerr << " Tensor size mismatch !\n";
	}
	return acc;
}
/*
Tensor2D outer(Tensor1D& lhs, Tensor1D& rhs) {
	Tensor2D r = Tensor2D(lhs._size, rhs._size);
	for (int i = 0; i < lhs._size; i++)
		for (int j = 0; j < rhs._size; j++)
			r._data[i][j] = lhs._data[i] * rhs._data[j];
	return r; 	
}
*/
std::ostream& operator<<(std::ostream& os, const Tensor1D& t)
{
	for (int i = 0; i < t._size; i++) {
		os << std::setw(OSTREAM_WIDTH);
		os << t._data[i] << ' ';
	}
	os << std::endl;
	return os;
}


		
