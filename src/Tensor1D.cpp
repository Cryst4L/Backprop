#include "Log.h"
#include "Tensor1D.h"

Tensor1D::Tensor1D(int size) 
  : _size(size) 
{
	_data = new float[size];
}

Tensor1D::Tensor1D(const Tensor1D& t)
  : _size(t._size)
{
	_data = new float[_size];
	for (int i = 0; i < _size; i++)	
		_data[i] = t._data[i];
}

Tensor1D& Tensor1D::operator=(const Tensor1D& t) {

	if (this == &t) return *this;

	if (_data != 0) delete [] _data;

	_size = t._size; 
	_data = new float [_size];
	
	for (int i = 0; i < _size; i++)
		_data[i] = t._data[i];

	return *this;
}

int Tensor1D::size() {
	return _size;
}

void Tensor1D::set(float value) {
	for (int i = 0; i < _size; i++)
		_data[i] = value;
}

float& Tensor1D::operator()(int n) {
	return _data[n];
}

float dot(Tensor1D& lhs, Tensor1D& rhs) {
	float acc = 0;
	if (lhs.size() == rhs.size()) {
		for (int i = 0; i < lhs.size(); i++)
			acc += lhs(i) * rhs(i);
	} else {
		std::cerr << " Tensor size mismatch !" << std::endl;
	}
	return acc;
}

Tensor1D::~Tensor1D() {
	if (_data)
		delete _data;
}

		
