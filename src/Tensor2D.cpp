#include "Log.h"
#include "Tensor2D.h"

Tensor2D::Tensor2D(int rows, int cols)
  : _rows(rows), _cols(cols) 
{
	_data = new float*[rows];
	for (int i = 0; i < rows; i++)
		_data[i] = new float[cols];

	for (int i = 0; i < _rows; i++)
		for (int j = 0; j < _cols; j++)
			_data[i][j] = 0;

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
		std::cerr << " Tensor size mismatch !\n";
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

Tensor2D& Tensor2D::operator+=(const Tensor2D& t) {
	if (_rows == t._rows && _cols == t._cols) {
		for (int i = 0; i < _rows; i++)
			for (int j = 0; j < _cols; j++)
				_data[i][j] += t._data[i][j];
	} else {
		std::cerr << " Tensor size mismatch !\n";
	}
	return *this;
}

Tensor2D& Tensor2D::operator-=(const Tensor2D& t) {
	if (_rows == t._rows && _cols == t._cols) {
		for (int i = 0; i < _rows; i++)
			for (int j = 0; j < _cols; j++)
				_data[i][j] -= t._data[i][j];
	} else {
		std::cerr << " Tensor size mismatch !" << std::endl;
	}
	return *this;
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
		std::cerr << " Tensor size mismatch !\n";
	}
	return r;
}

Tensor1D Tensor2D::operator*(const Tensor1D& t) {
	Tensor1D r(t._size);
	if (_cols == t._size) {
		for (int i = 0; i < _rows; i++) {
			float acc = 0;
			for (int j = 0; j < _cols; j++)
				acc += _data[i][j] * t._data[j];
			r._data[i] = acc;
		}
	} else {
		std::cerr << " Tensor size mismatch !\n";
	}
	return r;
}
	
Tensor2D& Tensor2D::operator=(const Tensor2D& t) {

	if (this == &t) 
		return *this;

//	if (_data != 0) 
	for (int i = 0; i < _rows; i++)
		delete [] _data[i];
	delete [] _data;

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


int Tensor2D::rows() const {
	return _rows;
}

int Tensor2D::cols() const {
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

Tensor1D Tensor2D::row(const int n) {
	Tensor1D t(_cols);
	for (int i = 0; i < _cols; i++)
		t._data[i] = _data[n][i];
	return t;
}

Tensor1D Tensor2D::col(const int n) {
	Tensor1D t(_rows);
	for (int i = 0; i < _rows; i++)
		t._data[i] = _data[i][n];
	return t;
}

Tensor1D Tensor2D::flat() {
	Tensor1D t(_rows * _cols);
	int idx = 0;
	for (int i = 0; i < _rows; i++)
		for (int j = 0; j < _cols; j++) {
			t(idx) = _data[i][j];
			idx++;
		}
	return t;
}

Tensor2D Tensor2D::convolve(const Tensor2D& kernel) {
	Tensor2D r(_rows - kernel._rows + 1, _cols - kernel._cols + 1);
	for (int i = 0; i < r._rows; i++)
		for (int j = 0; j < r._cols; j++)
			for (int di = 0; di < kernel._rows; di++)
				for (int dj = 0; dj < kernel._cols; dj++)
					r._data[i][j] = _data[i + di][j + dj] * kernel._data[di][dj];
	return r;		
}	

Tensor2D operator*(const Tensor2D& t, float lambda) {
	Tensor2D r(t._rows, t._cols);
	for (int i = 0; i < t._rows; i++)
		for (int j = 0; j < t._cols; j++)
			r._data[i][j] = lambda * t._data[i][j];
	return r;
} 

Tensor2D operator*(float lambda, const Tensor2D& t) {
	Tensor2D r(t._rows, t._cols);
	for (int i = 0; i < t._rows; i++)
		for (int j = 0; j < t._cols; j++)
			r._data[i][j] = lambda * t._data[i][j];
	return r;
} 

Tensor1D operator*(Tensor2D& lhs, Tensor1D& rhs) {
	Tensor1D r(lhs.rows());
	if (lhs.cols() == rhs.size()) {
		for (int i = 0; i < lhs.rows(); i++)
			for (int j = 0; j < lhs.cols(); j++)
				r(i) += lhs(i,j) * rhs(j);
	} else {
		std::cerr << " Tensor size mismatch !\n";
	}
	return r;			
}

Tensor1D operator*(Tensor1D& lhs, Tensor2D& rhs) {
	Tensor1D r(rhs.cols());
	if (lhs.size() == rhs.rows()) {
		for (int j = 0; j < rhs.cols(); j++)
			for (int i = 0; i <  rhs.rows(); i++)
				r(j) += lhs(i) * rhs(i,j); 
	} else {
		std::cerr << " Tensor size mismatch !\n";
	}
	return r;
}

Tensor2D outer(Tensor1D& lhs, Tensor1D& rhs) {
	Tensor2D r(lhs.size(), rhs.size());
	for (int i = 0; i < lhs.size(); i++)
		for (int j = 0; j < rhs.size(); j++)
			r(i,j) = lhs(i) * rhs(j);
	return r; 
}

std::ostream& operator<<(std::ostream& os, const Tensor2D& t) {
	for (int i = 0; i < t._rows; i++) {
		for (int j = 0; j < t._cols; j++) {
			os << std::setw(OSTREAM_WIDTH);
			os << t._data[i][j] << ' ';
		}
		os << '\n';
	}
	return os;
}
	 

