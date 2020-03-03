
class Tensor2D {
  private:
	int _rows;
	int _cols;
	float ** _data;
 public:
	Tensor2D(int rows, int cols);
	int rows();
	int cols();
	void set(float value);
	float& operator()(int i, int j);
//	Tensor1D row(int n);
//	Tensor1D col(int n);
//	Tensor1D operator*(const Tensor1D& rhs);
	~Tensor2D();
};


