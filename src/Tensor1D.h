class Tensor1D 
{
  private:
	int _size;
	float * _data;
  public:
	Tensor1D(int size);
	Tensor1D(const Tensor1D& t);
	Tensor1D& operator=(const Tensor1D& t);
	int size();
	void set(float value);
	float& operator()(int n);
	~Tensor1D();
};

float dot(Tensor1D& lhs, Tensor1D& rhs);
