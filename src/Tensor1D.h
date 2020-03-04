class Tensor1D 
{
  private:
	int _size;
	float * _data;

  public:
	Tensor1D(int size);
	Tensor1D(const Tensor1D& t);
	~Tensor1D();

	Tensor1D operator+(const Tensor1D& t);
	Tensor1D operator-(const Tensor1D& t);

	float& operator()(int n);
	Tensor1D& operator=(const Tensor1D& t);

	int size();
	void setValue(float v);
	friend float dot(Tensor1D& lhs, Tensor1D& rhs);
};


