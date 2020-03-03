#include <iostream>
#include "src/Tensor1D.h"
#include "src/Tensor2D.h"
int main(void)
{
	Tensor1D A(100); A.set(5);
	Tensor1D B(100); B.set(3);
	int temp = dot(A,B);
	std::cout << temp << '\n';


	Tensor2D C(10,4);
	C.set(5);
//	float temp2 = C(8,3);
//	std::cout << temp2 << '\n';
	return 0;
}
