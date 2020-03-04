#include <iostream>
#include "src/Tensor1D.h"
#include "src/Tensor2D.h"
int main(void)
{
	Tensor1D A(10); A.setValue(5);
	Tensor1D B(10); B.setValue(3);
	int temp = dot(A,B);
	std::cout << temp << '\n';

	Tensor2D C(5,4); C.setValue(5);
	Tensor2D D(5,4); D.setValue(3);
/*
	Tensor2D E = C-D;
	for (int i = 0; i < E.rows(); i++) {
		for (int j = 0; j < E.cols(); j++) {
			std::cout << E(i,j) << ' ';
		}
		std::cout << std::endl; 
	}
*/
	return 0;
}
