#pragma once
#include <Eigen/Core>
#include <vector>

using namespace Eigen;

namespace Backprop
{
void reshape(VectorXd& unshaped, std::vector<MatrixXd>& maps)
{
	if (maps.empty()) return;

	int rows = maps[0].rows();
	int cols = maps[0].cols();
	int nb_maps = maps.size();

	for (int i = 0; i < nb_maps; i++) {
		VectorXd unshaped_map = unshaped.segment(i * rows * cols, rows * cols);
		maps[i] = Map <MatrixXd> (unshaped_map.data(), rows, cols);
	}
}

void unshape(std::vector<MatrixXd>& maps, VectorXd& unshaped)
{
	if (maps.empty()) {
		unshaped = VectorXd(0);
		return;
	}

	int unshaped_map_size = maps[0].rows() * maps[0].cols();

	if (unshaped.size() != (int) maps.size() * unshaped_map_size)
		unshaped = VectorXd(maps.size() * unshaped_map_size);

	for (int i = 0; i < (int) maps.size(); i++) {
		VectorXd unshaped_map = Map <VectorXd> (maps[i].data(), unshaped_map_size);
		unshaped.segment(i * unshaped_map_size, unshaped_map_size) = unshaped_map;
	}
}
}
