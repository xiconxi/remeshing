//
// Created by pupa on 2021/4/27.
//

#pragma once
#include <pmp/SurfaceMesh.h>
#include <Eigen/Core>
#include <vector>


void write_colored_off(std::string file_path, pmp::SurfaceMesh& mesh) ;

void face_coloring(pmp::SurfaceMesh& mesh, pmp::FaceProperty<int> f_label) ;

void voronoi_cell(std::vector<Eigen::RowVector2d>& S,
                  Eigen::MatrixX2d& V, Eigen::MatrixX2i& E);

void mesh_voronoi_overlapping(Eigen::MatrixX2d& V_tri, Eigen::MatrixX3i& F,
                              Eigen::MatrixX2d& V_cell, Eigen::MatrixX2i& E,
                              Eigen::MatrixX2d& V_sample,
                              pmp::SurfaceMesh& cell_mesh);

void triangulation(const Eigen::MatrixX2d& V, const Eigen::MatrixX2i& E,
                   Eigen::MatrixX2d& V2, Eigen::MatrixX3i& F2);



