//
// Created by pupa on 2021/4/20.
//

#pragma once
#include <pmp/SurfaceMesh.h>
#include <pmp/algorithms/TriangleKdTree.h>
#include <pmp/algorithms/BarycentricCoordinates.h>

using pmp::Scalar;


class VoronoiCellMesh{
public:
    explicit VoronoiCellMesh(const pmp::SurfaceMesh& mesh);

    void random_density_sampling(size_t n_sample);

    void lloyd_relaxation();

    void lift_up(Eigen::MatrixX3d& V, Eigen::MatrixX3i& F);

    std::vector<Eigen::RowVector2d> samples_;

private:

    double density(pmp::Vertex v) const { return v_density_[v];}

    double density(pmp::Point p) const ;

    template <class Type>
    Type bc_interp(pmp::VertexProperty<Type> vprop, pmp::Face f, pmp::Vector<float, 3> bc) const {
        auto fvit = tex_mesh_.vertices(f);
        pmp::Vertex uvw[3] = {*fvit, *++fvit, *++fvit};
        return bc[0] * vprop[ uvw[0] ] + bc[1] * vprop[ uvw[1] ] + bc[2] * vprop[ uvw[2] ];
    }

    pmp::SurfaceMesh tex_mesh_;
    Eigen::MatrixX2d V_tex;
    Eigen::MatrixX3i F;
    Eigen::MatrixX2i E_bnd;
    std::shared_ptr< pmp::TriangleKdTree > kd_tree_;
    pmp::VertexProperty<double> v_density_;
};

void write_off(std::string file_path, Eigen::MatrixX3d& V, Eigen::MatrixX3i& F) ;

void write_colored_off(std::string file_path, pmp::SurfaceMesh& mesh) ;

void face_coloring(pmp::SurfaceMesh& mesh, pmp::FaceProperty<int> f_label) ;

/*
 * input: S
 * output: V, E
 */
void voronoi_cell(std::vector<Eigen::RowVector2d>& S,
                  Eigen::MatrixX2d& V, Eigen::MatrixX2i& E);

/*
 * input: V_tri, F, V_cell, E, V_sample
 * output: cell_mesh
 */
void mesh_voronoi_overlapping(Eigen::MatrixX2d& V_tri, Eigen::MatrixX3i& F,
                              Eigen::MatrixX2d& V_cell, Eigen::MatrixX2i& E,
                              Eigen::MatrixX2d& V_sample,
                              pmp::SurfaceMesh& cell_mesh);

/*
 * input: V, E
 * output: V2, F2
 */
void triangulation(const Eigen::MatrixX2d& V, const Eigen::MatrixX2i& E,
                   Eigen::MatrixX2d& V2, Eigen::MatrixX3i& F2);


