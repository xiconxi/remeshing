//
// Created by pupa on 2021/4/20.
//

#pragma once
#include <pmp/SurfaceMesh.h>
#include <pmp/algorithms/TriangleKdTree.h>
#include <pmp/algorithms/BarycentricCoordinates.h>

using pmp::Scalar;

typedef Eigen::Matrix<int  , Eigen::Dynamic, 2, Eigen::RowMajor> MatrixX2Ir;
typedef Eigen::Matrix<int  , Eigen::Dynamic, 3, Eigen::RowMajor> MatrixX3Ir;
typedef Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor> MatrixX2dr;

class VoronoiCellMesh;

void write_sample_off(std::string file_path, MatrixX2dr& V) ;

void write_off(std::string file_path, Eigen::MatrixX3d& V, Eigen::MatrixX3i& F) ;

void write_colored_off(std::string file_path, pmp::SurfaceMesh& mesh) ;

void face_coloring(pmp::SurfaceMesh& mesh, pmp::FaceProperty<int> f_label) ;

/*
 * input: Samples
 * output: V, E
 * 依据Sample构建Voronoi Diagram, 并提取Voronoi Cell的顶点V和边E
 */
void voronoi_cell(MatrixX2dr& S, MatrixX2dr& V, MatrixX2Ir & E);

/*
 * input: V_tri, F, V_cell, E, V_sample
 * output: cell_mesh
 * Overlap参数化网格(V_tri, F)、VoronoiCell（V_cell, E)和随机采样点V_sample
 */
void mesh_voronoi_overlapping(MatrixX2dr& V_tri, MatrixX3Ir& F,
                              MatrixX2dr& V_cell, MatrixX2Ir& E,
                              MatrixX2dr& V_sample,
                              pmp::SurfaceMesh& cell_mesh);

/*
 * input: V, E
 * output: V2, F2
 */
void triangulation(const MatrixX2dr& V, const MatrixX2Ir& E,
                   Eigen::MatrixX2d& V2, Eigen::MatrixX3i & F2);




class VoronoiCellMesh: public pmp::SurfaceMesh{
public:
    explicit VoronoiCellMesh(const pmp::SurfaceMesh& mesh);

    void random_density_sampling(size_t n_sample);

    void triangular_voronoi_overlapping(pmp::SurfaceMesh& cell_mesh_, pmp::FaceProperty<int>& f_region);

    void lloyd_relaxation(pmp::SurfaceMesh& cell_mesh_, pmp::FaceProperty<int>& f_region);

    void lift_up(Eigen::MatrixX3d& V, Eigen::MatrixX3i& F);

    MatrixX2dr V_site_;

private:

    double density(pmp::Vertex v) const { return v_density_[v];}

    double density(pmp::Point p) const ;

    template <class Type>
    Type bc_interp(pmp::VertexProperty<Type> vprop, pmp::Face f, pmp::Vector<float, 3> bc) const {
        auto fvit = vertices(f);
        pmp::Vertex uvw[3] = {*fvit, *++fvit, *++fvit};
        return bc[0] * vprop[ uvw[0] ] + bc[1] * vprop[ uvw[1] ] + bc[2] * vprop[ uvw[2] ];
    }

    MatrixX2dr V_tex;
    MatrixX3Ir F;
    MatrixX2Ir E_bnd;
    std::shared_ptr< pmp::TriangleKdTree > kd_tree_;
    pmp::VertexProperty<double> v_density_;
};