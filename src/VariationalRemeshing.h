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
    VoronoiCellMesh(const pmp::SurfaceMesh& mesh);

    double density(pmp::Vertex v) const { return v_density_[v];}

    template <class Type>
    Type bc_interp(pmp::VertexProperty<Type> vprop, pmp::Face f, pmp::Vector<float, 3> bc) const {
        auto fvit = tex_mesh_.vertices(f);
        pmp::Vertex uvw[3] = {*fvit, *++fvit, *++fvit};
        return bc[0] * vprop[ uvw[0] ] + bc[1] * vprop[ uvw[1] ] + bc[2] * vprop[ uvw[2] ];
    }

    double density(pmp::Point p) const ;

    void random_sampling(size_t n_sample);

    void random_density_sampling(size_t n_sample);

    void dithering_sampling(size_t n_sample);

    void lloyd_relaxation();

    void lift_up(Eigen::MatrixX3d& V, Eigen::MatrixX3i& F);

    pmp::SurfaceMesh cell_mesh_;

    std::vector<Eigen::RowVector2d> samples_;

private:
    pmp::SurfaceMesh tex_mesh_;
    Eigen::MatrixX2d V_tex;
    Eigen::MatrixX3i F;
    Eigen::MatrixX2i E_bnd;
    std::shared_ptr< pmp::TriangleKdTree > kd_tree_;
    pmp::VertexProperty<double> v_density_;
};


class VariationalRemeshing {
public:
    //! Construct with mesh to be remeshed.
    VariationalRemeshing(pmp::SurfaceMesh& mesh): mesh_(mesh){}

    //! parameterize and compute the local density
    void harmonic_parameterization();

    void initial_sampling(size_t n_samples);

    void wire_merged_lloyd_relaxation(size_t n_iters);

private:
    pmp::SurfaceMesh& mesh_;

    std::shared_ptr<VoronoiCellMesh> voronoi_mesh_;
};


