// Copyright 2011-2019 the Polygon Mesh Processing Library developers.
// Distributed under a MIT-style license, see LICENSE.txt for details.

#include "VariationalRemeshing.h"

int num_iters = 20;

int main(int argc, char** argv)
{
    pmp::SurfaceMesh mesh;
    mesh.read(argv[1]);

    VoronoiCellMesh cellMesh(mesh);
    cellMesh.random_density_sampling(mesh.n_vertices());
    for(size_t i = 0; i < num_iters ; i++){
        std::cout << "\riteration: " << i << "/" << num_iters;
        cellMesh.lloyd_relaxation();

    }

    Eigen::MatrixX3d V;
    Eigen::MatrixX3i F;
    cellMesh.lift_up(V, F);
    write_off("remeshed.off", V, F);
}
