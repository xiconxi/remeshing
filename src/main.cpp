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

    cellMesh.write("uv.off");
    // save initial random samples
    write_sample_off("./initial_samples.off", cellMesh.V_site_);
    std::cout << "construct cvt :" << std::endl;
    pmp::SurfaceMesh overlapped_mesh;
    pmp::FaceProperty<int> f_region;
    for(size_t i = 0; i < num_iters ; i++){
        std::cout << "\riteration: " << i << "/" << num_iters;
        cellMesh.triangular_voronoi_overlapping(overlapped_mesh, f_region);
        cellMesh.lloyd_relaxation(overlapped_mesh, f_region);
    }

    write_sample_off("./samples.off", cellMesh.V_site_);
    face_coloring(overlapped_mesh, f_region);
    write_colored_off("voronoi_overlap.off", overlapped_mesh);

    Eigen::MatrixX3d V;
    Eigen::MatrixX3i F;
    cellMesh.lift_up(V, F);
    write_off("remeshed.off", V, F);
}
