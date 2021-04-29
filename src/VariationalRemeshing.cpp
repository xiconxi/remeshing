//
// Created by pupa on 2021/4/20.
//

#include "VariationalRemeshing.h"
#include <pmp/algorithms/SurfaceParameterization.h>
#include <pmp/algorithms/DifferentialGeometry.h>
#include <queue>
#include <random>
#include <fstream>
#include "MeshUtils.h"

VoronoiCellMesh::VoronoiCellMesh(const pmp::SurfaceMesh& mesh): tex_mesh_(mesh) {
    auto v_tex = tex_mesh_.get_vertex_property<pmp::TexCoord>("v:tex");
    auto v_coord = tex_mesh_.vertex_property<pmp::Point>("v:coord3");
    v_density_ = tex_mesh_.vertex_property<double>("v:density");

    V_tex.resize(mesh.n_vertices(), 2);
    for(auto v: mesh.vertices()){
        v_coord[v] = tex_mesh_.position(v);
        tex_mesh_.position(v) = pmp::Point(v_tex[v][0], v_tex[v][1], 0);
        V_tex.row(v.idx()) = Eigen::Vector2d(v_tex[v][0], v_tex[v][1]);
        v_density_[v] = pmp::voronoi_area(mesh, v)/pmp::voronoi_area(tex_mesh_, v);
    }
    tex_mesh_.remove_vertex_property(v_tex);


    F.resize(mesh.n_faces(), 3);
    for(auto f: mesh.faces()) {
        auto fvit = mesh.vertices(f);
        F(f.idx(), 0) = (*fvit).idx();
        F(f.idx(), 1) = (*++fvit).idx();
        F(f.idx(), 2) = (*++fvit).idx();
    }
    size_t be_cnt = 0, idx = 0;
    for(auto e: mesh.edges())
        if(mesh.is_boundary(e)) be_cnt++;
    E_bnd.resize(be_cnt, 2);
    for(auto e: mesh.edges()) {
        if(mesh.is_boundary(e)){
            E_bnd(idx, 0) = mesh.vertex(e, 0).idx();
            E_bnd(idx++, 1) = mesh.vertex(e, 1).idx();
        }
    }
    kd_tree_ = std::make_shared<pmp::TriangleKdTree>(tex_mesh_);


//    tex_mesh_.write("output/tex.obj");
}

double VoronoiCellMesh::density(pmp::Point p) const {
    auto neighbor = kd_tree_->nearest(p);
    auto fvit = tex_mesh_.vertices(neighbor.face);
    pmp::Vertex uvw[3] = {*fvit, *++fvit, *++fvit};
    pmp::Point bc = pmp::barycentric_coordinates(neighbor.nearest, tex_mesh_.position(uvw[0]),
                                 tex_mesh_.position(uvw[1]), tex_mesh_.position(uvw[2]));
    return bc_interp(v_density_, neighbor.face, bc);
}


void VoronoiCellMesh::random_sampling(size_t n_sample) {
    samples_.resize(n_sample);
    for (size_t i = 0; i < n_sample; i++) {
        double r = std::sqrt((rand() % 1000) / 1000.0)*0.496;
        double angle = (rand() % 1000) / 1000.0 * M_PI * 2;
        samples_[i] = Eigen::RowVector2d (0.5, 0.5);
        samples_[i] += r*Eigen::RowVector2d (std::cos(angle), std::sin(angle));
    }
}

void VoronoiCellMesh::random_density_sampling(size_t n_sample) {
    samples_.reserve(n_sample);
    auto coord0 = tex_mesh_.get_vertex_property<pmp::Point>("v:coord3");
    auto coord1 = tex_mesh_.get_vertex_property<pmp::Point>("v:point");
    assert(coord0 && coord1);
    auto f_density = tex_mesh_.face_property<double>("f:density");
    // step1. calc face density
    for(auto f: tex_mesh_.faces()) {
        auto v0 = pmp::Vertex(F(f.idx(), 0));
        auto v1 = pmp::Vertex(F(f.idx(), 1));
        auto v2 = pmp::Vertex(F(f.idx(), 2));
        f_density[f] = pmp::triangle_area(coord0[v0], coord0[v1], coord0[v2]);
        f_density[f] /= pmp::triangle_area(coord1[v0], coord1[v1], coord1[v2]);
        f_density[f] = std::log(10+f_density[f]);
    }

    double max_density = *std::max_element(f_density.data(), f_density.data() + tex_mesh_.n_faces());
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> real_dis(0, max_density);
    while (samples_.size() != n_sample) {
        auto f = pmp::Face(rand()%tex_mesh_.n_faces());
        double d = real_dis(gen);
        if(d > f_density[f]) continue;
        Eigen::Vector3f bc = Eigen::Vector3f::Random().cwiseAbs();
        pmp::Point p = bc_interp(coord1, f, bc/bc.sum());
        samples_.push_back(Eigen::RowVector2d(p[0], p[1]));
    }

    tex_mesh_.remove_face_property(f_density);
}

void VoronoiCellMesh::dithering_sampling(size_t n_sample) {
//    for(auto name: tex_mesh_.vertex_properties())
//        std::cout << name << std::endl;
    auto coord0 = tex_mesh_.get_vertex_property<pmp::Point>("v:coord3");
    auto coord1 = tex_mesh_.get_vertex_property<pmp::Point>("v:point");
    assert(coord0 && coord1);
    auto f_density = tex_mesh_.face_property<double>("f:density");
    // step1. calc face density
    double density_sum = 0;
    for(auto f: tex_mesh_.faces()) {
        auto fvit = tex_mesh_.vertices(f);
        auto v0 = *(fvit);
        auto v1 = *(++fvit);
        auto v2 = *(++fvit);
        f_density[f] = pmp::triangle_area(coord0[v0], coord0[v1], coord0[v2]);
        density_sum += f_density[f];
    }
    for(auto f: tex_mesh_.faces())
        f_density[f] *= n_sample/density_sum;

    auto root_face = kd_tree_->nearest(pmp::Point(0.5, 0.5, 0)).face;

    auto f_visited = tex_mesh_.face_property<bool>("f:visited", false);
    std::queue<pmp::Face> Q;
    Q.push(root_face);
    std::vector<pmp::Face> face_sequence;
    face_sequence.reserve(tex_mesh_.n_faces());
    f_visited[root_face] = true;
    while(!Q.empty()) {
        auto face = Q.front(); Q.pop();
        face_sequence.push_back(face);
        for(auto h : tex_mesh_.halfedges(face)) {
            auto next_f = tex_mesh_.face(tex_mesh_.opposite_halfedge(h));
            if(next_f.is_valid() && !f_visited[next_f]) {
                Q.push(next_f);
                f_visited[next_f] = true;
            }
        }
    }

    samples_.clear();
    double error = 0;
    for(int i = 0; i < face_sequence.size(); i++) {
        double face_error = error + f_density[face_sequence[i]];
        while(face_error >= 1) {
            Eigen::RowVector3d bc = Eigen::RowVector3d::Random().cwiseAbs();
            pmp::Point p = bc_interp(coord1, face_sequence[i], bc/bc.sum());
            samples_.push_back(Eigen::RowVector2d (p[0], p[1]));
            face_error -= 1;
        }
        error = face_error;
    }
    tex_mesh_.remove_face_property(f_density);
    tex_mesh_.remove_face_property(f_visited);
}

void VoronoiCellMesh::lloyd_relaxation() {
    Eigen::MatrixX2d V_cell, V_sample;
    Eigen::MatrixX2i E_cell, E;
    voronoi_cell(samples_, V_cell, E_cell);

    V_sample.resize(samples_.size(), 2);
    for(size_t i = 0; i < samples_.size(); i++)
        V_sample.row(i) = Eigen::RowVector2d(samples_[i][0], samples_[i][1]);

    E.resize(E_cell.rows()+E_bnd.rows(), 2);
    E.topRows(E_cell.rows()) = E_cell.array() + tex_mesh_.n_vertices();
    E.bottomRows(E_bnd.rows()) = E_bnd;

    cell_mesh_.clear();
    mesh_voronoi_overlapping(V_tex, F, V_cell, E, V_sample, cell_mesh_);

//    cell_mesh_.write("output/cell_mesh_.obj");

    auto cell_v_density = cell_mesh_.vertex_property<double>("v:density");
    for(auto v: cell_mesh_.vertices())
        cell_v_density[v] = density(cell_mesh_.position(v));

    auto f_region = cell_mesh_.face_property<int>("f:site", -1);
    std::vector<double> mass(samples_.size(), 1e-6);
    std::vector<pmp::Point> site_centroid(samples_.size());
    for(size_t i = 0; i < samples_.size(); i++)
        site_centroid[i] = pmp::Point(samples_[i][0], samples_[i][1] ,0)*1e-6;
    for(auto f: cell_mesh_.faces()) {
        if(f_region[f] == -1) continue;
        double face_area = pmp::triangle_area(cell_mesh_, f);
        for(auto v: cell_mesh_.vertices(f)) {
            site_centroid[f_region[f] ] += cell_mesh_.position(v) * cell_v_density[v] * face_area;
            mass[f_region[f]] += cell_v_density[v] * face_area;
        }
    }
    for(size_t i = 0; i < samples_.size(); i++) {
        site_centroid[i] /= mass[i];
        samples_[i] = Eigen::RowVector2d (site_centroid[i][0], site_centroid[i][1]);
    }

//    for(auto v: cell_mesh_.vertices())
//        cell_mesh_.position(v)[2] = cell_v_density[v];
//
//    cell_mesh_.write("output/cell_mesh_3.obj");

}

void VoronoiCellMesh::lift_up(Eigen::MatrixX3d& V3d, Eigen::MatrixX3i& F) {
    auto v_reid = tex_mesh_.vertex_property<int>("v:bnd", -1);
    size_t bnd_size = 0;
    for(auto v: tex_mesh_.vertices())
        if(tex_mesh_.is_boundary(v))
            v_reid[v] = bnd_size++;
    Eigen::MatrixX2d V2d(bnd_size + samples_.size(), 2);

    for(size_t i = 0; i < samples_.size(); i++)
        V2d.row(i+bnd_size) = samples_[i];
    for(auto v: tex_mesh_.vertices())
        if(tex_mesh_.is_boundary(v))
            V2d.row(v_reid[v]) = V_tex.row(v.idx());

    Eigen::MatrixX2i Edge(E_bnd.rows(), 2);
    for(size_t i = 0 ; i < E_bnd.rows(); i++) {
        Edge(i, 0) = v_reid[pmp::Vertex(E_bnd(i, 0))];
        Edge(i, 1) = v_reid[pmp::Vertex(E_bnd(i, 1))];
    }

    triangulation(V2d, Edge, V2d, F);
    V3d.resize(V2d.rows(), 3);
    auto v_coord = tex_mesh_.get_vertex_property<pmp::Point>("v:coord3");
    for(size_t i = 0; i < V2d.rows(); i++) {
        pmp::Point p(V2d(i, 0), V2d(i, 1), 0);
        auto root_face = kd_tree_->nearest(p).face;
        auto fvit = tex_mesh_.vertices(root_face);
        pmp::Vertex uvw[3] = {*fvit, *++fvit, *++fvit};
        pmp::Point bc = pmp::barycentric_coordinates(p, tex_mesh_.position(uvw[0]),
                            tex_mesh_.position(uvw[1]), tex_mesh_.position(uvw[2]));
        p = bc[0] * v_coord[uvw[0]] + bc[1] * v_coord[uvw[1]] + bc[2] * v_coord[uvw[2]];
        V3d.row(i) = Eigen::RowVector3d(p[0], p[1], p[2]);
    }


//    std::ofstream file;
//    file.open("output/unlift_up_2d.obj");
//    for (int i = 0; i < V2d.rows(); i++)
//        file << "v " << V2d.row(i) << " 0\n";
//    for (int i = 0; i < F.rows(); i++)
//        file << "f " << F.row(i)+Eigen::RowVector3i(1,1,1)<< " \n";
//    file.close();
}


void VariationalRemeshing::harmonic_parameterization() {
    // parameterize
    pmp::SurfaceParameterization param_(mesh_);
    param_.lscm();
    auto tex = mesh_.get_vertex_property<pmp::TexCoord>("v:tex");
    for(auto v: mesh_.vertices()) {
        auto diff = tex[v] - pmp::TexCoord (0.5, 0.5);
        if(pmp::norm(diff) > 0.5+1e-3)
            tex[v] = pmp::TexCoord (0.5, 0.5)+diff/pmp::norm(diff);
    }

    voronoi_mesh_ = std::make_shared<VoronoiCellMesh>(mesh_);
}

void VariationalRemeshing::initial_sampling(size_t n_samples) {
    voronoi_mesh_->random_density_sampling(n_samples);
//    voronoi_mesh_->random_sampling(n_samples);
//    voronoi_mesh_->dithering_sampling(n_samples);
}

void VariationalRemeshing::wire_merged_lloyd_relaxation(size_t n_iters) {

    std::ofstream file;
//    file.open("output/sample.obj");
//    for (int i = 0; i < voronoi_mesh_->samples_.size(); i++)
//        file << "v " << voronoi_mesh_->samples_[i] << " 0\n";
//    file.close();

    for(int i = 0; i < n_iters; i++){
        voronoi_mesh_->lloyd_relaxation();
//        std::ofstream file;
//        file.open("output/sample"+std::to_string(i)+".obj");
//        for (int i = 0; i < voronoi_mesh_->samples_.size(); i++)
//            file << "v " << voronoi_mesh_->samples_[i] << " 0\n";
//        file.close();
    }
    Eigen::MatrixX3d V;
    Eigen::MatrixX3i F;
    voronoi_mesh_->lift_up(V, F);

    file.open("output/sample.obj");
    for (int i = 0; i < voronoi_mesh_->samples_.size(); i++)
        file << "v " << voronoi_mesh_->samples_[i] << " 0\n";
    file.close();

    file.open("output/lift_up_3d.obj");
    for (int i = 0; i < V.rows(); i++)
        file << "v " << V.row(i) << " 0\n";
    for (int i = 0; i < F.rows(); i++)
        file << "f " << F.row(i)+Eigen::RowVector3i(1,1,1)<< " \n";
    file.close();

//    celled_mesh.write("output/celled_mesh.obj");
}