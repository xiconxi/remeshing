//
// Created by pupa on 2021/4/20.
//

#include "VariationalRemeshing.h"
#include <pmp/algorithms/SurfaceParameterization.h>
#include <pmp/algorithms/DifferentialGeometry.h>
#include <queue>
#include <random>
#include <fstream>

#define JCV_REAL_TYPE double
#define JC_VORONOI_IMPLEMENTATION
#include "jc_voronoi.h"
#include <memory>
#include <stack>
#include <pmp/SurfaceMeshIO.h>
#include <triangulation.h>

void write_colored_off(std::string file_path, pmp::SurfaceMesh& mesh) {
    std::ofstream file(file_path);
    file << "OFF\n" << mesh.n_vertices() << ' ' << mesh.n_faces() << " 0\n";
    auto f_color = mesh.get_face_property<pmp::Color>("f:color");
    for (auto v: mesh.vertices())
        file << mesh.position(v) << " \n";
    for(auto f: mesh.faces()) {
        auto fvit = mesh.vertices(f);
        size_t v0 = (*fvit).idx();
        size_t v1  = (*++fvit).idx();
        size_t v2 = (*++fvit).idx();
        file << "3 " << v0 << ' ' << v1  << ' '  << v2  << ' ';
        file << f_color[f] << std::endl;
    }
    file.close();
}

void write_off(std::string file_path, Eigen::MatrixX3d& V, Eigen::MatrixX3i& F) {
    std::ofstream file(file_path);
    file << "OFF\n" << V.rows() << ' ' << F.rows() << " 0\n";
    for (size_t i = 0; i < V.rows(); i++)
        file << V.row(i) << " \n";
    for (size_t i = 0; i < F.rows(); i++)
        file << "3 " << F.row(i) << '\n';
    file.close();
}

void face_coloring(pmp::SurfaceMesh& mesh, pmp::FaceProperty<int> f_label) {
    assert(mesh.is_triangle_mesh());
    auto f_color = mesh.face_property<pmp::Color>("f:color");
    size_t max_value = *std::max_element(f_label.data(), f_label.data()+mesh.n_faces());
    Eigen::MatrixX3f color_map(max_value+1, 3);
    color_map.setRandom();
    for(auto f: mesh.faces()){
        if(f_label[f] >= 0)
            f_color[f] = color_map.row(f_label[f]) * 255.0;
    }
}


const float eps = std::numeric_limits<double>::epsilon()*20;

bool greater(const jcv_point& lhs, const jcv_point& rhs) {
    return std::abs(lhs.x - rhs.x) < eps ? ( lhs.y - rhs.y > eps): (lhs.x - rhs.x  > eps); // 2078 5954
//    return lhs.x == rhs.x ? (lhs.y > rhs.y): lhs.x > rhs.x; // 4490 5948
}

void voronoi_cell(MatrixX2dr& S,
                  MatrixX2dr& V, MatrixX2Ir& E) {
    jcv_diagram diagram;
    memset(&diagram, 0, sizeof(jcv_diagram));
    jcv_rect rect{0, 0, 1, 1};
    jcv_diagram_generate(S.rows(), (jcv_point*)(S.data()), &rect, 0, &diagram);
    const jcv_site* sites = jcv_diagram_get_sites(&diagram);

    std::vector<Eigen::Vector2i> edges;
    std::vector<Eigen::Vector3i> triangles;
    std::map<jcv_point, int, decltype(greater)*> V_map(greater);

    S.conservativeResize(diagram.numsites, 2);
    edges.reserve(6*diagram.numsites);
    triangles.reserve(3*diagram.numsites);
    for (int i = 0; i < diagram.numsites; i++){
        S.row(i) = Eigen::RowVector2d(sites[i].p.x, sites[i].p.y);
//        std::cout << i << ' '  << sites[i].index << std::endl;
        for (auto e = sites[i].edges; e; e = e->next){
            if(V_map.find(e->pos[0]) == V_map.end())
                V_map[e->pos[0]] = V_map.size();
            if(V_map.find(e->pos[1]) == V_map.end())
                V_map[e->pos[1]] = V_map.size();
            edges.emplace_back(V_map[e->pos[0]], V_map[e->pos[1]]);
        }
    }

    jcv_diagram_free( &diagram );

    V.resize(V_map.size(), 2);
    for(auto& kv: V_map)
        V.row(kv.second) = Eigen::RowVector2d(kv.first.x, kv.first.y);

    int e_cnt = 0;
    E.resize(edges.size(), 2);
    for (auto & edge : edges) {
        if(edge[0] < edge[1])
            E.row(e_cnt++) = edge;
    }
    E.conservativeResize(e_cnt, 2);


//    std::ofstream file;
//    file.open("output/jcv_voronoi.obj");
//    for (int i = 0; i < V.rows(); i++)
//        file << "v " << V.row(i) << " 0\n";
//    for(int i =0 ; i < edges.size(); i++) {
//        if(edges[i][0] < edges[i][1]) continue;
//        file << "l " << edges[i][0]+1 << ' ' << edges[i][1]+1 << "\n";
//    }
//    file.close();
}



void mesh_voronoi_overlapping(MatrixX2dr& V_tri, MatrixX3Ir& F,
                              MatrixX2dr& V_cell, MatrixX2Ir& E,
                              MatrixX2dr& V_sample,
                              pmp::SurfaceMesh& celled_mesh) {
    MatrixX2dr V(V_tri.rows()+V_cell.rows()+V_sample.rows(),2);
    V.topRows(V_tri.rows()) = V_tri;
    V.middleRows(V_tri.rows(), V_cell.rows()) = V_cell;
    V.bottomRows(V_sample.rows()) = V_sample;

    PyMesh::TriangleWrapper triangle_wrapper;
    triangle_wrapper.set_points(V);
    triangle_wrapper.set_segments(E);
    triangle_wrapper.set_triangles(F);
    triangle_wrapper.set_verbosity(0);
    triangle_wrapper.set_conforming_delaunay(true);
    triangle_wrapper.set_min_angle(5);
    triangle_wrapper.set_split_boundary(false);
    triangle_wrapper.set_exact_arithmetic(true);
    triangle_wrapper.set_max_num_steiner_points(0);
    triangle_wrapper.run();

    Eigen::MatrixXd V2 = triangle_wrapper.get_vertices();
    Eigen::MatrixXi F2 = triangle_wrapper.get_faces();
    celled_mesh.clear();
    for(auto i = 0; i < V2.rows(); i ++)
        celled_mesh.add_vertex(pmp::Point(V2(i, 0), V2(i, 1), 0));

    for(auto i = 0; i < F2.rows(); i++)
        celled_mesh.add_triangle(pmp::Vertex(F2(i,0)),
                                 pmp::Vertex(F2(i,1)),
                                 pmp::Vertex(F2(i,2)));

    MatrixX2Ir E2 = triangle_wrapper.get_edges();
    Eigen::VectorXi E2_marks = triangle_wrapper.get_edge_marks();


    auto e_cell = celled_mesh.edge_property<bool>("e:cell", false);
    for(auto i = 0; i < E2_marks.size(); i++) {
        auto e = celled_mesh.find_edge(pmp::Vertex(E2(i, 0) ), pmp::Vertex(E2(i, 1)));
        e_cell[e] = E2_marks[i];
    }

//    std::ofstream file;
//    file.open("output/v_cell.obj");
//    for (int i = 0; i < V2.rows(); i++)
//        file << "v " << V2.row(i) << " 0\n";
//    for (int i = 0; i < E2_marks.size(); i++){
//        if(E2_marks[i])
//            file << "l " << E2(i, 0)+1 << ' ' << E2(i, 1)+1 << "\n";
//    }
//    file.close();

    //site-region exaction
    auto f_region = celled_mesh.face_property<int>("f:site", -1);
    for(int i = 0 ; i < V_sample.rows(); i++){
        auto v_root = pmp::Vertex(V_tri.rows() + V_cell.rows() + i);
        std::stack<pmp::Face> S;
        for (auto f : celled_mesh.faces(v_root)){
            S.push(f);
            f_region[f] = i;
        }
        while (!S.empty()){
            auto f = S.top();
            S.pop();
            f_region[f] = i;
            for (auto h : celled_mesh.halfedges(f)){
                if (e_cell[celled_mesh.edge(h)])
                    continue;
                auto next_f = celled_mesh.face(celled_mesh.opposite_halfedge(h));
                if (f_region[next_f] == -1){
                    S.push(next_f);
                    f_region[next_f] = i;
                }
            }
        }
    }

    for(auto v: celled_mesh.vertices()) {
        auto diff = celled_mesh.position(v) - pmp::Point(0.5, 0.5, 0);
        if(pmp::norm(diff) > 0.5+1e-3)
            celled_mesh.delete_vertex(v);
    }

    celled_mesh.garbage_collection();

//    face_coloring(celled_mesh, f_region);
//    write_colored_off("output/colored.off", celled_mesh);
}

void triangulation(const MatrixX2dr& V, const MatrixX2Ir& E, Eigen::MatrixX2d& V2, Eigen::MatrixX3i& F2) {
    PyMesh::TriangleWrapper triangle_wrapper;
    triangle_wrapper.set_points(V);
    triangle_wrapper.set_segments(E);
    triangle_wrapper.set_verbosity(0);
    triangle_wrapper.set_conforming_delaunay(true);
    triangle_wrapper.set_split_boundary(false);
    triangle_wrapper.set_exact_arithmetic(true);
    triangle_wrapper.set_max_num_steiner_points(0);
    triangle_wrapper.run();

    V2 = triangle_wrapper.get_vertices();
    F2 = triangle_wrapper.get_faces();
}


VoronoiCellMesh::VoronoiCellMesh(const pmp::SurfaceMesh& mesh): pmp::SurfaceMesh(mesh) {
    // parameterize
    pmp::SurfaceParameterization param_(*this);
    param_.lscm();
    auto v_tex = get_vertex_property<pmp::TexCoord>("v:tex");
    for(auto v: vertices()) {
        auto diff = v_tex[v] - pmp::TexCoord (0.5, 0.5);
        if(pmp::norm(diff) > 0.5+1e-3)
            v_tex[v] = pmp::TexCoord (0.5, 0.5)+diff/pmp::norm(diff);
    }

    auto v_coord = vertex_property<pmp::Point>("v:coord3");
    v_density_ = vertex_property<double>("v:density");

    V_tex.resize(mesh.n_vertices(), 2);
    for(auto v: mesh.vertices()){
        v_coord[v] = position(v);
        position(v) = pmp::Point(v_tex[v][0], v_tex[v][1], 0);
        V_tex.row(v.idx()) = Eigen::Vector2d(v_tex[v][0], v_tex[v][1]);
    }
    for(auto v : mesh.vertices())
        v_density_[v] = pmp::voronoi_area(mesh, v)/pmp::voronoi_area(*this, v);
    remove_vertex_property(v_tex);


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
    kd_tree_ = std::make_shared<pmp::TriangleKdTree>(*this);
}

double VoronoiCellMesh::density(pmp::Point p) const {
    auto neighbor = kd_tree_->nearest(p);
    auto fvit = vertices(neighbor.face);
    pmp::Vertex uvw[3] = {*fvit, *++fvit, *++fvit};
    pmp::Point bc = pmp::barycentric_coordinates(neighbor.nearest, position(uvw[0]),
                                 position(uvw[1]), position(uvw[2]));
    return bc_interp(v_density_, neighbor.face, bc);
}



void write_sample_off(std::string file_path, MatrixX2dr& V)  {
    std::ofstream file(file_path);
    file << "OFF\n" << V.rows() <<" 0 0\n";
    for (size_t i = 0; i < V.rows(); i++)
        file << V.row(i) << " 0\n";
    file.close();
}

///！ 依据密度的随机拒绝采样，密度越大被拒绝的概率越低
void VoronoiCellMesh::random_density_sampling(size_t n_sample) {
    auto v3d_ = get_vertex_property<pmp::Point>("v:coord3");
    auto v2d_ = get_vertex_property<pmp::Point>("v:point");
    auto f_density = face_property<double>("f:density");
    // step1. calc face density，for O(1) density query
    for(auto f: faces()) {
        auto v0 = pmp::Vertex(F(f.idx(), 0));
        auto v1 = pmp::Vertex(F(f.idx(), 1));
        auto v2 = pmp::Vertex(F(f.idx(), 2));
        f_density[f] = pmp::triangle_area(v3d_[v0], v3d_[v1], v3d_[v2]);
        f_density[f] /= pmp::triangle_area(v2d_[v0], v2d_[v1], v2d_[v2]);
        f_density[f] = std::log(10+f_density[f]);
    }

    double max_density = *std::max_element(f_density.data(), f_density.data() + n_faces());
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> real_dis(0, max_density);
    V_site_.resize(n_sample, 2);
    for(size_t i = 0; i < n_sample; ) {
        auto f = pmp::Face(rand()%n_faces());
        double d = real_dis(gen);
        if(d > f_density[f]) continue;
        Eigen::Vector3f bc = Eigen::Vector3f::Random().cwiseAbs();
        pmp::Point p = bc_interp(v2d_, f, bc/bc.sum());
        V_site_.row(i++) = Eigen::RowVector2d(p[0], p[1]);
    }

    remove_face_property(f_density);
}

void VoronoiCellMesh::triangular_voronoi_overlapping(pmp::SurfaceMesh& cell_mesh_, pmp::FaceProperty<int>& f_region) {
    MatrixX2dr V_cell;
    MatrixX2Ir E_cell, E;
    voronoi_cell(V_site_, V_cell, E_cell);

    E.resize(E_cell.rows()+E_bnd.rows(), 2);
    E.topRows(E_cell.rows()) = E_cell.array() + n_vertices();
    E.bottomRows(E_bnd.rows()) = E_bnd;

    mesh_voronoi_overlapping(V_tex, F, V_cell, E, V_site_, cell_mesh_);
    f_region = cell_mesh_.get_face_property<int>("f:site");
    assert(f_region);
}

void VoronoiCellMesh::lloyd_relaxation(pmp::SurfaceMesh& cell_mesh_, pmp::FaceProperty<int>& f_region) {
    auto cell_v_density = cell_mesh_.vertex_property<double>("v:density");
    for(auto v: cell_mesh_.vertices())
        cell_v_density[v] = density(cell_mesh_.position(v));

    MatrixX2dr site_centroid = V_site_ * 1e-6;
    Eigen::VectorXd mass = Eigen::VectorXd::Ones(V_site_.rows()) * 1e-6;
    for(auto f: cell_mesh_.faces()) {
        if(f_region[f] == -1) continue;
        double face_area = pmp::triangle_area(cell_mesh_, f);
        for(auto v: cell_mesh_.vertices(f)) {
            auto p = cell_mesh_.position(v) * cell_v_density[v] * face_area;
            site_centroid.row(f_region[f]) += Eigen::RowVector2d(p[0], p[1]);
            mass[f_region[f]] += cell_v_density[v] * face_area;
        }
    }
    V_site_ = site_centroid.array().colwise() / mass.array();
}

void VoronoiCellMesh::lift_up(Eigen::MatrixX3d& V3d, Eigen::MatrixX3i& F) {
    auto v_reid = vertex_property<int>("v:bnd", -1);
    size_t bnd_size = 0;
    for(auto v: vertices())
        if(is_boundary(v))
            v_reid[v] = bnd_size++;
    Eigen::MatrixX2d V2d(bnd_size + V_site_.rows(), 2);
    for(auto v: vertices())
        if(is_boundary(v))
            V2d.row(v_reid[v]) = V_tex.row(v.idx());
    V2d.bottomRows(V_site_.rows()) = V_site_;

    MatrixX2Ir Edge(E_bnd.rows(), 2);
    for(size_t i = 0 ; i < E_bnd.rows(); i++) {
        Edge(i, 0) = v_reid[pmp::Vertex(E_bnd(i, 0))];
        Edge(i, 1) = v_reid[pmp::Vertex(E_bnd(i, 1))];
    }
    triangulation(V2d, Edge, V2d, F);


    V3d.resize(V2d.rows(), 3);
    auto v_coord = get_vertex_property<pmp::Point>("v:coord3");
    for(size_t i = 0; i < V2d.rows(); i++) {
        pmp::Point p(V2d(i, 0), V2d(i, 1), 0);
        auto root_face = kd_tree_->nearest(p).face;
        auto fvit = vertices(root_face);
        pmp::Vertex uvw[3] = {*fvit, *++fvit, *++fvit};
        pmp::Point bc = pmp::barycentric_coordinates(p, position(uvw[0]),
                            position(uvw[1]), position(uvw[2]));
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

