//
// Created by pupa on 2021/4/27.
//
#include "MeshUtils.h"
#include <fstream>

#define JCV_REAL_TYPE double
#define JC_VORONOI_IMPLEMENTATION
#include "jc_voronoi.h"
#include <fstream>
#include <memory>
#include <stack>
#include <queue>
#include <unordered_map>
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

void voronoi_cell(std::vector<Eigen::RowVector2d>& S,
                  Eigen::MatrixX2d& V, Eigen::MatrixX2i& E) {
    jcv_diagram diagram;
    memset(&diagram, 0, sizeof(jcv_diagram));
    jcv_diagram_generate(S.size(), (jcv_point*)(S.data()), 0, 0, &diagram);
    const jcv_site* sites = jcv_diagram_get_sites(&diagram);

    std::vector<Eigen::Vector2i> edges;
    std::vector<Eigen::Vector3i> triangles;
    std::map<jcv_point, int, decltype(greater)*> V_map(greater);

    S.resize(diagram.numsites);
    edges.reserve(6*diagram.numsites);
    triangles.reserve(3*diagram.numsites);
    for (int i = 0; i < diagram.numsites; i++){
        S[i] = Eigen::RowVector2d(sites[i].p.x, sites[i].p.y);
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



void mesh_voronoi_overlapping(Eigen::MatrixX2d& V_tri, Eigen::MatrixX3i& F,
                              Eigen::MatrixX2d& V_cell, Eigen::MatrixX2i& E,
                              Eigen::MatrixX2d& V_sample,
                              pmp::SurfaceMesh& celled_mesh) {
    Eigen::MatrixX2d V(V_tri.rows()+V_cell.rows()+V_sample.rows(),2);
    V.topRows(V_tri.rows()) = V_tri;
    V.middleRows(V_tri.rows(), V_cell.rows()) = V_cell;
    V.bottomRows(V_sample.rows()) = V_sample;

    PyMesh::TriangleWrapper triangle_wrapper;
    triangle_wrapper.set_points(V);
    triangle_wrapper.set_segments(E);
    triangle_wrapper.set_triangles(F);
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

    Eigen::MatrixX2i E2 = triangle_wrapper.get_edges();
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

void triangulation(const Eigen::MatrixX2d& V, const Eigen::MatrixX2i& E, Eigen::MatrixX2d& V2, Eigen::MatrixX3i& F2) {
    PyMesh::TriangleWrapper triangle_wrapper;
    triangle_wrapper.set_points(V);
    triangle_wrapper.set_segments(E);
    triangle_wrapper.set_conforming_delaunay(true);
    triangle_wrapper.set_split_boundary(false);
    triangle_wrapper.set_exact_arithmetic(true);
    triangle_wrapper.set_max_num_steiner_points(0);
    triangle_wrapper.run();

    V2 = triangle_wrapper.get_vertices();
    F2 = triangle_wrapper.get_faces();
}