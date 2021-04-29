//
// Created by pupa on 2021/4/20.
//

#pragma once

#include <pmp/BoundingBox.h>
#include <pmp/SurfaceMesh.h>

namespace pmp {

class BVHTriangle
{
private:

    // Node of the tree: contains parent, children and splitting plane
    struct Node
    {
        Node() : left_child(nullptr), right_child(nullptr){};

        ~Node() {
            delete left_child;
            delete right_child;
        }

        unsigned char axis;
        Scalar split;
        std::vector<Face> faces;
        BoundingRect brect;
        Node* left_child;
        Node* right_child;
    };
    

public: 
    //! Construct with mesh.
    BVHTriangle(SurfaceMesh& mesh, VertexProperty< TexCoord >& VProp,
               size_t max_faces = 10, size_t max_depth = 30);

    //! destructor
    ~BVHTriangle() { delete root_; }

    std::vector<Face> intersection(const BoundingRect& b_rect);

private:
    // Recursive part of build()
    size_t build_recurse(Node* node, size_t max_handles,
                               size_t depth);

    void intersect_recurse(Node* node, const BoundingRect& b_rect, std::vector<Face>& faces);

    Node* root_;
    SurfaceMesh& mesh_;
    VertexProperty< TexCoord > v_prop_;
    FaceProperty< BoundingRect > f_brect_;
};


}



bool inline to_left_test(const pmp::TexCoord& p, const pmp::TexCoord& p1, const pmp::TexCoord& p2) {
    return (p2[1] - p1[1]) * p[0] + (p1[0] - p2[0]) * p[1] + (p2[0] * p1[1] - p1[0] * p2[1]) < 0;
}

pmp::TexCoord inline intersection(const pmp::TexCoord& cp1, pmp::TexCoord cp2, const pmp::TexCoord& s, pmp::TexCoord e) {
    pmp::TexCoord dc{ cp1[0] - cp2[0], cp1[1] - cp2[1]};
    pmp::TexCoord dp{ s[0] - e[0], s[1] - e[1]};

    float n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0];
    float n2 = s[0] * e[1] - s[1] * e[0];
    float n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0]);

    return pmp::TexCoord{ (n1 * dp[0] - n2 * dc[0]) * n3, (n1 * dp[1] - n2 * dc[1]) * n3};
}


pmp::TexCoord inline polygon_centroid(std::vector<pmp::TexCoord>& poly){
    pmp::TexCoord sum = std::accumulate(poly.begin(), poly.end(), pmp::TexCoord(0, 0));
    return sum/poly.size();
}

double inline polygon_area(std::vector<pmp::TexCoord>& poly){
    double area = 0;
    for(size_t i = 0; i < poly.size(); i++) {
        auto& v1 = poly[(i+1)%poly.size()];
        area += poly[i][0] * v1[1] - poly[i][1]*v1[0];
    }
    return std::abs(area)/2;
}

// Sutherland-Hodgman clipping
void static clipping_sutherland_hodgman(std::vector<pmp::TexCoord>& subj,
                                        std::vector<pmp::TexCoord>& clip,
                                        std::vector<pmp::TexCoord>& solution) {
    pmp::TexCoord cp1, cp2, s, e;

    std::vector<pmp::TexCoord> input;
    solution = subj;

    for(size_t j = 0; j < clip.size(); j++) {
        std::swap(input, solution);
        solution.clear();

        // get clipping polygon edge
        cp1 = clip[j]; cp2 = clip[(j + 1) % clip.size()];

        for(size_t i = 0; i < input.size(); i++) {
            // get subject polygon edge
            s = input[i]; e = input[(i + 1) % input.size()];

            // Case 1: Both vertices are inside:
            // Only the second vertex is added to the output list
            if(to_left_test(s, cp1, cp2) && to_left_test(e, cp1, cp2))
                solution.push_back(e);
                // Case 2: First vertex is outside while second one is inside:
                // Both the point of intersection of the edge with the clip boundary
                // and the second vertex are added to the output list
            else if(!to_left_test(s, cp1, cp2) && to_left_test(e, cp1, cp2)){
                solution.push_back( intersection(cp1, cp2, s, e) );
                solution.push_back(e);
            }
                // Case 3: First vertex is inside while second one is outside:
                // Only the point of intersection of the edge with the clip boundary
                // is added to the output list
            else if(to_left_test(s, cp1, cp2) && !to_left_test(e, cp1, cp2))
                solution.push_back(  intersection(cp1, cp2, s, e) );
                // Case 4: Both vertices are outside
                // No vertices are added to the output list
            else if(!to_left_test(s, cp1, cp2) && !to_left_test(e, cp1, cp2))
                ;
        }

    }
}