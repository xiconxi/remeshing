//
// Created by pupa on 2021/4/20.
//

#include "BVHTriangle.h"

using namespace pmp;

BVHTriangle::BVHTriangle(SurfaceMesh& mesh, VertexProperty< TexCoord >& VProp,
           size_t max_faces,size_t max_depth)
    :mesh_(mesh), v_prop_(VProp){
    assert(mesh.is_triangle_mesh());
    // init
    root_ = new Node();

    f_brect_ = mesh.face_property< BoundingRect >("f:brect");
    root_->faces.reserve(mesh.n_faces());
    for (auto f : mesh.faces()) {
        auto vfit = mesh.vertices(f);
        f_brect_[f] += v_prop_[*vfit];
        f_brect_[f] += v_prop_[*++vfit];
        f_brect_[f] += v_prop_[*++vfit];

        root_->faces.push_back(f);
        root_->brect += f_brect_[f];
    }

    std::cout << root_->brect.min() << '\t' << root_->brect.max() << std::endl;
    for(auto v: mesh.vertices())
        root_->brect += v_prop_[v];

    std::cout << root_->brect.min() << '\t' << root_->brect.max() << std::endl;

    // call recursive helper
    build_recurse(root_, max_faces, max_depth);
}

// Recursive part of build()
size_t BVHTriangle::build_recurse(Node* node, size_t max_faces, size_t depth) {
    // should we stop at this level ?
    if ((depth == 0) || (node->faces.size() <= max_faces))
        return depth;

    BoundingRect brect = node->brect;

    // split longest side of bounding box
    TexCoord bb = brect.max() - brect.min();
    size_t axis = std::max_element(bb.data(), bb.data()+bb.size()) - bb.data();

    Scalar split = brect.center()[axis];

    // create children, may allocate some empty leaves~
    auto* left = new Node();
    left->faces.reserve(node->faces.size() / 2);
    auto* right = new Node();
    right->faces.reserve(node->faces.size() / 2);

    // partition for left and right child
    for (auto& f: node->faces) {
        if(f_brect_[f].min()[axis] > split ){
            right->brect += f_brect_[f];
            right->faces.push_back(f);
        }else {
            left->brect += f_brect_[f];
            left->faces.push_back(f);
        }
    }

    node->faces.clear();
    node->faces.shrink_to_fit();
    // store internal data
    node->axis = axis;
    node->split = split;
    node->left_child = left;
    node->right_child = right;

    // recurse to childen
    int depthLeft = build_recurse(node->left_child, max_faces, depth - 1);
    int depthRight = build_recurse(node->right_child, max_faces, depth - 1);

    return std::min(depthLeft, depthRight);
}

std::vector<Face> BVHTriangle::intersection(const BoundingRect& b_rect) {
    std::vector<Face> results;

    if(root_->brect.is_intersected(b_rect))
        intersect_recurse(root_, b_rect, results);
    return results;
}

void BVHTriangle::intersect_recurse(Node* node, const BoundingRect& b_rect, std::vector<Face>& faces) {
    if( !node->faces.empty() )
        for(auto f: node->faces)
            if (f_brect_[f].is_intersected(b_rect))
                faces.push_back(f);

    if(node->left_child && node->left_child->brect.is_intersected(b_rect))
        intersect_recurse(node->left_child, b_rect, faces);

    if(node->right_child && node->right_child->brect.is_intersected(b_rect))
        intersect_recurse(node->right_child, b_rect, faces);
}

