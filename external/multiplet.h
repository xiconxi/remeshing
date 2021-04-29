//
// Created by pupa on 2021/4/13.
//

#pragma once

#include <Eigen/Dense>

template<typename T, int Dim>
class Multiplet {
public:
    using Data = Eigen::Matrix<T, Dim, 1>;

    Multiplet() = default;
    Multiplet(const Data&& items) : m_ori_data(items) {
        init_data();
    }

    virtual ~Multiplet() = default;

public:

    inline bool operator==(const Multiplet<T, Dim>& other) const {
        return m_data == other.m_data;
    }

    bool operator<(const Multiplet<T, Dim>& other) const {
        for (size_t i=0; i<Dim; i++) {
            if (m_data[i] < other.m_data[i]) return true;
            else if (m_data[i] > other.m_data[i]) return false;
        }
        // this == other
        return false;
    }

    inline const Data& get_data() const { return m_data; }
    inline const Data& get_ori_data() const { return m_ori_data; }

    virtual int hash() const =0;

private:
    void init_data() {
        // Sort m_ori_data into descending order.
        m_data = m_ori_data;
        std::sort(m_data.data(), m_data.data() + Dim, std::greater<T>());
    }

protected:
    Data m_data;
    Data m_ori_data;
};



class Duplet : public Multiplet<int, 2> {
public:
    using Parent = Multiplet<int, 2>;
    Duplet(int v1=0, int v2=0) {
        Parent::m_ori_data << v1, v2;
        if (v1 >= v2) {
            Parent::m_data << v1, v2;
        } else {
            Parent::m_data << v2, v1;
        }
    }
    virtual ~Duplet() = default;

    inline virtual int hash() const {
        // Magic primes (p1, p2) from the following paper.
        // "Optimized Spacial Hashing for Collision Detection of
        // Deformable Objects" by Teschner et al. (VMV 2003)
        constexpr int p1 = 73856093;
        constexpr int p2 = 19349663;
        return (Parent::m_data.coeff(0)*p1) ^ (Parent::m_data.coeff(1)*p2);
    }
};

template<class DataType>
struct HashFunc {
    int operator()(const DataType& key) const {
        return key.hash();
    }
};