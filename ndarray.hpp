#ifndef NDARRAY_HPP
#define NDARRAY_HPP

#include <cstddef>
#include <functional>
#include <ostream>
#include <vector>

template<typename T>
class ndarray {
    std::vector<T> data_;
    std::vector<size_t> shape_;
    std::vector<size_t> strides_;
    size_t offset_;

    static std::vector<size_t> compute_strides(const std::vector<size_t> &shape);
    
    static size_t total_size(const std::vector<size_t> &shape);

    static void iterate(const std::vector<size_t> &shape, std::function<void(const std::vector<size_t>&)> fn);

    template<typename Op>
    ndarray &apply_inplace(const ndarray &rhs, Op op);

    template<typename Op>
    ndarray apply(const ndarray &rhs, Op op) const;
    
    void print_recursive(std::ostream& os, std::vector<size_t>& idx, size_t dim) const;

public:
    ndarray(const std::vector<size_t> &shape, T init = T{});

    ndarray(const std::vector<size_t> &shape, std::vector<T> data);
    
    static ndarray fill(const std::vector<size_t> &shape, T value);
    static ndarray zeros(const std::vector<size_t> &shape);
    static ndarray ones(const std::vector<size_t> &shape);
    static ndarray arange(T start, T stop, T step = T{1});
    ndarray slice(size_t dim, size_t start, size_t stop) const;
    ndarray transpose() const;
    ndarray matmul(const ndarray &rhs) const;

    T &operator()(const std::vector<size_t> &idx);
    const T& operator()(const std::vector<size_t>& idx) const;

    template<typename... Idx>
    T &operator()(Idx... idx);

    ndarray &reshape(const std::vector<size_t> &new_shape);

    ndarray &operator+=(const ndarray &rhs);
    ndarray &operator-=(const ndarray &rhs);
    ndarray &operator*=(const ndarray &rhs);
    ndarray &operator/=(const ndarray &rhs);

    ndarray operator+(const ndarray &rhs) const;
    ndarray operator-(const ndarray &rhs) const;
    ndarray operator*(const ndarray &rhs) const;
    ndarray operator/(const ndarray &rhs) const;

    friend std::ostream &operator<<(std::ostream &os, const ndarray &a);

    size_t ndim() const;
    size_t size() const;
    const std::vector<size_t> &shape() const;
    size_t shape(size_t i) const;
};

#endif // !NDARRAY_HPP
