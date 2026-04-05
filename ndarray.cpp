#include <algorithm>
#include <cstddef>
#include <endian.h>
#include <functional>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>

#include "ndarray.hpp"

template<typename T>
std::vector<size_t> ndarray<T>::compute_strides(const std::vector<size_t> &shape) {
    auto n = shape.size();
    std::vector<size_t> strides(n, 1);
    for (size_t i = n - 2; i >= 0; --i) {
       strides[i] = strides[i + 1] * shape[i + 1]; 
    }
    return strides;
}
    
template<typename T>
size_t ndarray<T>::total_size(const std::vector<size_t>& shape) {
    return std::accumulate(shape.begin(), shape.end(), size_t{1}, std::multiplies<size_t>{});
}

template<typename T>
void ndarray<T>::iterate(const std::vector<size_t>& shape, std::function<void(const std::vector<size_t>&)> fn) {
    std::vector<size_t> idx(shape.size(), 0);
    while (true) {
        fn(idx);
        int d = shape.size() - 1;
        while (d >= 0) {
            if (++idx[d] < shape[d]) break;
            idx[d--] = 0;
        }
        if (d < 0) break;
    }
}
template<typename T>
template<typename Op>
ndarray<T> &ndarray<T>::apply_inplace(const ndarray<T> &rhs, Op op) {
    if (shape_ != rhs.shape_)
        throw std::invalid_argument("ndarray: shape mismatch in element-wise operation");
    std::transform(data_.begin(), data_.end(), rhs.data_.begin(), data_.begin(), op);
    return *this;
}

template<typename T>
template <typename Op>
ndarray<T> ndarray<T>::apply(const ndarray& rhs, Op op) const {
    ndarray result = *this;
    result.apply_inplace(rhs, op);
    return result;
}

template<typename T>
void ndarray<T>::print_recursive(std::ostream& os, std::vector<size_t>& idx, size_t dim) const {
    os << "[";
    for (size_t i = 0; i < shape(dim); ++i) {
        idx[dim] = i;
        if (dim == ndim() - 1) {
            os << operator()(idx);
        } else {
            print_recursive(os, idx, dim + 1);
        }
        if (i + 1 < shape(dim)) {
            os << ",\n" << std::string(dim + 1, ' ');
        }
    }
    os << "]";
}

template<typename T>
ndarray<T>::ndarray(const std::vector<size_t> &shape, T init) 
    : shape_(shape), strides_(compute_strides(shape)), offset_(0) {
    data_.resize(total_size(shape), init);
}

template<typename T>
ndarray<T>::ndarray(const std::vector<size_t> &shape, std::vector<T> data) 
    : shape_(shape)
    , data_(std::move(data))
    , strides_(compute_strides(shape))
    , offset_(0)
{
    if (data_.size() != total_size(shape_)) {
        throw std::invalid_argument("ndarray: data size does not match shape");
    }
}

template<typename T>
ndarray<T> ndarray<T>::fill(const std::vector<size_t> &shape, T value) {
    return ndarray(shape, value);
}

template<typename T>
ndarray<T> ndarray<T>::zeros(const std::vector<size_t> &shape) {
    return fill(shape, T{0});
}

template<typename T>
ndarray<T> ndarray<T>::ones(const std::vector<size_t> &shape) {
    return fill(shape, T{1});
}

template<typename T>
ndarray<T> ndarray<T>::arange(T start, T stop, T step) {
    std::vector<T> data;
    while (start < stop) {
        data.push_back(start);
        start += step;
    }
    return ndarray({data.size()}, std::move(data));
}

template<typename T>
ndarray<T> ndarray<T>::slice(size_t dim, size_t start, size_t stop) const {
    std::vector<size_t> new_shape = shape_;
    new_shape[dim] = stop - start;
    ndarray result(new_shape);
    iterate(new_shape, [this, &result, start, dim](const std::vector<size_t> &idx){
        auto input_idx = idx;
        input_idx[dim] = idx[dim] + start;
        result(idx) = operator()(input_idx);
    }); 
    return result;
}

template<typename T>
ndarray<T> ndarray<T>::transpose() const {
    if (ndim() != 2) {
        throw std::logic_error("ndarray::transpose: only 2D arrays are supported");
    }
    std::vector<size_t> new_shape(ndim());
    std::reverse_copy(shape_.begin(), shape_.end(), new_shape.begin());
    ndarray result(new_shape);
    iterate(new_shape, [this, &result](const std::vector<size_t> &idx){
        auto input_idx = idx;
        std::reverse(input_idx.begin(), input_idx.end());
        result(idx) = operator()(input_idx); 
    });
    return result;
}

template<typename T>
ndarray<T> ndarray<T>::matmul(const ndarray<T> &rhs) const {
    if (ndim() != 2) {
        throw std::logic_error("ndarray::matmul: only 2D arrays are supported");
    }
    if (shape(1) != rhs.shape(0)) {
        throw std::invalid_argument("ndarray::matmul: inner dimensions must agree, got "
            + std::to_string(shape(1)) + " and " + std::to_string(rhs.shape(0)));
    }
    std::vector<size_t> new_shape({shape(0), rhs.shape(1)});
    ndarray result(new_shape);
    for (size_t i = 0; i < shape(0); ++i) {
        for (size_t j = 0; j < rhs.shape(1); ++j) {
            for (size_t k = 0; k < shape(1); ++k) {
                result(i, j) += operator()(i, k) * rhs(k, j);
            }
        }
    }
    return result;
}

template<typename T>
T &ndarray<T>::operator()(const std::vector<size_t> &idx) {
    if (idx.size() != ndim()) {
        throw std::out_of_range("ndarray::operator(): wrong number of indices: expected"
            + std::to_string(ndim()) + " got" + std::to_string(idx.size()));
    }
    return data_[std::inner_product(idx.begin(), idx.end(), strides_.begin(), offset_)];
}

template<typename T>
const T& ndarray<T>::operator()(const std::vector<size_t>& idx) const {
    if (idx.size() != ndim()) {
        throw std::out_of_range("ndarray::operator(): wrong number of indices: expected"
            + std::to_string(ndim()) + " got" + std::to_string(idx.size()));
    }
    return data_[std::inner_product(idx.begin(), idx.end(), strides_.begin(), offset_)];
}

template<typename T>
template<typename... Idx>
T &ndarray<T>::operator()(Idx... idx) {
    return operator()({static_cast<size_t>(idx)...});
}

template<typename T>
ndarray<T> &ndarray<T>::reshape(const std::vector<size_t> &new_shape) {
    auto total = total_size(new_shape);
    if (total != size()) {
        throw std::invalid_argument("ndarray::reshape: total size must not change, expected "
            + std::to_string(size()) + " got " + std::to_string(total));
    }
    shape_ = new_shape;
    strides_ = compute_strides(new_shape);
    return *this;
}

template<typename T>
ndarray<T> &ndarray<T>::operator+=(const ndarray &rhs) {
    return apply_inplace(rhs, std::plus<T>{});
}

template<typename T>
ndarray<T> &ndarray<T>::operator-=(const ndarray &rhs) {
    return apply_inplace(rhs, std::minus<T>{});
}

template<typename T>
ndarray<T> &ndarray<T>::operator*=(const ndarray &rhs) {
    return apply_inplace(rhs, std::multiplies<T>{});
}

template<typename T>
ndarray<T> &ndarray<T>::operator/=(const ndarray &rhs) {
    return apply_inplace(rhs, std::divides<T>{});
}

template<typename T>
ndarray<T> ndarray<T>::operator+(const ndarray<T> &rhs) const {
    return apply(rhs, std::plus<T>{});
}

template<typename T>
ndarray<T> ndarray<T>::operator-(const ndarray<T> &rhs) const {
    return apply(rhs, std::minus<T>{});
}

template<typename T>
ndarray<T> ndarray<T>::operator*(const ndarray<T> &rhs) const {
    return apply(rhs, std::multiplies<T>{});
}

template<typename T>
ndarray<T> ndarray<T>::operator/(const ndarray<T> &rhs) const {
    return apply(rhs, std::divides<T>{});
}

template<typename T>
std::ostream &operator<<(std::ostream &os, const ndarray<T> &a) {
    std::vector<size_t> idx(a.ndim(), 0);
    a.print_recursive(os, idx, 0);
    return os;
}

template<typename T>
size_t ndarray<T>::ndim() const {
    return shape_.size();
}

template<typename T>
size_t ndarray<T>::size() const {
    return data_.size();
}

template<typename T>
const std::vector<size_t> &ndarray<T>::shape() const {
    return shape_;
}

template<typename T>
size_t ndarray<T>::shape(size_t i) const {
    if (i >= shape_.size()) {
        throw std::out_of_range("ndarray::shape: dim out of range");
    }
    return shape_[i];
}


