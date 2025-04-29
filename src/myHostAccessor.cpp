#pragma once

#include <sycl/sycl.hpp>
#include <iostream>
#include "myBuffer.cpp"

using namespace sycl;

template <typename T>
class myHostAccessor {
private:
    T* host_data;

public:

    myHostAccessor(myBuffer<T>& buf) : host_data(buf.get_host_data()) {}

    T& operator[](size_t index) {
        return host_data[index];
    }

};