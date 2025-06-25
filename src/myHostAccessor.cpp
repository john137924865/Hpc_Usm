#pragma once

#include <sycl/sycl.hpp>
#include <iostream>
#include "myBuffer.cpp"

template <typename T>
class myHostAccessor {

    private:
        T* host_data;
        sycl::access::mode mode;

    public:

        myHostAccessor(myBuffer<T>& buf, sycl::access::mode mode = sycl::access::mode::read_write) :
            host_data(buf.get_host_data()), mode(mode) {}

        T& operator[](size_t index) {
            return host_data[index];
        }

};