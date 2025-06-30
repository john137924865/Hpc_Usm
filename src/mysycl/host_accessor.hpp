#pragma once

#include <sycl/sycl.hpp>
#include <iostream>

namespace mysycl {

    template <typename T>
    class host_accessor {

    private:
        T* host_data;
        sycl::access::mode mode;

    public:

        host_accessor(mysycl::buffer<T>& buf, sycl::access::mode mode = sycl::access::mode::read_write) :
            host_data(buf.get_host_data()), mode(mode) {
        }

        T& operator[](size_t index) {
            return host_data[index];
        }

    };
}