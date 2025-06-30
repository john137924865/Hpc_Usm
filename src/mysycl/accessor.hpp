#pragma once

#include <sycl/sycl.hpp>
#include <iostream>

namespace mysycl {

    template <typename T>
    class accessor {

    private:
        T* device_data;
        sycl::access::mode mode;

    public:

        accessor(mysycl::buffer<T>& buf, sycl::handler& h, sycl::access::mode mode = sycl::access::mode::read_write) :
            device_data(buf.get_device_data()), mode(mode) {
            buf.check_mode(mode, h);
        }

        T& operator[](size_t index) const {
            return device_data[index];
        }

    };
}