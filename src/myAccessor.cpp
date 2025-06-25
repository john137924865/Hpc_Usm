#pragma once

#include <sycl/sycl.hpp>
#include <iostream>
#include "myBuffer.cpp"

template <typename T>
class myAccessor {

    private:
        T* device_data;
        sycl::access::mode mode;

    public:

        myAccessor(myBuffer<T>& buf, sycl::handler& h, sycl::access::mode mode = sycl::access::mode::read_write) : 
                device_data(buf.get_device_data()), mode(mode) {
            buf.check_mode(mode, h);
        }

        T& operator[](size_t index) const {
            return device_data[index];
        }

};