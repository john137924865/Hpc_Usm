#pragma once

#include <sycl/sycl.hpp>
#include <iostream>
#include "myBuffer.cpp"

using namespace sycl;

template <typename T>
class myAccessor {

    private:
        T* device_data;
        access::mode mode;

    public:

        myAccessor(myBuffer<T>& buf, access::mode mode = access::mode::read_write) :
            device_data(buf.get_device_data()), mode(mode) {}

        T& operator[](size_t index) const {
            return device_data[index];
        }

};