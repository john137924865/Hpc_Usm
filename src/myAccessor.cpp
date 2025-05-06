#pragma once

#include <sycl/sycl.hpp>
#include <iostream>
#include "myBuffer.cpp"

using namespace sycl;

template <typename T>
class myAccessor {

    private:
        T* device_data;

    public:

        myAccessor(myBuffer<T>& buf) : device_data(buf.get_device_data()) {}

        T& operator[](size_t index) const {
            return device_data[index];
        }

};