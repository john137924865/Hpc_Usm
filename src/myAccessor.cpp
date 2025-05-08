#pragma once

#include <sycl/sycl.hpp>
#include <iostream>
#include "deviceBuffer.cpp"

using namespace sycl;

template <typename T>
class myAccessor {

    private:
        T* device_data;

    public:

        myAccessor(deviceBuffer<T>& dev_buf) : device_data(dev_buf.get_device_data()) {}

        T& operator[](size_t index) const {
            return device_data[index];
        }

};