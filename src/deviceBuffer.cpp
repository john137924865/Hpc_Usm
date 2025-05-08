#pragma once

#include <sycl/sycl.hpp>
#include <iostream>
#include "myBuffer.cpp"

using namespace sycl;

template <typename T>
class deviceBuffer {

    private:
        T* device_data;
        T* host_data;
        size_t size;
        queue* q;
        access::mode mode;

    public:

        deviceBuffer(myBuffer<T>& buf, access::mode mode = access::mode::read_write) :
            device_data(buf.get_device_data()), host_data(buf.get_host_data()), size(buf.get_size()), q(buf.get_queue()), mode(mode) {}

        void sync() {
            if (mode != access::mode::read) {
                (*q).memcpy(host_data, device_data, sizeof(T) * size).wait();
            }
        }

        T* get_device_data() {
            return device_data;
        }

};