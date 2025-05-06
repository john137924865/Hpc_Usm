#pragma once

#include <sycl/sycl.hpp>
#include <iostream>

using namespace sycl;

template <typename T>
class myBuffer {

    private:
        T* device_data;
        T* host_data;
        size_t size;
        queue* q;

    public:

        // Costruttore senza inizializzazione;
        myBuffer(queue& q, size_t n) : size(n) {
            this->device_data = static_cast<T*>(malloc_device(sizeof(T) * size, q));
            this->host_data = new T[size];
            this->q = &q;
        }

        void copy_host_to_device() {
            (*this->q).memcpy(device_data, host_data, sizeof(T) * size).wait();
        }

        void copy_device_to_host() {
            (*this->q).memcpy(host_data, device_data, sizeof(T) * size);
        }

        T* get_host_data() {
            return host_data;
        }

        T* get_device_data() {
            return device_data;
        }

};