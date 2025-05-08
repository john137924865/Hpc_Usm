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

        T* get_device_data() {
            return device_data;
        }
        
        T* get_host_data() {
            return host_data;
        }

        size_t get_size() {
            return size;
        }

        queue* get_queue() {
            return q;
        }

};