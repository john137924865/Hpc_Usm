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

    // Costruttore per dati non inizializzati, alloco senza inizializzazione;
    myBuffer(queue& q, size_t n) : size(n) {
        this->q = &q;
        device_data = malloc_device<T>(size, q);
        host_data = new T[size];
    }

    auto copy_host_to_device(queue& q) {
        return q.memcpy(device_data, host_data, sizeof(T) * size);
    }

    auto copy_host_to_device(queue& q, event& e) {
        return q.submit([&](handler& h) {
            h.depends_on(e);
            q.memcpy(device_data, host_data, sizeof(T) * size);
        });
    }

    auto copy_device_to_host(queue& q, event& e) {
        return q.submit([&](handler& h) {
            h.depends_on(e);
            h.memcpy(host_data, device_data, sizeof(T) * size);
        });
    }

    T* get_host_data() {
        return host_data;
    }

    T* get_device_data() {
        return device_data;
    }

    queue* get_queue() {
        return this->q;
    }

    int get_size() {
        return this->size;
    }

};