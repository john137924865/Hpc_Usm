#pragma once

#include <sycl/sycl.hpp>
#include <iostream>
#include "myBuffer.cpp"

using namespace sycl;

template <typename T>
class myAccessor {
private:
    T* device_data;
    T* host_data;
    int size;
    handler* h;

public:

    myAccessor(myBuffer<T>& buf, handler& h) {
        this->device_data = buf.get_device_data();
        this->host_data = buf.get_host_data();
        this->h = &h;
        this->size = buf.get_size();
    }

    ~myAccessor() {
        //(*h).memcpy(host_data, device_data, sizeof(T) * size);
    }/*

    myAccessor(const myAccessor& other) {
        this->device_data = other.device_data;
        this->host_data = other.host_data;
        this->size = other.size;
        this->h = other.h;
    }

    myAccessor& operator=(const myAccessor& other) {
        if (this != &other) {
            this->device_data = other.device_data;
            this->host_data = other.host_data;
            this->size = other.size;
            this->h = other.h;
        }
        return *this;
    }*/

    T& operator[](size_t index) const {
        return device_data[index];
    }

};