#pragma once

#include <sycl/sycl.hpp>
#include <vector>
#include <iostream>

using namespace sycl;

template <typename T>
class myBuffer {

    private:
        std::string name;
        T* device_data = nullptr;
        T* host_data = nullptr;
        size_t size;
        queue* q = nullptr;
        std::vector<event> events;
        std::vector<std::string> names;
        std::vector<event> last_write;
        std::vector<std::string> last_write_name;
        bool current_write = false;
        bool ever_write = false;

    public:

        myBuffer(queue& q, size_t n, std::string name) : size(n) {
            this->device_data = static_cast<T*>(malloc_device(sizeof(T) * size, q));
            this->host_data = new T[size];
            this->q = &q;
            this->name = name;
        }

        ~myBuffer() {
            if (device_data) {
                sycl::free(device_data, *q);
            }
            if (host_data) {
                delete[] host_data;
            }
        }

        void copy_host_to_device() {
            (*q).memcpy(device_data, host_data, sizeof(T) * size).wait();
        }

        void copy_device_to_host() {
            (*q).memcpy(host_data, device_data, sizeof(T) * size).wait();
        }

        T* get_device_data() {
            return device_data;
        }
        
        T* get_host_data() {
            return host_data;
        }

        void check_mode(access::mode mode, handler& h) {
            if (mode != access::mode::read) {
                std::cout << name << " depends_on " << events.size() << " ";
                for (int i = 0; i < events.size(); i++) {
                    h.depends_on(events[i]);
                    std::cout << names[i] << " ";
                }
                std::cout << ";" << std::endl;
                events.clear();
                names.clear();
                current_write = true;
            }
            if (ever_write) {
                h.depends_on(last_write[0]);
                std::cout << name << " depends_on last_write " << last_write_name[0] << std::endl;
            }
            //std::cout << "check_mode; size : " << events.size() << ", write: " << std::boolalpha << write << std::endl;
        }

        void add_event(event e, std::string name) {
            if (current_write) {
                ever_write = true;
                last_write.clear();
                last_write_name.clear();
                last_write.push_back(e);
                last_write_name.push_back(name);
                std::cout << this->name << " aggiorno last_write con " << name << std::endl;
                current_write = false;
            } else {
                events.push_back(e);
                names.push_back(name);
                std::cout << this->name << " push_back events e names " << name << std::endl;
            }
            //std::cout << "add_event; size : " << events.size() << ", write: " << std::boolalpha << write << std::endl;
        }

        /*~myBuffer() {
            std::cout << events.size() << std::endl << modes.size() << std::endl;
        }*/

};