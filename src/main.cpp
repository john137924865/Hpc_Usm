#include <sycl/sycl.hpp>
#include <iostream>
#include <chrono>
#include "mysycl/mysycl.hpp"
#include "diamante_4.cpp"
#include "no_dipendenze.cpp"
#include "dipendenza_lineare.cpp"
#include "dipendenza_albero_binario.cpp"

int main() {
    int size, num_kernels;
    std::cout << "Milioni: ";
    std::cin >> size;
    std::cout << size << std::endl << std::endl;
    std::cout << "Numero kernel +1 (scrivi 8 ma albero ha 7): ";
    std::cin >> num_kernels;
    std::cout << num_kernels << std::endl << std::endl;
    size *= 1000000;

    //diamante_4::diamante_4(size);
    //no_dipendenze::no_dipendenze(size, num_kernels);
    //dipendenza_lineare::dipendenza_lineare(size, num_kernels);
    dipendenza_albero_binario::dipendenza_albero_binario(size, num_kernels);
}