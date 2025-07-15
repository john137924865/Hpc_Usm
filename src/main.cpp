#include <sycl/sycl.hpp>
#include <iostream>
#include <chrono>
#include "diamante_4.cpp"
#include "no_dipendenze.cpp"
#include "dipendenza_lineare.cpp"
#include "dipendenza_albero_binario.cpp"

int main() {
    int size, num_kernels, scelta;

    std::cout << "Migliaia: ";
    std::cin >> size;
    size *= 1000;
    std::cout << size << std::endl << std::endl;

    std::cout << "Scegli quale funzione eseguire:\n"
        << "1. diamante_4\n"
        << "2. no_dipendenze\n"
        << "3. dipendenza_lineare\n"
        << "4. dipendenza_albero_binario\n"
        << "Scelta: ";
    std::cin >> scelta;
    std::cout << std::endl;

    switch (scelta) {
    case 1:
        diamante_4::diamante_4(size);
        break;
    case 2:
    case 3:
    case 4:
        std::cout << "Numero di kernel: ";
        std::cin >> num_kernels;
        std::cout << num_kernels << std::endl << std::endl;

        if (scelta == 2)
            no_dipendenze::no_dipendenze(size, num_kernels);
        else if (scelta == 3)
            dipendenza_lineare::dipendenza_lineare(size, num_kernels);
        else
            dipendenza_albero_binario::dipendenza_albero_binario(size, num_kernels);

        break;
    default:
        std::cout << "Scelta non valida.\n";
        break;
    }
}