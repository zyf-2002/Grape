#pragma once
#include "fr-tensor.cuh"
#include "ec_operation.cuh"
#include <vector>



class Polynomial {
public:
    Polynomial();
    Polynomial(int degree);
    Polynomial(int degree, Fr* coefficients);
    Polynomial(const Polynomial& other);
    Polynomial(const Fr& constant);
    Polynomial(const std::vector<Fr>& coefficients);
    ~Polynomial();

    Polynomial& operator=(const Polynomial& other);

    Polynomial operator+(const Polynomial& other) const;
    Polynomial operator-(const Polynomial& other) const;
    Polynomial operator*(const Polynomial& other) const;
    Polynomial operator*(const Fr& val) const;
    Polynomial operator-() const;

    Polynomial& operator+=(const Polynomial& other);
    Polynomial& operator-=(const Polynomial& other);
    Polynomial& operator*=(const Polynomial& other);

    Fr operator()(const Fr& x) const;
    Fr operator()(const int& x) const;

    int getDegree() const;
    void setCoefficients(int degree, Fr* coefficients);

    static Polynomial eq(const Fr& u);
    static Fr eq(const Fr& u, const Fr& v);

    friend std::ostream& operator<<(std::ostream& os, const Polynomial& poly);

private:
    Fr* coefficients_{nullptr};
    int degree_{0};
};
