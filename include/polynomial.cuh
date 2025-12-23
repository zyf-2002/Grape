#pragma once
#include "fr-tensor.cuh"
#include "ec_operation.cuh"


class Polynomial{
public:
    Polynomial();
    Polynomial(int degree);
    Polynomial(int degree, Fr* coefficients);
    Polynomial(const Polynomial& other);
    Polynomial(const Fr& constant);
    Polynomial(const vector<Fr>& coefficients);
    ~Polynomial();

    Polynomial& operator=(const Polynomial& other);
    Polynomial operator+(const Polynomial& other);
    Polynomial operator-(const Polynomial& other);
    Polynomial operator*(const Polynomial& other);
    Polynomial operator-();

    Polynomial& operator+=(const Polynomial& other);
    Polynomial& operator-=(const Polynomial& other);
    Polynomial& operator*=(const Polynomial& other);

    Fr operator()(const Fr& x);
    Fr operator()(const int& x);

    int getDegree() const;
    void setCoefficients(int degree, Fr* coefficients);

    static Polynomial eq(const Fr& u);
    static Polynomial eq(const Fr& before, const Fr& u);

    friend std::ostream& operator<<(std::ostream& os, const Polynomial& poly);
private:
    Fr* coefficients_;
    int degree_;
};

Polynomial::Polynomial(const Polynomial& other) : degree_(other.degree_) {
    coefficients_ = new Fr[degree_ + 1];
     std::memcpy(coefficients_,
                other.coefficients_,
                (degree_ + 1) * sizeof(Fr));
}

Polynomial::Polynomial(int degree) : degree_(degree) {
    coefficients_ = new Fr[degree_ + 1];
    for(int i = 0; i <= degree; i++)  coefficients_[i] = Fr::zero();
    //memset(coefficients_, 0, (degree + 1) * sizeof(Fr));
    
}

Polynomial::Polynomial(const vector<Fr>& coefficients)
    : degree_(coefficients.size() - 1) {
    coefficients_ = new Fr[degree_ + 1];
    std::memcpy(coefficients_,
                coefficients.data(),
                (degree_ + 1) * sizeof(Fr));
}

Polynomial Polynomial::eq(const Fr& u){
    Polynomial eq(1);
    eq.coefficients_[0] = Fr::one() - u;
    eq.coefficients_[1] = u + u - Fr::one();
    return eq;
}

Polynomial Polynomial::eq(const Fr& before, const Fr& u)
{
    Polynomial eq(1);
    eq.coefficients_[0] = (Fr::one() - u) * before;
    eq.coefficients_[1] = (u + u - Fr::one()) * before;
    return eq;
}

Polynomial::~Polynomial() {
    if (coefficients_ != nullptr) {
        delete[] coefficients_;
    }
}

Polynomial& Polynomial::operator=(const Polynomial& other) {
    if (coefficients_ != nullptr) {
        delete[] coefficients_;
    }
    degree_ = other.degree_;
    coefficients_ = new Fr[degree_ + 1];
    std::memcpy(coefficients_,
                other.coefficients_,
                (degree_ + 1) * sizeof(Fr));
    return *this;
}

Polynomial Polynomial::operator*(const Polynomial& other) {
    int resultDegree = degree_ + other.degree_;
    Polynomial result(resultDegree);

    for (int i = 0; i <= degree_; i++) {
        for (int j = 0; j <= other.degree_; j++) {
            result.coefficients_[i + j] +=
                coefficients_[i] * other.coefficients_[j];
        }
    }
    return result;
}

Polynomial& Polynomial::operator*=(const Polynomial& other)
{
    (*this) = (*this) * other;
    return *this;
}

Fr Polynomial::operator()(const int& x)
{
    Fr result = coefficients_[0];
    
    if(x == 1){
        for(int i = 1; i <= degree_; i++) result += coefficients_[i];
    }
    return result;
}

Fr Polynomial::operator()(const Fr& x)
{
    Fr result;
    result = coefficients_[degree_];
    for(int i = degree_ - 1; i >= 0; i--){
        result *= x;
        result += coefficients_[i];
    }
    return result;
}





