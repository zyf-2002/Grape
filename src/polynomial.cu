#include "polynomial.cuh"


Polynomial::~Polynomial() {
    delete[] coefficients_;
}

Polynomial::Polynomial() : degree_(0) {
    coefficients_ = new Fr[1];
    coefficients_[0] = Fr::zero();
}

Polynomial::Polynomial(int degree) : degree_(degree) {
    coefficients_ = new Fr[degree_ + 1];
    for (int i = 0; i <= degree_; i++)
        coefficients_[i] = Fr::zero();
}

Polynomial::Polynomial(const Polynomial& other) : degree_(other.degree_) {
    coefficients_ = new Fr[degree_ + 1];
    std::memcpy(coefficients_,
                other.coefficients_,
                (degree_ + 1) * sizeof(Fr));
}

Polynomial::Polynomial(const std::vector<Fr>& coefficients)
    : degree_(coefficients.size() - 1) {
    coefficients_ = new Fr[degree_ + 1];
    std::memcpy(coefficients_,
                coefficients.data(),
                (degree_ + 1) * sizeof(Fr));
}

Polynomial::Polynomial(const Fr& constant) : degree_(0) {
    coefficients_ = new Fr[1];
    coefficients_[0] = constant;
}


Polynomial Polynomial::eq(const Fr& u) {
    Polynomial eq(1);
    eq.coefficients_[0] = Fr::one() - u;
    eq.coefficients_[1] = u + u - Fr::one();
    return eq;
}

Fr Polynomial::eq(const Fr& u, const Fr& v) {
    Fr result = u * v;
    result += result;
    result -= (u + v);
    result += Fr::one();
    return result;
}


Polynomial& Polynomial::operator=(const Polynomial& other) {
    if (this == &other) return *this;

    delete[] coefficients_;
    degree_ = other.degree_;
    coefficients_ = new Fr[degree_ + 1];
    std::memcpy(coefficients_,
                other.coefficients_,
                (degree_ + 1) * sizeof(Fr));
    return *this;
}



Polynomial Polynomial::operator+(const Polynomial& other) const {
    int resultDegree = std::max(degree_, other.degree_);
    Polynomial result(resultDegree);

    #pragma omp parallel for
    for (int i = 0; i <= resultDegree; i++) {
        if (i > degree_) result.coefficients_[i] = other.coefficients_[i];
        else if (i > other.degree_) result.coefficients_[i] = coefficients_[i];
        else result.coefficients_[i] = coefficients_[i] + other.coefficients_[i];
    }
    return result;
}

Polynomial Polynomial::operator-(const Polynomial& other) const {
    int resultDegree = std::max(degree_, other.degree_);
    Polynomial result(resultDegree);

    #pragma omp parallel for
    for (int i = 0; i <= resultDegree; i++) {
        if (i > degree_) result.coefficients_[i] = -other.coefficients_[i];
        else if (i > other.degree_) result.coefficients_[i] = coefficients_[i];
        else result.coefficients_[i] = coefficients_[i] - other.coefficients_[i];
    }
    return result;
}

Polynomial Polynomial::operator*(const Polynomial& other) const {
    Polynomial result(degree_ + other.degree_);
    for (int i = 0; i <= degree_; i++)
        for (int j = 0; j <= other.degree_; j++)
            result.coefficients_[i + j] +=
                coefficients_[i] * other.coefficients_[j];
    return result;
}

Polynomial Polynomial::operator*(const Fr& val) const {
    Polynomial result(*this);
    #pragma omp parallel for
    for (int i = 0; i <= degree_; i++)
        result.coefficients_[i] *= val;
    return result;
}

Polynomial& Polynomial::operator+=(const Polynomial& other) {
    *this = *this + other;
    return *this;
}

Polynomial& Polynomial::operator-=(const Polynomial& other) {
    *this = *this - other;
    return *this;
}

Polynomial& Polynomial::operator*=(const Polynomial& other) {
    *this = *this * other;
    return *this;
}


Fr Polynomial::operator()(const int& x) const {
    Fr result = coefficients_[0];
    if (x == 1) {
        for (int i = 1; i <= degree_; i++)
            result += coefficients_[i];
    }
    return result;
}

Fr Polynomial::operator()(const Fr& x) const {
    Fr result = coefficients_[degree_];
    for (int i = degree_ - 1; i >= 0; i--) {
        result *= x;
        result += coefficients_[i];
    }
    return result;
}
