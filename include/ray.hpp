#ifndef RAY_HPP
#define RAY_HPP

#include "mfem.hpp"

class Ray
{
public:
    // Constructor
    Ray(const mfem::Vector &origin, const mfem::Vector &direction, double weight = 1.0)
        : origin_(origin), direction_(direction), t_min_(0.0), t_max_(1.0), weight_(weight)
    {
        const double n = direction_.Norml2();
        if (n > 0.0)
        {
            direction_ /= n;
        }
    }

    // Evaluates the ray equation at a given step distance 't' -> P(t) = O + t*D
    void Evaluate(double t, mfem::Vector &point) const {
        point.SetSize(origin_.Size());
        add(origin_, t, direction_, point); // MFEM math: point = origin_ + t * direction_
    }

    // Getters
    const mfem::Vector &GetOrigin() const { return origin_; }
    const mfem::Vector &GetDirection() const { return direction_; }
    double GetTMin() const { return t_min_; }
    double GetTMax() const { return t_max_; }
    double GetWeight() const { return weight_; }

    // Setters
    void SetTMin(double t_min) { t_min_ = t_min; }
    void SetTMax(double t_max) { t_max_ = t_max; }
    void SetWeight(double w) { weight_ = w; }

private:
    mfem::Vector origin_;
    mfem::Vector direction_;
    double t_min_;
    double t_max_;
    double weight_;
};

#endif