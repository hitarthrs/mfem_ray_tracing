#ifndef MFEM_RAYTRACING_EMBREE_INTERFACE_HPP
#define MFEM_RAYTRACING_EMBREE_INTERFACE_HPP

#ifdef MFEM_RAYTRACING_EMBREE4

#include "embree/embree4.hpp"

#elif defined(MFEM_RAYTRACING_EMBREE3)

#include "embree/embree3.hpp"

#else

#error "Embree support is disabled or no Embree version macro was provided to the compiler"

#endif

#endif
