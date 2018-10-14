// --------------------------------------------------------
// CCNN 
// Copyright (c) 2015 [See LICENSE file for details]
// Written by Deepak Pathak, Philipp Krähenbühl
// --------------------------------------------------------

#pragma once
#include "util/eigen.h"
#include <functional>

typedef std::function<float(const VectorXf & x, VectorXf * gradient)> function_t;
typedef std::function<VectorXf(const VectorXf & x)> projection_t;

VectorXf identity(const VectorXf & x );
VectorXf fista( VectorXf x0, function_t f, projection_t p = identity, bool verbose=false );
VectorXf pgd( VectorXf x0, function_t f, projection_t p = identity, bool verbose=false, bool * converged=NULL );
