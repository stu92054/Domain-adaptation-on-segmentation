// --------------------------------------------------------
// CCNN 
// Copyright (c) 2015 [See LICENSE file for details]
// Written by Deepak Pathak, Philipp Krähenbühl
// --------------------------------------------------------

#pragma once
#include "util/eigen.h"

struct LinearConstraint {
	LinearConstraint( const VectorXf & a, float b, float slack=1e10 );
	// A constraint \sum_i a*x_i >= b - slack
	VectorXf a;
	float b,slack;
	float eval( const RMatrixXf & x ) const;
};

class ConstraintSoftmax {
protected:
	float scale_;
	std::vector<LinearConstraint> linear_constraints_;
	VectorXb zero_constraints_;
public:
	ConstraintSoftmax( float scale=1.0 );
	// A constraint \sum_i a*x_i >= b
	void addLinearConstraint( const VectorXf & a, float b, float slack=1e10 );
	// A constraint \sum_i a*x_i == 0  where a >= 0
	void addZeroConstraint( const VectorXf & a );
	RMatrixXf compute( const RMatrixXf & f ) const;
	RMatrixXf computeLog( const RMatrixXf & f ) const;
};
