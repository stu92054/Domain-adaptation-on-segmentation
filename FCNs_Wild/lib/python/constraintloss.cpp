// --------------------------------------------------------
// CCNN 
// Copyright (c) 2015 [See LICENSE file for details]
// Written by Deepak Pathak, Philipp Krähenbühl
// --------------------------------------------------------

#include "constraintloss.h"
#include "ccnn.h"
#include "constraintloss/constraintsoftmax.h"

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(ConstraintSoftmax_addLinearConstraint_o, ConstraintSoftmax::addLinearConstraint, 2, 3 );

void defineConstraintloss() {
	ADD_MODULE( constraintloss );
	
	class_<ConstraintSoftmax>("ConstraintSoftmax",init<>())
	.def(init<float>())
	.def( "addLinearConstraint", &ConstraintSoftmax::addLinearConstraint, ConstraintSoftmax_addLinearConstraint_o() )
	.def( "addZeroConstraint", &ConstraintSoftmax::addZeroConstraint )
	.def( "compute", &ConstraintSoftmax::compute )
	.def( "computeLog", &ConstraintSoftmax::computeLog );
}
