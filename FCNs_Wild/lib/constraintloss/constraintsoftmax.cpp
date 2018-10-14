// --------------------------------------------------------
// CCNN 
// Copyright (c) 2015 [See LICENSE file for details]
// Written by Deepak Pathak, Philipp Krähenbühl
// --------------------------------------------------------

#include "constraintsoftmax.h"
#include "optimization/fista.h"
#include "util/eigen.h"
#include <iostream>

LinearConstraint::LinearConstraint( const VectorXf & a, float b, float slack ):a(a),b(b),slack(slack) {
}
float LinearConstraint::eval( const RMatrixXf & x ) const {
	// Return   \sum_i a*x_i - b
	return (x*a).array().sum()-b;
}

// Performed across columns i.e. across channels
static RMatrixXf expAndNormalize( const RMatrixXf & m ) {
	VectorXf mx = m.rowwise().maxCoeff();
	RMatrixXf r = (m.colwise()-mx).array().exp();
	return r.array().colwise() / r.array().rowwise().sum();
}
static VectorXf logSumExp( const RMatrixXf & m ) {
	VectorXf mx = m.rowwise().maxCoeff();
	return mx.array() + (m.colwise()-mx).array().exp().rowwise().sum().log();
}

// scale_ : determines the hardness of optimization. In hard case, entropy term in KL divergence is zero.
// Alternate way to implement this is to scale the second cross entropy term in KL divergence by 1000 times !
ConstraintSoftmax::ConstraintSoftmax( float scale ):scale_(scale) {
}

void ConstraintSoftmax::addLinearConstraint( const VectorXf & a, float b, float slack ) {
	linear_constraints_.push_back( LinearConstraint(a, b, slack) );
}
void ConstraintSoftmax::addZeroConstraint( const VectorXf & a ) {
	eassert( (a.array() >= 0).all() );
	if( zero_constraints_.size() )
		zero_constraints_.array() = zero_constraints_.array() || (a.array() > 0);
	else
		zero_constraints_ = a.array() > 0;
}
RMatrixXf ConstraintSoftmax::compute( const RMatrixXf & f ) const {
	return expAndNormalize( scale_*computeLog( f ) );
}
RMatrixXf ConstraintSoftmax::computeLog( const RMatrixXf & f ) const {
	const int N = f.rows(), M = f.cols();
	// Special handling for zero constraints, let's remove all dimensions
	// that are constraint to 0
	int pM = M;
	RMatrixXf pf = f, P;
	std::vector<LinearConstraint> pc = linear_constraints_;
	
	// Project onto the zero constraints
	if( zero_constraints_.size() ) {
		pM = (zero_constraints_.array()==0).cast<int>().sum();
		if( pM <= 1 ) {
			RMatrixXf r = 1*f;
			for( int i=0; i<M; i++ )
				if( zero_constraints_[i]>0 )
					r.col(i).setConstant(-1e10);
			return r;
		}
		// Build the projection matrix
		P = RMatrixXf::Zero(M,pM);
		for( int i=0,k=0; i<M; i++ )
			if( !zero_constraints_[i] )
				P(i,k++) = 1;
		// Project onto the matrix : Means remove the variables with zero equality constraints
		pf = pf * P;
		for(auto & c: pc)
			c.a = P.transpose() * c.a;
	}
	
	// Let's formulate the constraints as Ap >= b 		(with slack : Ap >= b - slack)
	// Then our objective is D(p||q) = \sum p log p - \sum p log q + l' (b - Ap)
	//                               = - H_p - \sum p pf - l' A p + lb
	//                  d/dp D(p||q) = log p + 1 + c - pf - A' l = 0
	//                             p = 1/Z exp(fp + A'l)
	//                             where l >= 0
	// The objective then simplifies to
	//                       D(p||q) = \sum p (fp + A'l) - log Z - \sum p pf + l' (b - Ap)
	//                               = -log Z + l' b
	
	RMatrixXf A(pc.size(),pM);
	VectorXf b(pc.size()), slack(pc.size());
	for( int i=0; i<(int)pc.size(); i++ ) {
		A.row(i) = pc[i].a.transpose() / N; 		// Normalize by spatial_dim (no change theoretically, for implementation stability)
		b[i] = pc[i].b;
		slack[i] = pc[i].slack * N;					// Scale regularizer of slack according to spatial_dim
	}
	
	// Solve for the soft assignment to the laten variables
	// This function returns 
	//	g : Gradient vector for dual variables. Returned as function argument.
	//  return : objective value for dual optimization (which is to be minimized)
	auto fun = [&](const VectorXf & l, VectorXf * g) -> double {
		if( g ) {
			RMatrixXf p = expAndNormalize( scale_*(pf.rowwise() + l.transpose()*A) ).colwise().sum();
			*g =  A*(p.colwise().sum()).transpose() - b;
		}
		return 1.0/scale_*logSumExp( scale_*(pf.rowwise() + l.transpose()*A) ).sum() - l.dot(b);
	};
	auto proj = [&](const VectorXf & x)->VectorXf {
		// if (x.array().maxCoeff() > 0) {
		// 	std::cout<<"\nActive Dual before slack : "<<x.array().maxCoeff()<<"\n";
		// 	std::cout<<"Scaled slack value : "<<slack.array().minCoeff()<<"  Input scale value : "<<(float)slack.array().minCoeff()/(float)N<<"  scale N : "<<N<<"\n";
		// }
		return x.array().max(0.f).min(slack.array()); 					// 0 <= dual variable <= slack
	};
	
	// Solve for the lagrangian dual to the constraint optimization
	// VectorXf l = fista( VectorXf::Zero(pc.size()), fun, proj );		// accelerated pgd : pgd + momentum
	bool converged = false;
	VectorXf l = pgd( VectorXf::Zero(pc.size()), fun, proj, false, &converged );
	if(!converged)
		printf("Projected gradient descent didn't converge. The problem might not be satisfiable!\n");

	// Compute the 'labels'
	RMatrixXf r = pf.rowwise() + l.transpose()*A;
	
	// Construct the result
	if( zero_constraints_.size() ) {
		r = r * P.transpose();
		for( int i=0; i<M; i++ )
			if( zero_constraints_[i]>0 )
				r.col(i).setConstant(-1e10);
	}
	return r;
}

