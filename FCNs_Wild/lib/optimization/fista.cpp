// --------------------------------------------------------
// CCNN 
// Copyright (c) 2015 [See LICENSE file for details]
// Written by Deepak Pathak, Philipp Krähenbühl
// --------------------------------------------------------

#include "fista.h"
#include <iostream>

VectorXf identity(const VectorXf & x ) { return x; }

VectorXf fista( VectorXf x0, function_t f, projection_t p, bool verbose ) {
	const int N_ITER = 3000;
	const float beta = 0.5;
	float alpha = 1e-1;
	
	VectorXf r = x0;
	float best_e = 1e10;
	VectorXf x1 = x0, g = 0*x0;
	for( int k=1; k<=N_ITER && alpha>1e-5; k++ ) {
		// Strictly speaking this is not "legal" FISTA, but it seems to work well in practice
		alpha *= 1.05;
		
		// Compute y
		VectorXf y = x1 + (k-2.) / (k+1.)*(x1 - x0);
		// Evaluate the gradient at y
		float fy = f(y,&g), fx = 1e10;
		// Update the old x
		x0 = x1;
		// Update x
		x1 = p( y - alpha*g );
		while( alpha >= 1e-5 && (fx=f(x1,NULL)) > fy + g.dot(x1-y)+1./(2.*alpha)*(x1-y).dot(x1-y) ) {
			alpha *= beta;
			x1 = p( y - alpha*g );
		}
		if ( fx < best_e ) {
			best_e = fx;
			r = x0;
		}
		if (verbose){
			printf("it = %d   df = %f   alpha = %f\n", k, (x0-x1).array().abs().maxCoeff(), alpha );
			std::cout<<y.transpose()<<std::endl;
			std::cout<<g.transpose()<<std::endl;
			VectorXf gg;
			f(x1,&gg);
			std::cout<<x1.transpose()<<std::endl;
			std::cout<<gg.transpose()<<std::endl;
			std::cout<<std::endl;
		}
		if( (x0-x1).array().abs().maxCoeff() < 1e-4 )
			break;
	}
	return r;
}

VectorXf pgd( VectorXf x0, function_t f, projection_t p, bool verbose, bool * converged ) {
	const int N_ITER = 3000;
	const float beta = 0.5;
	float alpha = 1e5;
	
	VectorXf g = 0*x0;
	float prev_fx = f(x0,&g);

	int k=1;
	for( k=1; k<=N_ITER && alpha>1e-8; k++ ) {
		VectorXf ng;
		float fx = f(p(x0-alpha*g),&ng);
		if( fx < prev_fx ) {
			x0 = p(x0-alpha*g);
			g = ng;
			prev_fx = fx;
			alpha *= 1.1;
		}
		else
			alpha *= beta;
	}

	// Debugging
	// if (k>N_ITER){
	// 	std::cout<<"PGD didn't converge\n";
	// 	std::cout<<"K="<<k<<" alpha="<<alpha<<"\n";
	// 	std::cout<<"Dual before slack :\n"<<x0.array()<<"\n"; 		//x0.array().maxCoeff()
	// 	std::cout<<"Gradient at last step :\n"<<g.array()<<"\n";
	// }
	if( converged ) *converged = (k < N_ITER);

	return x0;
}
