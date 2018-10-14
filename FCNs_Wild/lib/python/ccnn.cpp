// --------------------------------------------------------
// CCNN 
// Copyright (c) 2015 [See LICENSE file for details]
// Written by Deepak Pathak, Philipp Krähenbühl
// --------------------------------------------------------

#include "util.h"
#include "constraintloss.h"
#include "ccnn.h"

BOOST_PYTHON_MODULE(ccnn)
{
	import_array1();

	defineUtil();
	defineConstraintloss();
}
