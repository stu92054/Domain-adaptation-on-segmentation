// --------------------------------------------------------
// CCNN 
// Copyright (c) 2015 [See LICENSE file for details]
// Written by Deepak Pathak, Philipp Krähenbühl
// --------------------------------------------------------

#pragma once
#include "boost.h"
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#define ADD_MODULE( name ) object name ## Module(handle<>(borrowed(PyImport_AddModule(((std::string)"ccnn."+# name).c_str()))));\
scope().attr(# name) = name ## Module;\
scope name ## _scope = name ## Module;
