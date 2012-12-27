/*****************************************************************************
#                                                                            /
#                              parallelFFT.h                                 /
#                                                                            /
#                        Matching Pursuit Library                            /
#                                                                            /
# M. Moussallam                                             Tue Mar 22 2011  /
# -------------------------------------------------------------------------- /
#                                                                            /
#                                                                            /
#  This program is free software; you can redistribute it and/or             /
#  modify it under the terms of the GNU General Public License               /
#  as published by the Free Software Foundation; either version 2            /
#  of the License, or (at your option) any later version.                    /
#                                                                            /
#  This program is distributed in the hope that it will be useful,           /
#  but WITHOUT ANY WARRANTY; without even the implied warranty of            /
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the             /
#  GNU General Public License for more details.                              /
#                                                                            /
#  You should have received a copy of the GNU General Public License         /
#  along with this program; if not, write to the Free Software               /
#  Foundation, Inc., 59 Temple Place - Suite 330,                            /
#  Boston, MA  02111-1307, USA.                                              /
#                                                                            /
#******************************************************************************/


#include <Python.h>
#include <stdio.h>
#include "ndarrayobject.h"
/*#include <complex.h>*/
#include <fftw3.h>
#include <omp.h>
#include <unistd.h>

#include "parProj.h"
#define PI 3.14159265359

/* Basic feature 64  bits platform */
#define X 2

/* modify if 32 bit detected */
#ifdef _M_X86
#define X 1
#endif

/* MCLT functions */

/*
 * Classic MDCT projection of the framed signal 
 */
static PyObject * project(PyObject *self, PyObject *args);

/*
 * Simulating a subsampled dictionary
 */
static PyObject * subproject(PyObject *self, PyObject *args);

/* MCLT projection of the framed signal : use for LOMP algorithm */

static PyObject * project_mclt(PyObject *self, PyObject *args);

static PyObject * project_mclt_set(PyObject *self, PyObject *args);


static PyObject * project_atom_set(PyObject *self, PyObject *args);

static PyObject * project_atom(PyObject *self, PyObject *args);

/* Imported utilities from cookbook to handle vectors */
double *pyvector_to_Carrayptrs(PyArrayObject *arrayin);

/* Modified utilities from cookbook to handle complex vectors */
fftw_complex *pyvector_to_complexCarrayptrs(PyArrayObject *arrayin);

int *pyvector_to_intCarrayptrs(PyArrayObject *arrayin);

static PyObject * test(PyObject *self, PyObject *args);

/* Multi-Threaded plan initialization */
static PyObject *  initialize_plans(PyObject *self, PyObject *args);

static PyObject *  clean_plans(PyObject *self, PyObject *args);

static PyObject * get_atom(PyObject *self, PyObject *args);

static PyObject * get_real_gabor_atom(PyObject *self, PyObject *args);
