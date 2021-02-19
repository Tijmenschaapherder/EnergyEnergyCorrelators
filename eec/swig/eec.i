// -*- C++ -*-
//
// EnergyEnergyCorrelators - Evaluates EECs on particle physics events
// Copyright (C) 2020 Patrick T. Komiske III
// 
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

%module("threads"=1) eec
%nothreadallow;

#define EECNAMESPACE eec

// this can be used to ensure that swig parses classes correctly
#define SWIG_PREPROCESSOR

%feature("autodoc", "1");

// C++ standard library wrappers
%include <exception.i>
%include <std_array.i>
%include <std_string.i>
%include <std_vector.i>

%{
// include these to avoid needing to define them at compile time 
#ifndef SWIG
#define SWIG
#endif
#ifndef SWIG_EEC
#define SWIG_EEC
#endif

// needed by numpy.i, harmless otherwise
#define SWIG_FILE_WITH_INIT

// standard library headers we need
#include <cstdlib>
#include <cstring>

// EEC library headers
#include "EEC.hh"

// using namespaces
using namespace eec;
using namespace eec::hist;
%}

// include numpy typemaps
%include numpy.i
%init %{
import_array();
%}

%pythoncode %{
__all__ = ['EECLongestSide', 'EECTriangleOPE',
           'EECLongestSideId', 'EECLongestSideLog',
           'EECTriangleOPEIdIdId', 'EECTriangleOPEIdLogId',
           'EECTriangleOPELogIdId', 'EECTriangleOPELogLogId']
%}

// vector templates
%template(vectorDouble) std::vector<double>;
%template(vectorUnsigned) std::vector<unsigned>;

// array templates
%template(arrayUnsigned13) std::array<unsigned, 13>;

// allow threads in PairwiseEMD computation
%threadallow EECNAMESPACE::EECBase::_batch_compute;
;

// numpy typemaps
//%apply (double* IN_ARRAY1, int DIM1) {(double* weights0, int n0), (double* weights1, int n1)}
%apply (double* IN_ARRAY2, int DIM1, int DIM2) {(double* particles, int mult, int nfeatures)}
%apply (double* INPLACE_ARRAY2, int DIM1, int DIM2) {(double* particles_noconvert, int mult, int nfeatures)}
//%apply (double* INPLACE_ARRAY1, int DIM1) {(double* weights, int n0)}
//%apply (double* INPLACE_ARRAY2, int DIM1, int DIM2) {(double* coords, int n1, int d)}
%apply (double** ARGOUTVIEWM_ARRAY1, int* DIM1) {(double** arr_out0, int* n0),
                                                 (double** arr_out1, int* n1)}
%apply (double** ARGOUTVIEWM_ARRAY3, int* DIM1, int* DIM2, int* DIM3) {(double** arr_out0, int* n0, int* n1, int* n2),
                                                                       (double** arr_out1, int* m0, int* m1, int* m2)}

// makes python class printable from a description method
%define ADD_STR_FROM_DESCRIPTION
std::string __str__() const {
  return $self->description();
}
std::string __repr__() const {
  return $self->description();
}
%enddef

// mallocs a 1D array of doubles of the specified size
%define MALLOC_1D_VALUE_ARRAY(arr_out, n, size, nbytes)
  *n = size;
  size_t nbytes = size_t(*n)*sizeof(double);
  *arr_out = (double *) malloc(nbytes);
  if (*arr_out == NULL) {
    PyErr_Format(PyExc_MemoryError, "Failed to allocate %zu bytes", nbytes);
    return;
  }
%enddef

// mallocs a 3D array of doubles of the specified size
%define MALLOC_3D_VALUE_ARRAY(arr_out, n0, n1, n2, size0, size1, size2, nbytes)
  *n0 = size0;
  *n1 = size1;
  *n2 = size2;
  size_t nbytes = size_t(*n0)*size_t(*n1)*size_t(*n2)*sizeof(double);
  *arr_out = (double *) malloc(nbytes);
  if (*arr_out == NULL) {
    PyErr_Format(PyExc_MemoryError, "Failed to allocate %zu bytes", nbytes);
    return;
  }
%enddef

%define RETURN_1DNUMPY_FROM_VECTOR(pyname, cppname, size)
void pyname(double** arr_out0, int* n0) {
  MALLOC_1D_VALUE_ARRAY(arr_out0, n0, size, nbytes)
  memcpy(*arr_out0, $self->cppname().data(), nbytes);
}
%enddef

// basic exception handling for all functions
%exception {
  try { $action }
  SWIG_CATCH_STDEXCEPT
}

namespace EECNAMESPACE {

// ignore/rename EECHist functions
namespace hist {
  %ignore EECHistBase::get_hist_errs;
  %rename(bin_centers_vec) EECHistBase::bin_centers;
  %rename(bin_edges_vec) EECHistBase::bin_edges;
  %rename(bin_centers) EECHistBase::npy_bin_centers;
  %rename(bin_edges) EECHistBase::npy_bin_edges;
  %rename(get_hist_errs) EECHist1D::npy_get_hist_errs;
  %rename(get_hist_errs) EECHist3D::npy_get_hist_errs;
}

// ignore/rename Multinomial functions
%ignore multinomial;
%rename(multinomial) multinomial_vector;

// ignore EEC functions
%ignore EECEvents::append;
%ignore EECBase::batch_compute;
%ignore EECBase::compute;
%rename(compute) EECBase::npy_compute;

} // namespace EECNAMESPACE

// include EECHist and declare templates
%include "EECHist.hh"

namespace EECNAMESPACE {
  namespace hist {

    // extend EECHistBase
    %extend EECHistBase {
      void npy_bin_centers(double** arr_out0, int* n0, int i = 0) {
        MALLOC_1D_VALUE_ARRAY(arr_out0, n0, $self->nbins(i), nbytes)
        memcpy(*arr_out0, $self->bin_centers(i).data(), nbytes);
      }
      void npy_bin_edges(double** arr_out0, int* n0, int i = 0) {
        MALLOC_1D_VALUE_ARRAY(arr_out0, n0, $self->nbins(i)+1, nbytes)
        memcpy(*arr_out0, $self->bin_edges(i).data(), nbytes);
      }
    }

    // extend EECHist1D code
    %extend EECHist1D {
      void npy_get_hist_errs(double** arr_out0, int* n0,
                             double** arr_out1, int* n1,
                             unsigned hist_i = 0, bool include_overflows = true) {
        MALLOC_1D_VALUE_ARRAY(arr_out0, n0, $self->hist_size(include_overflows), nbytes0)
        MALLOC_1D_VALUE_ARRAY(arr_out1, n1, $self->hist_size(include_overflows), nbytes1)
        try {
          $self->get_hist_errs(*arr_out0, *arr_out1, hist_i, include_overflows);
        }
        catch (std::exception & e) {
          free(*arr_out0);
          free(*arr_out1);
          throw;
        }
      }
    }

    // extend EECHist3D code
    %extend EECHist3D {
      void npy_get_hist_errs(double** arr_out0, int* n0, int* n1, int* n2,
                             double** arr_out1, int* m0, int* m1, int* m2,
                             unsigned hist_i = 0, bool include_overflows = true) {
        MALLOC_3D_VALUE_ARRAY(arr_out0, n0, n1, n2, $self->hist_size(include_overflows, 0),
                                                    $self->hist_size(include_overflows, 1),
                                                    $self->hist_size(include_overflows, 2), nbytes0)
        MALLOC_3D_VALUE_ARRAY(arr_out1, m0, m1, m2, $self->hist_size(include_overflows, 0),
                                                    $self->hist_size(include_overflows, 1),
                                                    $self->hist_size(include_overflows, 2), nbytes1)
        try {
          $self->get_hist_errs(*arr_out0, *arr_out1, hist_i, include_overflows);
        }
        catch (std::exception & e) {
          free(*arr_out0);
          free(*arr_out1);
          throw;
        }
      }
    }

    // declare histogram templates
    %template(EECHistBase1DId) EECHistBase<EECHist1D<axis::id>>;
    %template(EECHistBase1DLog) EECHistBase<EECHist1D<axis::log>>;
    %template(EECHistBaseIdIdId) EECHistBase<EECHist3D<axis::id, axis::id, axis::id>>;
    %template(EECHistBaseLogIdId) EECHistBase<EECHist3D<axis::log, axis::id, axis::id>>;
    %template(EECHistBaseIdLogId) EECHistBase<EECHist3D<axis::id, axis::log, axis::id>>;
    %template(EECHistBaseLogLogId) EECHistBase<EECHist3D<axis::log, axis::log, axis::id>>;
    %template(EECHist1DId) EECHist1D<axis::id>;
    %template(EECHist1DLog) EECHist1D<axis::log>;
    %template(EECHist3DIdIdId) EECHist3D<axis::id, axis::id, axis::id>;
    %template(EECHist3DLogIdId) EECHist3D<axis::log, axis::id, axis::id>;
    %template(EECHist3DIdLogId) EECHist3D<axis::id, axis::log, axis::id>;
    %template(EECHist3DLogLogId) EECHist3D<axis::log, axis::log, axis::id>;

  } // namespace hist
} // namespace EECNAMESPACE

// include EEC code and declare templates
%include "EECBase.hh"
%include "EECMultinomial.hh"
%include "EECLongestSide.hh"
%include "EECTriangleOPE.hh"

namespace EECNAMESPACE {

  %extend Multinomial {

    template<unsigned i>
    void py_set_index(unsigned ind) {
      if (i == 0 || i >= $self->N() - 1)
        throw std::out_of_range("trying to set invalid index");
      $self->set_index<i>(ind);
    }
  }

  // extend functionality to include numpy support
  %extend EECEvents {
    void add_event(double* particles_noconvert, int mult, int nfeatures, double weight = 1.0) {
      $self->append(particles_noconvert, mult, nfeatures, weight);
    }
  }

  %extend EECBase {
    ADD_STR_FROM_DESCRIPTION

    void npy_compute(double* particles, int mult, int nfeatures, double weight = 1.0, int thread_i = 0) {
      if (nfeatures != (int) $self->nfeatures()) {
        std::ostringstream oss;
        oss << "Got array with " << nfeatures << " features per particle, expected "
            << $self->nfeatures() << " features per particle";
        throw std::runtime_error(oss.str());
        return;
      }
      $self->compute(particles, mult, weight, thread_i);
    }

    // this is needed because we've hidden batch_compute
    void _batch_compute(const EECEvents & evs) {
      $self->batch_compute(evs);
    }

    %pythoncode %{

      def __call__(self, events, weights=None):
          import numpy as np

          if weights is None:
              weights = np.ones(len(events), order='C', dtype=np.double)
          elif len(weights) != len(events):
              raise ValueError('events and weights have different length')

          ncol = 4 if self.use_charges() else 3
          eecevents = EECEvents(len(events), self.nfeatures())
          events_arr = []
          for event,weight in zip(events, weights):
              event = np.asarray(np.atleast_2d(event)[:,:ncol], dtype=np.double, order='C')
              eecevents.add_event(event, weight)
              events_arr.append(event)

          self._batch_compute(eecevents)
    %}
  }

  %extend EECLongestSide {
    ADD_STR_FROM_DESCRIPTION
  }

  %extend EECTriangleOPE {
    ADD_STR_FROM_DESCRIPTION
  }

  // instantiate EEC templates
  %template(set_index_1) Multinomial::py_set_index<1>;
  %template(set_index_2) Multinomial::py_set_index<2>;
  %template(set_index_3) Multinomial::py_set_index<3>;
  %template(set_index_4) Multinomial::py_set_index<4>;
  %template(Multinomial2) Multinomial<2>;
  %template(Multinomial3) Multinomial<3>;
  %template(Multinomial4) Multinomial<4>;
  %template(Multinomial5) Multinomial<5>;
  %template(Multinomial6) Multinomial<6>;

  %template(EECLongestSideId) EECLongestSide<axis::id>;
  %template(EECLongestSideLog) EECLongestSide<axis::log>;
  %template(EECTriangleOPEIdIdId) EECTriangleOPE<axis::id, axis::id, axis::id>;
  %template(EECTriangleOPELogIdId) EECTriangleOPE<axis::log, axis::id, axis::id>;
  %template(EECTriangleOPEIdLogId) EECTriangleOPE<axis::id, axis::log, axis::id>;
  %template(EECTriangleOPELogLogId) EECTriangleOPE<axis::log, axis::log, axis::id>;

} // namespace EECNAMESPACE

%pythoncode %{

def EECLongestSide(*args, axis='log', **kwargs):

    axis_range = kwargs.pop('axis_range', None)
    if axis_range is not None:
        assert len(axis_range) == 2, '`axis_range` must be length 2'
        kwargs['axis_min'] = axis_range[0]
        kwargs['axis_max'] = axis_range[1]

    if axis.lower() == 'log':
        return EECLongestSideLog(*args, **kwargs)
    elif axis.lower() == 'id':
        return EECLongestSideId(*args, **kwargs)
    else:
        raise TypeError('axis `{}` not understood'.format(axis))

def EECTriangleOPE(*args, axes=('log', 'log', 'id'), **kwargs):

    axes = tuple(map(lambda x: x.lower(), axes))

    nbins = kwargs.pop('nbins', None)
    if nbins is not None:
        assert len(nbins) == 3, '`nbins` must be length 3'
        kwargs['nbins0'], kwargs['nbins1'], kwargs['nbins2'] = nbins

    axis_ranges = kwargs.pop('axis_ranges', None)
    if axis_ranges is not None:
        assert len(axis_ranges) == 3, '`axis_ranges` must be length 3'
        for i,axis_range in enumerate(axis_ranges):
            assert len(axis_range) == 2, 'axis_range ' + str(axis_range) + ' not length 2'
            kwargs['axis{}_min'.format(i)] = axis_range[0]
            kwargs['axis{}_max'.format(i)] = axis_range[1]

    if axes == ('log', 'log', 'id'):
        return EECTriangleOPELogLogId(*args, **kwargs)
    elif axes == ('id', 'log', 'id'):
        return EECTriangleOPEIdLogId(*args, **kwargs)
    elif axes == ('log', 'id', 'id'):
        return EECTriangleOPELogIdId(*args, **kwargs)
    elif axes == ('id', 'id', 'id'):
        return EECTriangleOPEIdIdId(*args, **kwargs)
    else:
        raise TypeError('axes `{}` not understood'.format(axes))

%}
