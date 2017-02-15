/* File: s20evalmodule.c
 * This file is auto-generated with f2py (version:2).
 * f2py is a Fortran to Python Interface Generator (FPIG), Second Edition,
 * written by Pearu Peterson <pearu@cens.ioc.ee>.
 * See http://cens.ioc.ee/projects/f2py2e/
 * Generation date: Wed Feb 15 21:17:55 2017
 * $Revision:$
 * $Date:$
 * Do not edit this file directly unless you know what you are doing!!!
 */

#ifdef __cplusplus
extern "C" {
#endif

/*********************** See f2py2e/cfuncs.py: includes ***********************/
#include "Python.h"
#include <stdarg.h>
#include "fortranobject.h"
#include <string.h>
#include <math.h>

/**************** See f2py2e/rules.py: mod_rules['modulebody'] ****************/
static PyObject *s20eval_error;
static PyObject *s20eval_module;

/*********************** See f2py2e/cfuncs.py: typedefs ***********************/
typedef char * string;

/****************** See f2py2e/cfuncs.py: typedefs_generated ******************/
/*need_typedefs_generated*/

/********************** See f2py2e/cfuncs.py: cppmacros **********************/
\
#define FAILNULL(p) do {                                            \
    if ((p) == NULL) {                                              \
        PyErr_SetString(PyExc_MemoryError, "NULL pointer found");   \
        goto capi_fail;                                             \
    }                                                               \
} while (0)

#define STRINGMALLOC(str,len)\
  if ((str = (string)malloc(sizeof(char)*(len+1))) == NULL) {\
    PyErr_SetString(PyExc_MemoryError, "out of memory");\
    goto capi_fail;\
  } else {\
    (str)[len] = '\0';\
  }

#if defined(PREPEND_FORTRAN)
#if defined(NO_APPEND_FORTRAN)
#if defined(UPPERCASE_FORTRAN)
#define F_FUNC(f,F) _##F
#else
#define F_FUNC(f,F) _##f
#endif
#else
#if defined(UPPERCASE_FORTRAN)
#define F_FUNC(f,F) _##F##_
#else
#define F_FUNC(f,F) _##f##_
#endif
#endif
#else
#if defined(NO_APPEND_FORTRAN)
#if defined(UPPERCASE_FORTRAN)
#define F_FUNC(f,F) F
#else
#define F_FUNC(f,F) f
#endif
#else
#if defined(UPPERCASE_FORTRAN)
#define F_FUNC(f,F) F##_
#else
#define F_FUNC(f,F) f##_
#endif
#endif
#endif
#if defined(UNDERSCORE_G77)
#define F_FUNC_US(f,F) F_FUNC(f##_,F##_)
#else
#define F_FUNC_US(f,F) F_FUNC(f,F)
#endif

#define rank(var) var ## _Rank
#define shape(var,dim) var ## _Dims[dim]
#define old_rank(var) (PyArray_NDIM((PyArrayObject *)(capi_ ## var ## _tmp)))
#define old_shape(var,dim) PyArray_DIM(((PyArrayObject *)(capi_ ## var ## _tmp)),dim)
#define fshape(var,dim) shape(var,rank(var)-dim-1)
#define len(var) shape(var,0)
#define flen(var) fshape(var,0)
#define old_size(var) PyArray_SIZE((PyArrayObject *)(capi_ ## var ## _tmp))
/* #define index(i) capi_i ## i */
#define slen(var) capi_ ## var ## _len
#define size(var, ...) f2py_size((PyArrayObject *)(capi_ ## var ## _tmp), ## __VA_ARGS__, -1)

#define STRINGFREE(str) do {if (!(str == NULL)) free(str);} while (0)

#define CHECKSCALAR(check,tcheck,name,show,var)\
  if (!(check)) {\
    char errstring[256];\
    sprintf(errstring, "%s: "show, "("tcheck") failed for "name, var);\
    PyErr_SetString(s20eval_error,errstring);\
    /*goto capi_fail;*/\
  } else 
#ifdef DEBUGCFUNCS
#define CFUNCSMESS(mess) fprintf(stderr,"debug-capi:"mess);
#define CFUNCSMESSPY(mess,obj) CFUNCSMESS(mess) \
  PyObject_Print((PyObject *)obj,stderr,Py_PRINT_RAW);\
  fprintf(stderr,"\n");
#else
#define CFUNCSMESS(mess)
#define CFUNCSMESSPY(mess,obj)
#endif

#ifndef max
#define max(a,b) ((a > b) ? (a) : (b))
#endif
#ifndef min
#define min(a,b) ((a < b) ? (a) : (b))
#endif
#ifndef MAX
#define MAX(a,b) ((a > b) ? (a) : (b))
#endif
#ifndef MIN
#define MIN(a,b) ((a < b) ? (a) : (b))
#endif

#define STRINGCOPYN(to,from,buf_size)                           \
    do {                                                        \
        int _m = (buf_size);                                    \
        char *_to = (to);                                       \
        char *_from = (from);                                   \
        FAILNULL(_to); FAILNULL(_from);                         \
        (void)strncpy(_to, _from, sizeof(char)*_m);             \
        _to[_m-1] = '\0';                                      \
        /* Padding with spaces instead of nulls */              \
        for (_m -= 2; _m >= 0 && _to[_m] == '\0'; _m--) {      \
            _to[_m] = ' ';                                      \
        }                                                       \
    } while (0)


/************************ See f2py2e/cfuncs.py: cfuncs ************************/
static int f2py_size(PyArrayObject* var, ...)
{
  npy_int sz = 0;
  npy_int dim;
  npy_int rank;
  va_list argp;
  va_start(argp, var);
  dim = va_arg(argp, npy_int);
  if (dim==-1)
    {
      sz = PyArray_SIZE(var);
    }
  else
    {
      rank = PyArray_NDIM(var);
      if (dim>=1 && dim<=rank)
        sz = PyArray_DIM(var, dim-1);
      else
        fprintf(stderr, "f2py_size: 2nd argument value=%d fails to satisfy 1<=value<=%d. Result will be 0.\n", dim, rank);
    }
  va_end(argp);
  return sz;
}

static int string_from_pyobj(string *str,int *len,const string inistr,PyObject *obj,const char *errmess) {
  PyArrayObject *arr = NULL;
  PyObject *tmp = NULL;
#ifdef DEBUGCFUNCS
fprintf(stderr,"string_from_pyobj(str='%s',len=%d,inistr='%s',obj=%p)\n",(char*)str,*len,(char *)inistr,obj);
#endif
  if (obj == Py_None) {
    if (*len == -1)
      *len = strlen(inistr); /* Will this cause problems? */
    STRINGMALLOC(*str,*len);
    STRINGCOPYN(*str,inistr,*len+1);
    return 1;
  }
  if (PyArray_Check(obj)) {
    if ((arr = (PyArrayObject *)obj) == NULL)
      goto capi_fail;
    if (!ISCONTIGUOUS(arr)) {
      PyErr_SetString(PyExc_ValueError,"array object is non-contiguous.");
      goto capi_fail;
    }
    if (*len == -1)
      *len = (PyArray_ITEMSIZE(arr))*PyArray_SIZE(arr);
    STRINGMALLOC(*str,*len);
    STRINGCOPYN(*str,PyArray_DATA(arr),*len+1);
    return 1;
  }
  if (PyString_Check(obj)) {
    tmp = obj;
    Py_INCREF(tmp);
  }
#if PY_VERSION_HEX >= 0x03000000
  else if (PyUnicode_Check(obj)) {
    tmp = PyUnicode_AsASCIIString(obj);
  }
  else {
    PyObject *tmp2;
    tmp2 = PyObject_Str(obj);
    if (tmp2) {
      tmp = PyUnicode_AsASCIIString(tmp2);
      Py_DECREF(tmp2);
    }
    else {
      tmp = NULL;
    }
  }
#else
  else {
    tmp = PyObject_Str(obj);
  }
#endif
  if (tmp == NULL) goto capi_fail;
  if (*len == -1)
    *len = PyString_GET_SIZE(tmp);
  STRINGMALLOC(*str,*len);
  STRINGCOPYN(*str,PyString_AS_STRING(tmp),*len+1);
  Py_DECREF(tmp);
  return 1;
capi_fail:
  Py_XDECREF(tmp);
  {
    PyObject* err = PyErr_Occurred();
    if (err==NULL) err = s20eval_error;
    PyErr_SetString(err,errmess);
  }
  return 0;
}

static int int_from_pyobj(int* v,PyObject *obj,const char *errmess) {
  PyObject* tmp = NULL;
  if (PyInt_Check(obj)) {
    *v = (int)PyInt_AS_LONG(obj);
    return 1;
  }
  tmp = PyNumber_Int(obj);
  if (tmp) {
    *v = PyInt_AS_LONG(tmp);
    Py_DECREF(tmp);
    return 1;
  }
  if (PyComplex_Check(obj))
    tmp = PyObject_GetAttrString(obj,"real");
  else if (PyString_Check(obj) || PyUnicode_Check(obj))
    /*pass*/;
  else if (PySequence_Check(obj))
    tmp = PySequence_GetItem(obj,0);
  if (tmp) {
    PyErr_Clear();
    if (int_from_pyobj(v,tmp,errmess)) {Py_DECREF(tmp); return 1;}
    Py_DECREF(tmp);
  }
  {
    PyObject* err = PyErr_Occurred();
    if (err==NULL) err = s20eval_error;
    PyErr_SetString(err,errmess);
  }
  return 0;
}


/********************* See f2py2e/cfuncs.py: userincludes *********************/
/*need_userincludes*/

/********************* See f2py2e/capi_rules.py: usercode *********************/


/* See f2py2e/rules.py */
extern void F_FUNC(sph2v,SPH2V)(int*,double*,double*,double*,double*,string,int*,size_t);
/*eof externroutines*/

/******************** See f2py2e/capi_rules.py: usercode1 ********************/


/******************* See f2py2e/cb_rules.py: buildcallback *******************/
/*need_callbacks*/

/*********************** See f2py2e/rules.py: buildapi ***********************/

/*********************************** sph2v ***********************************/
static char doc_f2py_rout_s20eval_sph2v[] = "\
dv_d = sph2v(lat,lon,dep,dv_d,mfl,wasread,[npoints])\n\nWrapper for ``sph2v``.\
\n\nParameters\n----------\n"
"lat : input rank-1 array('d') with bounds (npoints)\n"
"lon : input rank-1 array('d') with bounds (npoints)\n"
"dep : input rank-1 array('d') with bounds (npoints)\n"
"dv_d : input rank-1 array('d') with bounds (npoints)\n"
"mfl : input string(len=80)\n"
"wasread : input int\n"
"\nOther Parameters\n----------------\n"
"npoints : input int, optional\n    Default: len(lat)\n"
"\nReturns\n-------\n"
"dv_d : rank-1 array('d') with bounds (npoints)";
/* extern void F_FUNC(sph2v,SPH2V)(int*,double*,double*,double*,double*,string,int*,size_t); */
static PyObject *f2py_rout_s20eval_sph2v(const PyObject *capi_self,
                           PyObject *capi_args,
                           PyObject *capi_keywds,
                           void (*f2py_func)(int*,double*,double*,double*,double*,string,int*,size_t)) {
  PyObject * volatile capi_buildvalue = NULL;
  volatile int f2py_success = 1;
/*decl*/

  int npoints = 0;
  PyObject *npoints_capi = Py_None;
  double *lat = NULL;
  npy_intp lat_Dims[1] = {-1};
  const int lat_Rank = 1;
  PyArrayObject *capi_lat_tmp = NULL;
  int capi_lat_intent = 0;
  PyObject *lat_capi = Py_None;
  double *lon = NULL;
  npy_intp lon_Dims[1] = {-1};
  const int lon_Rank = 1;
  PyArrayObject *capi_lon_tmp = NULL;
  int capi_lon_intent = 0;
  PyObject *lon_capi = Py_None;
  double *dep = NULL;
  npy_intp dep_Dims[1] = {-1};
  const int dep_Rank = 1;
  PyArrayObject *capi_dep_tmp = NULL;
  int capi_dep_intent = 0;
  PyObject *dep_capi = Py_None;
  double *dv_d = NULL;
  npy_intp dv_d_Dims[1] = {-1};
  const int dv_d_Rank = 1;
  PyArrayObject *capi_dv_d_tmp = NULL;
  int capi_dv_d_intent = 0;
  PyObject *dv_d_capi = Py_None;
  string mfl = NULL;
  int slen(mfl);
  PyObject *mfl_capi = Py_None;
  int wasread = 0;
  PyObject *wasread_capi = Py_None;
  static char *capi_kwlist[] = {"lat","lon","dep","dv_d","mfl","wasread","npoints",NULL};

/*routdebugenter*/
#ifdef F2PY_REPORT_ATEXIT
f2py_start_clock();
#endif
  if (!PyArg_ParseTupleAndKeywords(capi_args,capi_keywds,\
    "OOOOOO|O:s20eval.sph2v",\
    capi_kwlist,&lat_capi,&lon_capi,&dep_capi,&dv_d_capi,&mfl_capi,&wasread_capi,&npoints_capi))
    return NULL;
/*frompyobj*/
  /* Processing variable lat */
  ;
  capi_lat_intent |= F2PY_INTENT_IN;
  capi_lat_tmp = array_from_pyobj(NPY_DOUBLE,lat_Dims,lat_Rank,capi_lat_intent,lat_capi);
  if (capi_lat_tmp == NULL) {
    if (!PyErr_Occurred())
      PyErr_SetString(s20eval_error,"failed in converting 1st argument `lat' of s20eval.sph2v to C/Fortran array" );
  } else {
    lat = (double *)(PyArray_DATA(capi_lat_tmp));

  /* Processing variable mfl */
  slen(mfl) = 80;
  f2py_success = string_from_pyobj(&mfl,&slen(mfl),"",mfl_capi,"string_from_pyobj failed in converting 5th argument `mfl' of s20eval.sph2v to C string");
  if (f2py_success) {
  /* Processing variable wasread */
    wasread = (int)PyObject_IsTrue(wasread_capi);
    f2py_success = 1;
  if (f2py_success) {
  /* Processing variable npoints */
  if (npoints_capi == Py_None) npoints = len(lat); else
    f2py_success = int_from_pyobj(&npoints,npoints_capi,"s20eval.sph2v() 1st keyword (npoints) can't be converted to int");
  if (f2py_success) {
  CHECKSCALAR(len(lat)>=npoints,"len(lat)>=npoints","1st keyword npoints","sph2v:npoints=%d",npoints) {
  /* Processing variable lon */
  lon_Dims[0]=npoints;
  capi_lon_intent |= F2PY_INTENT_IN;
  capi_lon_tmp = array_from_pyobj(NPY_DOUBLE,lon_Dims,lon_Rank,capi_lon_intent,lon_capi);
  if (capi_lon_tmp == NULL) {
    if (!PyErr_Occurred())
      PyErr_SetString(s20eval_error,"failed in converting 2nd argument `lon' of s20eval.sph2v to C/Fortran array" );
  } else {
    lon = (double *)(PyArray_DATA(capi_lon_tmp));

  /* Processing variable dep */
  dep_Dims[0]=npoints;
  capi_dep_intent |= F2PY_INTENT_IN;
  capi_dep_tmp = array_from_pyobj(NPY_DOUBLE,dep_Dims,dep_Rank,capi_dep_intent,dep_capi);
  if (capi_dep_tmp == NULL) {
    if (!PyErr_Occurred())
      PyErr_SetString(s20eval_error,"failed in converting 3rd argument `dep' of s20eval.sph2v to C/Fortran array" );
  } else {
    dep = (double *)(PyArray_DATA(capi_dep_tmp));

  /* Processing variable dv_d */
  dv_d_Dims[0]=npoints;
  capi_dv_d_intent |= F2PY_INTENT_IN|F2PY_INTENT_OUT;
  capi_dv_d_tmp = array_from_pyobj(NPY_DOUBLE,dv_d_Dims,dv_d_Rank,capi_dv_d_intent,dv_d_capi);
  if (capi_dv_d_tmp == NULL) {
    if (!PyErr_Occurred())
      PyErr_SetString(s20eval_error,"failed in converting 4th argument `dv_d' of s20eval.sph2v to C/Fortran array" );
  } else {
    dv_d = (double *)(PyArray_DATA(capi_dv_d_tmp));

/*end of frompyobj*/
#ifdef F2PY_REPORT_ATEXIT
f2py_start_call_clock();
#endif
/*callfortranroutine*/
        (*f2py_func)(&npoints,lat,lon,dep,dv_d,mfl,&wasread,slen(mfl));
if (PyErr_Occurred())
  f2py_success = 0;
#ifdef F2PY_REPORT_ATEXIT
f2py_stop_call_clock();
#endif
/*end of callfortranroutine*/
    if (f2py_success) {
/*pyobjfrom*/
/*end of pyobjfrom*/
    CFUNCSMESS("Building return value.\n");
    capi_buildvalue = Py_BuildValue("N",capi_dv_d_tmp);
/*closepyobjfrom*/
/*end of closepyobjfrom*/
    } /*if (f2py_success) after callfortranroutine*/
/*cleanupfrompyobj*/
  }  /*if (capi_dv_d_tmp == NULL) ... else of dv_d*/
  /* End of cleaning variable dv_d */
  if((PyObject *)capi_dep_tmp!=dep_capi) {
    Py_XDECREF(capi_dep_tmp); }
  }  /*if (capi_dep_tmp == NULL) ... else of dep*/
  /* End of cleaning variable dep */
  if((PyObject *)capi_lon_tmp!=lon_capi) {
    Py_XDECREF(capi_lon_tmp); }
  }  /*if (capi_lon_tmp == NULL) ... else of lon*/
  /* End of cleaning variable lon */
  } /*CHECKSCALAR(len(lat)>=npoints)*/
  } /*if (f2py_success) of npoints*/
  /* End of cleaning variable npoints */
  } /*if (f2py_success) of wasread*/
  /* End of cleaning variable wasread */
    STRINGFREE(mfl);
  }  /*if (f2py_success) of mfl*/
  /* End of cleaning variable mfl */
  if((PyObject *)capi_lat_tmp!=lat_capi) {
    Py_XDECREF(capi_lat_tmp); }
  }  /*if (capi_lat_tmp == NULL) ... else of lat*/
  /* End of cleaning variable lat */
/*end of cleanupfrompyobj*/
  if (capi_buildvalue == NULL) {
/*routdebugfailure*/
  } else {
/*routdebugleave*/
  }
  CFUNCSMESS("Freeing memory.\n");
/*freemem*/
#ifdef F2PY_REPORT_ATEXIT
f2py_stop_clock();
#endif
  return capi_buildvalue;
}
/******************************** end of sph2v ********************************/
/*eof body*/

/******************* See f2py2e/f90mod_rules.py: buildhooks *******************/
/*need_f90modhooks*/

/************** See f2py2e/rules.py: module_rules['modulebody'] **************/

/******************* See f2py2e/common_rules.py: buildhooks *******************/

/*need_commonhooks*/

/**************************** See f2py2e/rules.py ****************************/

static FortranDataDef f2py_routine_defs[] = {
  {"sph2v",-1,{{-1}},0,(char *)F_FUNC(sph2v,SPH2V),(f2py_init_func)f2py_rout_s20eval_sph2v,doc_f2py_rout_s20eval_sph2v},

/*eof routine_defs*/
  {NULL}
};

static PyMethodDef f2py_module_methods[] = {

  {NULL,NULL}
};

#if PY_VERSION_HEX >= 0x03000000
static struct PyModuleDef moduledef = {
  PyModuleDef_HEAD_INIT,
  "s20eval",
  NULL,
  -1,
  f2py_module_methods,
  NULL,
  NULL,
  NULL,
  NULL
};
#endif

#if PY_VERSION_HEX >= 0x03000000
#define RETVAL m
PyMODINIT_FUNC PyInit_s20eval(void) {
#else
#define RETVAL
PyMODINIT_FUNC inits20eval(void) {
#endif
  int i;
  PyObject *m,*d, *s;
#if PY_VERSION_HEX >= 0x03000000
  m = s20eval_module = PyModule_Create(&moduledef);
#else
  m = s20eval_module = Py_InitModule("s20eval", f2py_module_methods);
#endif
  Py_TYPE(&PyFortran_Type) = &PyType_Type;
  import_array();
  if (PyErr_Occurred())
    {PyErr_SetString(PyExc_ImportError, "can't initialize module s20eval (failed to import numpy)"); return RETVAL;}
  d = PyModule_GetDict(m);
  s = PyString_FromString("$Revision: $");
  PyDict_SetItemString(d, "__version__", s);
#if PY_VERSION_HEX >= 0x03000000
  s = PyUnicode_FromString(
#else
  s = PyString_FromString(
#endif
    "This module 's20eval' is auto-generated with f2py (version:2).\nFunctions:\n"
"  dv_d = sph2v(lat,lon,dep,dv_d,mfl,wasread,npoints=len(lat))\n"
".");
  PyDict_SetItemString(d, "__doc__", s);
  s20eval_error = PyErr_NewException ("s20eval.error", NULL, NULL);
  Py_DECREF(s);
  for(i=0;f2py_routine_defs[i].name!=NULL;i++)
    PyDict_SetItemString(d, f2py_routine_defs[i].name,PyFortranObject_NewAsAttr(&f2py_routine_defs[i]));

/*eof initf2pywraphooks*/
/*eof initf90modhooks*/

/*eof initcommonhooks*/


#ifdef F2PY_REPORT_ATEXIT
  if (! PyErr_Occurred())
    on_exit(f2py_report_on_exit,(void*)"s20eval");
#endif

  return RETVAL;
}
#ifdef __cplusplus
}
#endif