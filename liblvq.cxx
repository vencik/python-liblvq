/**
 *  liblvq Python binding
 *
 *  \date    2015/07/24
 *  \author  Vaclav Krpec  <vencik@razdva.cz>
 *
 *
 *  LEGAL NOTICE
 *
 *  Copyright (c) 2015, Vaclav Krpec
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in
 *     the documentation and/or other materials provided with the distribution.
 *
 *  3. Neither the name of the copyright holder nor the names of
 *     its contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 *  THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 *  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER
 *  OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 *  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 *  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 *  OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 *  WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 *  OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 *  EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */


#include <liblvq/ml/lvq.hxx>
#include <liblvq/math/R_undef.hxx>

#include <Python.h>
#include <structmember.h>


/** Base numeric type */
typedef double base_t;

/** Extended real number type */
typedef math::realx<base_t> realx_t;

/** LVQ */
typedef ml::lvq<realx_t> lvq_t;

/** Training set */
typedef std::vector<std::pair<lvq_t::input_t, size_t> > training_set_t;


/** LVQ Python object */
typedef struct {
    PyObject_HEAD
    lvq_t * lvq;
} lvqObject_t;

/** LVQ object access */
#define python2lvq(self) \
    ((reinterpret_cast<lvqObject_t *>(self))->lvq)


/**
 *  \brief  Exception-safe wrapper for bindings
 *
 *  \tparam R_t  Function return type
 *  \tparam F_t  Function
 *  \tparam A_t  Argument types
 *
 *  \param  xres  Return value in case of exception
 *  \param  fn    Function
 *  \param  args  Function arguments
 *
 *  \return \c fn(args) or \c eres in case of an exception
 */
template <typename R_t, typename F_t, typename... A_t>
R_t wrap_X(R_t xres, F_t fn, A_t... args) {
    R_t res = xres;

    try {
        res = fn(args...);
    }
    catch (std::exception & x) {
        PyErr_SetString(PyExc_RuntimeError, x.what());
    }
    catch (...) {
        PyErr_SetString(PyExc_RuntimeError, "Unknown exception");
    }

    return res;
}

/**
 *  \brief  Exception-safe binding wrapper instance identifier
 *
 *  \param  binding  Binding function identifier
 */
#define BINDING_IDENT(binding) binding ## __wrap_X

/** Exception-safe binding wrapper instance */
#define BINDING_INST(binding) \
static PyObject * BINDING_IDENT(binding)(PyObject * self, PyObject * args) { \
    return wrap_X((PyObject *)NULL, binding, self, args); \
}


/**
 *  \brief  Parse bindings arguments
 *
 *  \tparam A_t  Arguments type
 *
 *  \param  py_args  Python arguments
 *  \param  fstr     Argument types format string
 *  \param  args     Parsed arguments
 */
template <typename... A_t>
static void parse_args(PyObject * py_args, const char * fstr, A_t... args) {
    PyArg_ParseTuple(py_args, fstr, args...);

    if (NULL != PyErr_Occurred())
        throw std::logic_error("Invalid arguments");
}


/**
 *  \brief  Transform Python weight sequence to \c std::vector
 *
 *  \param  py_weight  Python weight sequence
 *
 *  \return Weight vector
 */
static std::vector<double> python2weight(PyObject * py_weight) {
    Py_ssize_t weight_size = PyObject_Size(py_weight);
    if (-1 == weight_size)
        throw std::logic_error("Invalid weights (can't get size)");

    PyObject * py_iter = PyObject_GetIter(py_weight);
    if (NULL == py_iter)
        throw std::logic_error("Invalid weights (should be iterable)");

    std::vector<double> weight;
    weight.reserve(weight_size);

    PyObject * py_w;
    while (NULL != (py_w = PyIter_Next(py_iter))) {
        weight.push_back(PyFloat_AsDouble(py_w));

        if (NULL != PyErr_Occurred())
            throw std::logic_error("Invalid weight (should be double)");

        Py_DECREF(py_w);
    }

    Py_DECREF(py_iter);

    return weight;
}


/**
 *  \brief  Transform weight std::vector to Python tuple
 *
 *  \param  weight  Weight vector
 *
 *  \return Python weight tuple
 */
static PyObject * weight2python(const std::vector<double> & weight) {
    size_t     weight_size = weight.size();
    PyObject * py_weight   = PyTuple_New(weight_size);

    for (size_t i = 0; i < weight_size; ++i) {
        PyTuple_SetItem(py_weight, i, Py_BuildValue("d", weight[i]));
    }

    return py_weight;
}


/**
 *  \brief  Transform Python tuple of numbers to \c lvq_t::input_t
 *
 *  \param  py_input  Python tuple of numbers
 *
 *  \return \c lvq_t::input_t
 */
static lvq_t::input_t python2input(PyObject * py_input) {
    Py_ssize_t input_size = PyObject_Size(py_input);
    if (-1 == input_size)
        throw std::logic_error("Invalid input (can't get size)");

    PyObject * py_iter = PyObject_GetIter(py_input);
    if (NULL == py_iter)
        throw std::logic_error("Invalid input (should be iterable)");

    lvq_t::input_t input(input_size);

    PyObject * py_x;
    for (size_t i = 0; NULL != (py_x = PyIter_Next(py_iter)); ++i) {
        input[i] = Py_None == py_x
                 ? lvq_t::base_t::undef
                 : lvq_t::base_t(PyFloat_AsDouble(py_x));

        if (NULL != PyErr_Occurred())
            throw std::logic_error("Invalid input value");

        Py_DECREF(py_x);
    }

    Py_DECREF(py_iter);

    return input;
}


/**
 *  \brief  Transform \c lvq_t::input_t to Python tuple of numbers
 *
 *  \param  input  \c lvq_t::input_t instance
 *
 *  \return Python tuple
 */
static PyObject * input2python(const lvq_t::input_t & input) {
    size_t     input_rank = input.rank();
    PyObject * py_input   = PyTuple_New(input_rank);

    for (size_t i = 0; i < input_rank; ++i) {
        PyObject * x;

        if (input[i].is_defined())
            x = Py_BuildValue("d", input[i]);
        else {
            Py_INCREF(Py_None);
            x = Py_None;
        }

        PyTuple_SetItem(py_input, i, x);
    }

    return py_input;
}


/**
 *  \brief  Transform [cluster, weight] pairs vector to Python representation
 *
 *  \param  cw_vec  [cluster, weight] pairs vector
 *
 *  \return Python list of [cluster, weight] tuples
 */
static PyObject * cw_vec2python(const std::vector<lvq_t::cw_t> & cw_vec) {
    size_t     cw_vec_size = cw_vec.size();
    PyObject * py_cw_tuple = PyTuple_New(cw_vec_size);

    for (size_t i = 0; i < cw_vec_size; ++i) {
        PyTuple_SetItem(py_cw_tuple, i,
            Py_BuildValue("(nd)", cw_vec[i].first, cw_vec[i].second));
    }

    return py_cw_tuple;
}


/**
 *  \brief  Transform Python training set to \c lvq_t::input_t set
 *
 *  The Python training set is an iterable containing (input, cluster) tuples.
 *
 *  \param  py_set  Python iterable of number tuples
 *
 *  \return Training set
 */
static training_set_t python2training_set(PyObject * py_set) {
    Py_ssize_t set_size = PyObject_Size(py_set);
    if (-1 == set_size)
        throw std::logic_error("Invalid training set (can't get size)");

    PyObject * py_iter = PyObject_GetIter(py_set);
    if (NULL == py_iter)
        throw std::logic_error("Invalid training set (should be iterable)");

    training_set_t set;
    set.reserve(set_size);

    PyObject * py_ic;
    while (NULL != (py_ic = PyIter_Next(py_iter))) {
        if (!PyTuple_Check(py_ic))
            throw std::logic_error(
                "Invalid training set ((input, cluster) tuples expected");

        PyObject * py_input   = PyTuple_GetItem(py_ic, 0);
        PyObject * py_cluster = PyTuple_GetItem(py_ic, 1);

        const lvq_t::input_t input = python2input(py_input);

        if (!PyInt_Check(py_cluster))
            throw std::logic_error("Invalid cluster (integer expected)");

        Py_ssize_t cluster = PyInt_AsSsize_t(py_cluster);

        if (0 > cluster)
            throw std::logic_error("Invalid cluster (must be >= 0)");

        set.emplace_back(input, cluster);

        Py_DECREF(py_ic);
    }

    Py_DECREF(py_iter);

    return set;
}


/**
 *  \brief  Constructor
 *
 *  \param  py_lvq  Python LVQ object
 *  \param  args    Arguments
 *  \param  kwds    Keywords
 */
static void liblvq__lvq__create(
    lvqObject_t * py_lvq,
    PyObject    * args,
    PyObject    * kwds)
{
    // Get arguments
    size_t dimension;
    size_t clusters;
    parse_args(args, "nn", &dimension, &clusters);

    // Create ml::lvq instance
    py_lvq->lvq = new lvq_t(dimension, clusters);
}


/**
 *  \brief  Destructor
 *
 *  \param  py_lvq  Python LVQ object
 *
 *  \return 0
 */
static int liblvq__lvq__destroy(lvqObject_t * py_lvq) {
    lvq_t * lvq = py_lvq->lvq;
    py_lvq->lvq = NULL;

    if (NULL != lvq) delete lvq;

    return 0;
}

/** \cond */
static void BINDING_IDENT(liblvq__lvq__destroy)(lvqObject_t * py_lvq) {
    wrap_X(0, liblvq__lvq__destroy, py_lvq);
}
/** \endcond */


//
// ml::lvq member functions binding
//

/**
 *  \brief  Constructor
 *
 *  \param  type  Python LVQ type
 *  \param  args  Arguments
 *  \param  kwds  Keywords
 *
 *  \return LVQ instance
 */
static PyObject * liblvq__lvq__new(
    PyTypeObject * type,
    PyObject     * args,
    PyObject     * kwds)
{
    lvqObject_t * py_lvq = reinterpret_cast<lvqObject_t *>(
        type->tp_alloc(type, 0));

    if (NULL == py_lvq) return NULL;

    liblvq__lvq__create(py_lvq, args, kwds);

    return reinterpret_cast<PyObject *>(py_lvq);
}

/** \cond */
static PyObject * BINDING_IDENT(liblvq__lvq__new)(
    PyTypeObject * type,
    PyObject     * args,
    PyObject     * kwds)
{
    return wrap_X((PyObject *)NULL, liblvq__lvq__new, type, args, kwds);
}
/** \endcond */


/**
 *  \brief  Constructor (__init__)
 *
 *  \param  self  Python LQV object
 *  \param  args  Arguments
 *  \param  kwds  Keywords
 *
 *  \return 0
 */
static int liblvq__lvq__init(
    PyObject * self,
    PyObject * args,
    PyObject * kwds)
{
    lvqObject_t * py_lvq = reinterpret_cast<lvqObject_t *>(self);

    liblvq__lvq__destroy(py_lvq);

    liblvq__lvq__create(py_lvq, args, kwds);

    return 0;
}

/** \cond */
static int BINDING_IDENT(liblvq__lvq__init)(
    PyObject * self,
    PyObject * args,
    PyObject * kwds)
{
    return wrap_X((int)1, liblvq__lvq__init, self, args, kwds);
}
/** \endcond */


/**
 *  \brief  ml::lvq::set binding
 */
static PyObject * liblvq__lvq__set(PyObject * self, PyObject * args) {
    // Get arguments
    PyObject * py_input;
    size_t     cluster;
    parse_args(args, "On", &py_input, &cluster);

    const lvq_t::input_t input = python2input(py_input);

    // Call implementation
    python2lvq(self)->set(input, cluster);

    // No return value
    Py_INCREF(Py_None);
    return Py_None;
}

BINDING_INST(liblvq__lvq__set)


/**
 *  \brief  ml::lvq::get binding
 */
static PyObject * liblvq__lvq__get(PyObject * self, PyObject * args) {
    // Get arguments
    size_t cluster;
    parse_args(args, "n", &cluster);

    // Call implementation
    const lvq_t::input_t & representant = python2lvq(self)->get(cluster);

    // Transform result
    return input2python(representant);
}

BINDING_INST(liblvq__lvq__get)


/**
 *  \brief  \c ml::lvq::train1 binding
 */
static PyObject * liblvq__lvq__train1(PyObject * self, PyObject * args) {
    // Get arguments
    PyObject *    py_input;
    size_t        cluster;
    lvq_t::base_t lfactor;
    parse_args(args, "Ond", &py_input, &cluster, &lfactor);

    const lvq_t::input_t input = python2input(py_input);

    // Call implementation
    lvq_t::base_t dnorm2 = python2lvq(self)->train1(input, cluster, lfactor);

    // Transform result
    return Py_BuildValue("d", dnorm2);
}

BINDING_INST(liblvq__lvq__train1)


/**
 *  \brief  \c ml::lvq::train binding
 */
static PyObject * liblvq__lvq__train(PyObject * self, PyObject * args) {
    // Get arguments
    PyObject * py_set;
    unsigned   conv_win;
    unsigned   max_div_cnt;
    unsigned   max_tlc;
    parse_args(args, "OIII", &py_set, &conv_win, &max_div_cnt, &max_tlc);

    const training_set_t set = python2training_set(py_set);

    // Call implementation
    python2lvq(self)->train(set, conv_win, max_div_cnt, max_tlc);

    Py_INCREF(Py_None);
    return Py_None;
}

BINDING_INST(liblvq__lvq__train)


/**
 *  \brief  \c ml::lvq::classify binding
 */
static PyObject * liblvq__lvq__classify(PyObject * self, PyObject * args) {
    // Get arguments
    PyObject * py_input;
    parse_args(args, "O", &py_input);

    const lvq_t::input_t input = python2input(py_input);

    // Call implementation
    size_t cluster = python2lvq(self)->classify(input);

    // Transform result
    return Py_BuildValue("n", cluster);
}

BINDING_INST(liblvq__lvq__classify)


/**
 *  \brief  \c ml::lvq::classify_weight binding
 */
static PyObject * liblvq__lvq__classify_weight(
    PyObject * self,
    PyObject * args)
{
    // Get arguments
    PyObject * py_input;
    parse_args(args, "O", &py_input);

    const lvq_t::input_t input = python2input(py_input);

    // Call implementation
    std::vector<double> weight = python2lvq(self)->classify_weight(input);

    // Transform result
    return weight2python(weight);
}

BINDING_INST(liblvq__lvq__classify_weight)


/**
 *  \brief  ml::lvq::best binding
 */
static PyObject * liblvq__lvq__best(PyObject * self, PyObject * args) {
    // Get arguments
    PyObject * py_weight;
    size_t n;
    parse_args(args, "On", &py_weight, &n);

    const std::vector<double> weight = python2weight(py_weight);

    // Call implementation
    const std::vector<lvq_t::cw_t> cw_vec = lvq_t::best(weight, n);

    // Transform result
    return cw_vec2python(cw_vec);
}

BINDING_INST(liblvq__lvq__best)


/**
 *  \brief  \c ml::lvq::classify_best binding
 */
static PyObject * liblvq__lvq__classify_best(PyObject * self, PyObject * args) {
    // Get arguments
    PyObject * py_input;
    size_t     n;
    parse_args(args, "On", &py_input, &n);

    const lvq_t::input_t input = python2input(py_input);

    // Call implementation
    std::vector<lvq_t::cw_t> cw_vec = python2lvq(self)->classify_best(input, n);

    // Transform result
    return cw_vec2python(cw_vec);
}

BINDING_INST(liblvq__lvq__classify_best)


/**
 *  \brief  \c ml::lvq::weight_threshold binding
 */
static PyObject * liblvq__lvq__weight_threshold(
    PyObject * self,
    PyObject * args)
{
    PyObject * py_weight;
    double     wthres;
    parse_args(args, "Od", &py_weight, &wthres);

    const std::vector<double> weight = python2weight(py_weight);

    // Call implementation
    const std::vector<lvq_t::cw_t> cw_vec =
        lvq_t::weight_threshold(weight, wthres);

    // Transform result
    return cw_vec2python(cw_vec);
}

BINDING_INST(liblvq__lvq__weight_threshold)


/**
 *  \brief  \c ml::lvq::classify_weight_threshold binding
 */
static PyObject * liblvq__lvq__classify_weight_threshold(
    PyObject * self,
    PyObject * args)
{
    // Get arguments
    PyObject * py_input;
    double     wthres;
    parse_args(args, "Od", &py_input, &wthres);

    const lvq_t::input_t input = python2input(py_input);

    // Call implementation
    std::vector<lvq_t::cw_t> cw_vec =
        python2lvq(self)->classify_weight_threshold(input, wthres);

    // Transform result
    return cw_vec2python(cw_vec);
}

BINDING_INST(liblvq__lvq__classify_weight_threshold)


/**
 *  \brief  \c ml::lvq::learn_rate binding
 */
static PyObject * liblvq__lvq__learn_rate(PyObject * self, PyObject * args) {
    // Get arguments
    PyObject * py_set;
    parse_args(args, "O", &py_set);

    const training_set_t set = python2training_set(py_set);

    // Call implementation
    float lrate = python2lvq(self)->learn_rate(set);

    // Transform result
    return Py_BuildValue("f", lrate);
}

BINDING_INST(liblvq__lvq__learn_rate)


//
// Static data
//

/** LVQ data members */
static PyMemberDef lvqObject_attrs[] = {
    { NULL, 0, 0, 0, NULL }  // sentinel
};  // end of lvqObject_members


/** LVQ member functions */
static PyMethodDef lvqObject_methods[] = {
    {
        "set",
        BINDING_IDENT(liblvq__lvq__set),
        METH_VARARGS,
        "Set cluster representant"
    },
    {
        "get",
        BINDING_IDENT(liblvq__lvq__get),
        METH_VARARGS,
        "Get cluster representant"
    },
    {
        "train1",
        BINDING_IDENT(liblvq__lvq__train1),
        METH_VARARGS,
        "Training step"
    },
    {
        "train",
        BINDING_IDENT(liblvq__lvq__train),
        METH_VARARGS,
        "Train set"
    },
    {
        "classify",
        BINDING_IDENT(liblvq__lvq__classify),
        METH_VARARGS,
        "n-ary classification"
    },
    {
        "classify_weight",
        BINDING_IDENT(liblvq__lvq__classify_weight),
        METH_VARARGS,
        "Weighed classification"
    },
    {
        "best",
        BINDING_IDENT(liblvq__lvq__best),
        METH_VARARGS | METH_STATIC,
        "N best matching clusters"
    },
    {
        "classify_best",
        BINDING_IDENT(liblvq__lvq__classify_best),
        METH_VARARGS,
        "Classify to best matching clusters"
    },
    {
        "weight_threshold",
        BINDING_IDENT(liblvq__lvq__weight_threshold),
        METH_VARARGS | METH_STATIC,
        "Weight threshold reaching clusters"
    },
    {
        "classify_weight_threshold",
        BINDING_IDENT(liblvq__lvq__classify_weight_threshold),
        METH_VARARGS,
        "Classify to weight threshold"
    },
    {
        "learn_rate",
        BINDING_IDENT(liblvq__lvq__learn_rate),
        METH_VARARGS,
        "Compute learn rate of a training set"
    },

    { NULL, NULL, 0, NULL }  // sentinel
};  // end of lvqObject_methods


/** LVQ Python type */
static PyTypeObject lvqType = {
    PyObject_HEAD_INIT(NULL)

    /* ob_size          */  0,
    /* tp_name          */  "liblvq.lvq",
    /* tp_basicsize     */  sizeof(lvqObject_t),
    /* tp_itemsize      */  0,
    /* tp_dealloc       */  (destructor)BINDING_IDENT(liblvq__lvq__destroy),
    /* tp_print         */  0,
    /* tp_getattr       */  0,
    /* tp_setattr       */  0,
    /* tp_compare       */  0,
    /* tp_repr          */  0,
    /* tp_as_number     */  0,
    /* tp_as_sequence   */  0,
    /* tp_as_mapping    */  0,
    /* tp_hash          */  0,
    /* tp_call          */  0,
    /* tp_str           */  0,
    /* tp_getattro      */  0,
    /* tp_setattro      */  0,
    /* tp_as_buffer     */  0,
    /* tp_flags         */  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    /* tp_doc           */  "lvq objects",
    /* tp_traverse      */  0,
    /* tp_clear         */  0,
    /* tp_richcompare   */  0,
    /* tp_weaklistoffset*/  0,
    /* tp_iter          */  0,
    /* tp_iternext      */  0,
    /* tp_methods       */  lvqObject_methods,
    /* tp_members       */  lvqObject_attrs,
    /* tp_getset        */  0,
    /* tp_base          */  0,
    /* tp_dict          */  0,
    /* tp_descr_get     */  0,
    /* tp_descr_set     */  0,
    /* tp_dictoffset    */  0,
    /* tp_init          */  BINDING_IDENT(liblvq__lvq__init),
    /* tp_alloc         */  0,
    /* tp_new           */  BINDING_IDENT(liblvq__lvq__new),

};  // end of lvqType


/** Module member functions */
static PyMethodDef liblvq__methods[] = {

    { NULL, NULL, 0, NULL }  // sentinel
};  // end of liblvq__methods


/** Module initialiser */
PyMODINIT_FUNC
initliblvq(void) {
    if (PyType_Ready(&lvqType) < 0) return;

    PyObject * module = Py_InitModule3(
        "liblvq",
        liblvq__methods,
        "liblvq extension module");

    if (NULL == module) return;

    Py_INCREF(&lvqType);
    PyModule_AddObject(module, "lvq", (PyObject *)&lvqType);
}
