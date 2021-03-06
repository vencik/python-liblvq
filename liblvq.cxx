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

#include <cstdint>
#include <cstdlib>


/** Base numeric type */
typedef double base_t;

/** Extended real number type */
typedef math::realx<base_t> realx_t;

/** LVQ */
typedef ml::lvq<realx_t> lvq_t;

/** Classifier training/test set */
typedef lvq_t::tset_classifier tset_classifier_t;

/** Clustering training/test set */
typedef lvq_t::tset_clustering tset_clustering_t;

/** LVQ classifier statistics */
typedef lvq_t::classifier_statistics lvq_classifier_stats_t;

/** LVQ clustering statistics */
typedef lvq_t::clustering_statistics lvq_clustering_stats_t;


/** LVQ Python object */
typedef struct {
    PyObject_HEAD
    lvq_t * lvq;
} lvqObject_t;

/** LVQ object access */
#define python2lvq(self) \
    ((reinterpret_cast<lvqObject_t *>(self))->lvq)


/** LVQ classifier statistics Python object */
typedef struct {
    PyObject_HEAD
    lvq_classifier_stats_t * lvq_stats;
} lvqClassifierStatisticsObject_t;

/** LVQ classifier statistics object access */
#define python2lvq_classifier_stats(self) \
    ((reinterpret_cast<lvqClassifierStatisticsObject_t *>(self))->lvq_stats)


/** LVQ clustering statistics Python object */
typedef struct {
    PyObject_HEAD
    lvq_clustering_stats_t * lvq_stats;
} lvqClusteringStatisticsObject_t;

/** LVQ clustering statistics object access */
#define python2lvq_clustering_stats(self) \
    ((reinterpret_cast<lvqClusteringStatisticsObject_t *>(self))->lvq_stats)


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
 *  \brief  Transform Python training/test set to \c lvq_t::tset_classifier_t set
 *
 *  The Python training set is an iterable containing (input, cluster) tuples.
 *
 *  \param  py_set  Python iterable of number tuples
 *
 *  \return Training set
 */
static tset_classifier_t python2tset_classifier(PyObject * py_set) {
    Py_ssize_t set_size = PyObject_Size(py_set);
    if (-1 == set_size)
        throw std::logic_error("Invalid training set (can't get size)");

    PyObject * py_iter = PyObject_GetIter(py_set);
    if (NULL == py_iter)
        throw std::logic_error("Invalid training set (should be iterable)");

    tset_classifier_t set;

    PyObject * py_ic;
    while (NULL != (py_ic = PyIter_Next(py_iter))) {
        if (!PyTuple_Check(py_ic))
            throw std::logic_error(
                "Invalid training set ((input, cluster) tuples expected");

        PyObject * py_input   = PyTuple_GetItem(py_ic, 0);
        PyObject * py_cluster = PyTuple_GetItem(py_ic, 1);

        const lvq_t::input_t input = python2input(py_input);

        if (!PyLong_Check(py_cluster))
            throw std::logic_error("Invalid cluster (integer expected)");

        Py_ssize_t cluster = PyLong_AsSsize_t(py_cluster);

        if (0 > cluster)
            throw std::logic_error("Invalid cluster (must be >= 0)");

        set.emplace_back(input, cluster);

        Py_DECREF(py_ic);
    }

    Py_DECREF(py_iter);

    return set;
}


/**
 *  \brief  Transform Python training/test set to \c lvq_t::tset_clustering_t set
 *
 *  The Python training set is an iterable containing (input, cluster) tuples.
 *
 *  \param  py_set  Python iterable of number tuples
 *
 *  \return Training set
 */
static tset_clustering_t python2tset_clustering(PyObject * py_set) {
    Py_ssize_t set_size = PyObject_Size(py_set);
    if (-1 == set_size)
        throw std::logic_error("Invalid training set (can't get size)");

    PyObject * py_iter = PyObject_GetIter(py_set);
    if (NULL == py_iter)
        throw std::logic_error("Invalid training set (should be iterable)");

    tset_clustering_t set;

    PyObject * py_input;
    while (NULL != (py_input = PyIter_Next(py_iter))) {
        const lvq_t::input_t input = python2input(py_input);

        set.emplace_back(input);

        Py_DECREF(py_input);
    }

    Py_DECREF(py_iter);

    return set;
}


//
// Forward declarations
//

/** \cond */
static PyTypeObject * get_lvqType();
static PyTypeObject * get_lvqClassifierStatisticsType();
static PyTypeObject * get_lvqClusteringStatisticsType();
/** \endcond */


/**
 *  \brief  LVQ constructor
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
 *  \brief  LVQ destructor
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


/**
 *  \brief  LVQ classifier statistics constructor
 *
 *  \param  py_lvq_stats  Python LVQ classifier statistics object
 *  \param  args          Arguments
 *  \param  kwds          Keywords
 */
static void liblvq__lvq__classifier_statistics__create(
    lvqClassifierStatisticsObject_t * py_lvq_stats,
    PyObject                        * args,
    PyObject                        * kwds)
{
    // Get arguments
    size_t ccnt;
    parse_args(args, "n", &ccnt);

    // Create ml::lvq::classifier_statistics instance
    py_lvq_stats->lvq_stats = new lvq_classifier_stats_t(ccnt);
}


/**
 *  \brief  LVQ classifier statistics destructor
 *
 *  \param  py_lvq_stats  Python LVQ classifier statistics object
 *
 *  \return 0
 */
static int liblvq__lvq__classifier_statistics__destroy(
    lvqClassifierStatisticsObject_t * py_lvq_stats)
{
    lvq_classifier_stats_t * lvq_stats = py_lvq_stats->lvq_stats;
    py_lvq_stats->lvq_stats = NULL;

    if (NULL != lvq_stats) delete lvq_stats;

    return 0;
}

/** \cond */
static void BINDING_IDENT(liblvq__lvq__classifier_statistics__destroy)(
    lvqClassifierStatisticsObject_t * py_lvq_stats)
{
    wrap_X(0, liblvq__lvq__classifier_statistics__destroy, py_lvq_stats);
}
/** \endcond */


/**
 *  \brief  LVQ clustering statistics constructor
 *
 *  \param  py_lvq_stats  Python LVQ clustering statistics object
 *  \param  args          Arguments
 *  \param  kwds          Keywords
 */
static void liblvq__lvq__clustering_statistics__create(
    lvqClusteringStatisticsObject_t * py_lvq_stats,
    PyObject                        * args,
    PyObject                        * kwds)
{
    // Get arguments
    size_t ccnt;
    parse_args(args, "n", &ccnt);

    // Create ml::lvq::clustering_statistics instance
    py_lvq_stats->lvq_stats = new lvq_clustering_stats_t(ccnt);
}


/**
 *  \brief  LVQ clustering statistics destructor
 *
 *  \param  py_lvq_stats  Python LVQ clustering statistics object
 *
 *  \return 0
 */
static int liblvq__lvq__clustering_statistics__destroy(
    lvqClusteringStatisticsObject_t * py_lvq_stats)
{
    lvq_clustering_stats_t * lvq_stats = py_lvq_stats->lvq_stats;
    py_lvq_stats->lvq_stats = NULL;

    if (NULL != lvq_stats) delete lvq_stats;

    return 0;
}

/** \cond */
static void BINDING_IDENT(liblvq__lvq__clustering_statistics__destroy)(
    lvqClusteringStatisticsObject_t * py_lvq_stats)
{
    wrap_X(0, liblvq__lvq__clustering_statistics__destroy, py_lvq_stats);
}
/** \endcond */


//
// ml::lvq member functions binding
//

/** RNG seed */
static PyObject * rng_seed(PyObject * args) {
    int seed = 0;
    parse_args(args, "|i", &seed);
    srand(seed);

    // No return value
    Py_INCREF(Py_None);
    return Py_None;
}

/** \cond */
static PyObject * BINDING_IDENT(rng_seed)(PyObject * self, PyObject * args) {
    return wrap_X(self, rng_seed, args);
}
/** \endcond */


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
 *  \brief  ml::lvq::set_random binding
 */
static PyObject * liblvq__lvq__set_random(PyObject * self, PyObject * args) {
    // Get arguments
    size_t cluster = SIZE_MAX;
    parse_args(args, "|n", &cluster);

    // Call implementation
    if (SIZE_MAX == cluster)
        python2lvq(self)->set_random();
    else
        python2lvq(self)->set_random(cluster);

    // No return value
    Py_INCREF(Py_None);
    return Py_None;
}

BINDING_INST(liblvq__lvq__set_random)


/**
 *  \brief  \c ml::lvq::train1_supervised binding
 */
static PyObject * liblvq__lvq__train1_supervised(PyObject * self, PyObject * args) {
    // Get arguments
    PyObject *    py_input;
    size_t        cluster;
    lvq_t::base_t lfactor;
    parse_args(args, "Ond", &py_input, &cluster, &lfactor);

    const lvq_t::input_t input = python2input(py_input);

    // Call implementation
    lvq_t::base_t dnorm2 = python2lvq(self)->train1_supervised(input, cluster, lfactor);

    // Transform result
    return Py_BuildValue("d", dnorm2);
}

BINDING_INST(liblvq__lvq__train1_supervised)


/**
 *  \brief  \c ml::lvq::train1_unsupervised binding
 */
static PyObject * liblvq__lvq__train1_unsupervised(PyObject * self, PyObject * args) {
    // Get arguments
    PyObject *    py_input;
    lvq_t::base_t lfactor;
    parse_args(args, "Od", &py_input, &lfactor);

    const lvq_t::input_t input = python2input(py_input);

    // Call implementation
    lvq_t::base_t dnorm2 = python2lvq(self)->train1_unsupervised(input, lfactor);

    // Transform result
    return Py_BuildValue("d", dnorm2);
}

BINDING_INST(liblvq__lvq__train1_unsupervised)


/**
 *  \brief  \c ml::lvq::train_supervised binding
 */
static PyObject * liblvq__lvq__train_supervised(PyObject * self, PyObject * args) {
    // Get arguments
    PyObject * py_set;
    unsigned   conv_win    = LIBLVQ__ML__LVQ__TRAIN__CONV_WIN;
    unsigned   max_div_cnt = LIBLVQ__ML__LVQ__TRAIN__MAX_DIV_CNT;
    unsigned   max_tlc     = LIBLVQ__ML__LVQ__TRAIN__MAX_TLC;
    parse_args(args, "O|III", &py_set, &conv_win, &max_div_cnt, &max_tlc);

    const tset_classifier_t set = python2tset_classifier(py_set);

    // Call implementation
    python2lvq(self)->train_supervised(set, conv_win, max_div_cnt, max_tlc);

    Py_INCREF(Py_None);
    return Py_None;
}

BINDING_INST(liblvq__lvq__train_supervised)


/**
 *  \brief  \c ml::lvq::train_unsupervised binding
 */
static PyObject * liblvq__lvq__train_unsupervised(PyObject * self, PyObject * args) {
    // Get arguments
    PyObject * py_set;
    unsigned   conv_win    = LIBLVQ__ML__LVQ__TRAIN__CONV_WIN;
    unsigned   max_div_cnt = LIBLVQ__ML__LVQ__TRAIN__MAX_DIV_CNT;
    unsigned   max_tlc     = LIBLVQ__ML__LVQ__TRAIN__MAX_TLC;
    parse_args(args, "O|III", &py_set, &conv_win, &max_div_cnt, &max_tlc);

    const tset_clustering_t set = python2tset_clustering(py_set);

    // Call implementation
    python2lvq(self)->train_unsupervised(set, conv_win, max_div_cnt, max_tlc);

    Py_INCREF(Py_None);
    return Py_None;
}

BINDING_INST(liblvq__lvq__train_unsupervised)


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
 *  \brief  \c ml::lvq::test_classifier binding
 */
static PyObject * liblvq__lvq__test_classifier(PyObject * self, PyObject * args) {
    PyTypeObject * lvq_stats_type = get_lvqClassifierStatisticsType();

    lvqClassifierStatisticsObject_t * py_lvq_stats =
        reinterpret_cast<lvqClassifierStatisticsObject_t *>(
            lvq_stats_type->tp_alloc(lvq_stats_type, 0));

    if (NULL == py_lvq_stats) return NULL;

    // Get arguments
    PyObject * py_set;
    parse_args(args, "O", &py_set);

    const tset_classifier_t set = python2tset_classifier(py_set);

    // Call implementation
    py_lvq_stats->lvq_stats = new lvq_classifier_stats_t(
        python2lvq(self)->test_classifier(set));

    // Transform result
    return reinterpret_cast<PyObject *>(py_lvq_stats);
}

BINDING_INST(liblvq__lvq__test_classifier)


/**
 *  \brief  \c ml::lvq::test_clustering binding
 */
static PyObject * liblvq__lvq__test_clustering(PyObject * self, PyObject * args) {
    PyTypeObject * lvq_stats_type = get_lvqClusteringStatisticsType();

    lvqClusteringStatisticsObject_t * py_lvq_stats =
        reinterpret_cast<lvqClusteringStatisticsObject_t *>(
            lvq_stats_type->tp_alloc(lvq_stats_type, 0));

    if (NULL == py_lvq_stats) return NULL;

    // Get arguments
    PyObject * py_set;
    parse_args(args, "O", &py_set);

    const tset_clustering_t set = python2tset_clustering(py_set);

    // Call implementation
    py_lvq_stats->lvq_stats = new lvq_clustering_stats_t(
        python2lvq(self)->test_clustering(set));

    // Transform result
    return reinterpret_cast<PyObject *>(py_lvq_stats);
}

BINDING_INST(liblvq__lvq__test_clustering)


/**
 *  \brief  \c ml::lvq::store binding
 */
static PyObject * liblvq__lvq__store(PyObject * self, PyObject * args) {
    // Get arguments
    const char * file;
    parse_args(args, "s", &file);

    // Call implementation
    python2lvq(self)->store(file);

    Py_INCREF(Py_None);
    return Py_None;
}

BINDING_INST(liblvq__lvq__store)


/**
 *  \brief  \c ml::lvq::load binding
 */
static PyObject * liblvq__lvq__load(PyObject * type, PyObject * args) {
    lvqObject_t * py_lvq = reinterpret_cast<lvqObject_t *>(
        ((PyTypeObject *)type)->tp_alloc((PyTypeObject *)type, 0));

    if (NULL == py_lvq) return NULL;

    // Get arguments
    const char * file;
    parse_args(args, "s", &file);

    // Create dummy ml::lvq instance
    py_lvq->lvq = new lvq_t(0, 0);

    // Call implementation
    *py_lvq->lvq = lvq_t::load(file);

    return reinterpret_cast<PyObject *>(py_lvq);
}

BINDING_INST(liblvq__lvq__load)


//
// ml::lvq::classifier_statistics member functions binding
//

/**
 *  \brief  Constructor
 *
 *  \param  type  Python LVQ classifier statistics type
 *  \param  args  Arguments
 *  \param  kwds  Keywords
 *
 *  \return LVQ instance
 */
static PyObject * liblvq__lvq__classifier_statistics__new(
    PyTypeObject * type,
    PyObject     * args,
    PyObject     * kwds)
{
    lvqClassifierStatisticsObject_t * py_lvq_stats =
        reinterpret_cast<lvqClassifierStatisticsObject_t *>(type->tp_alloc(type, 0));

    if (NULL == py_lvq_stats) return NULL;

    liblvq__lvq__classifier_statistics__create(py_lvq_stats, args, kwds);

    return reinterpret_cast<PyObject *>(py_lvq_stats);
}

/** \cond */
static PyObject * BINDING_IDENT(liblvq__lvq__classifier_statistics__new)(
    PyTypeObject * type,
    PyObject     * args,
    PyObject     * kwds)
{
    return wrap_X((PyObject *)NULL, liblvq__lvq__classifier_statistics__new, type, args, kwds);
}
/** \endcond */


/**
 *  \brief  Constructor (__init__)
 *
 *  \param  self  Python LQV classifier statistics object
 *  \param  args  Arguments
 *  \param  kwds  Keywords
 *
 *  \return 0
 */
static int liblvq__lvq__classifier_statistics__init(
    PyObject * self,
    PyObject * args,
    PyObject * kwds)
{
    lvqClassifierStatisticsObject_t * py_lvq_stats =
        reinterpret_cast<lvqClassifierStatisticsObject_t *>(self);

    liblvq__lvq__classifier_statistics__destroy(py_lvq_stats);

    liblvq__lvq__classifier_statistics__create(py_lvq_stats, args, kwds);

    return 0;
}

/** \cond */
static int BINDING_IDENT(liblvq__lvq__classifier_statistics__init)(
    PyObject * self,
    PyObject * args,
    PyObject * kwds)
{
    return wrap_X((int)1, liblvq__lvq__classifier_statistics__init, self, args, kwds);
}
/** \endcond */


/**
 *  \brief  ml::lvq::classifier_statistics::accuracy binding
 */
static PyObject * liblvq__lvq__classifier_statistics__accuracy(
    PyObject * self, PyObject * args)
{
    // Call implementation
    double accuracy = python2lvq_classifier_stats(self)->accuracy();

    // Transform result
    return Py_BuildValue("d", accuracy);
}

BINDING_INST(liblvq__lvq__classifier_statistics__accuracy)


/**
 *  \brief  ml::lvq::classifier_statistics::precision binding
 */
static PyObject * liblvq__lvq__classifier_statistics__precision(
    PyObject * self, PyObject * args)
{
    // Get arguments
    int c1ass;
    parse_args(args, "I", &c1ass);

    // Call implementation
    double precision = python2lvq_classifier_stats(self)->precision(c1ass);

    // Transform result
    return Py_BuildValue("d", precision);
}

BINDING_INST(liblvq__lvq__classifier_statistics__precision)


/**
 *  \brief  ml::lvq::classifier_statistics::recall binding
 */
static PyObject * liblvq__lvq__classifier_statistics__recall(
    PyObject * self, PyObject * args)
{
    // Get arguments
    int c1ass;
    parse_args(args, "I", &c1ass);

    // Call implementation
    double recall = python2lvq_classifier_stats(self)->recall(c1ass);

    // Transform result
    return Py_BuildValue("d", recall);
}

BINDING_INST(liblvq__lvq__classifier_statistics__recall)


/**
 *  \brief  ml::lvq::classifier_statistics::F(beta) binding
 */
static PyObject * liblvq__lvq__classifier_statistics__F_beta(
    PyObject * self, PyObject * args)
{
    // Get arguments
    double beta;
    int    c1ass = -1;
    parse_args(args, "d|I", &beta, &c1ass);

    // Call implementation
    double F_beta = c1ass < 0
        ? python2lvq_classifier_stats(self)->F(beta)
        : python2lvq_classifier_stats(self)->F(beta, (size_t)c1ass);

    // Transform result
    return Py_BuildValue("d", F_beta);
}

BINDING_INST(liblvq__lvq__classifier_statistics__F_beta)


/**
 *  \brief  ml::lvq::classifier_statistics::F binding
 */
static PyObject * liblvq__lvq__classifier_statistics__F(
    PyObject * self, PyObject * args)
{
    // Get arguments
    int c1ass = -1;
    parse_args(args, "|I", &c1ass);

    // Call implementation
    double F = c1ass < 0
        ? python2lvq_classifier_stats(self)->F()
        : python2lvq_classifier_stats(self)->F((size_t)c1ass);

    // Transform result
    return Py_BuildValue("d", F);
}

BINDING_INST(liblvq__lvq__classifier_statistics__F)


//
// ml::lvq::clustering_statistics member functions binding
//

/**
 *  \brief  Constructor
 *
 *  \param  type  Python LVQ clustering statistics type
 *  \param  args  Arguments
 *  \param  kwds  Keywords
 *
 *  \return LVQ instance
 */
static PyObject * liblvq__lvq__clustering_statistics__new(
    PyTypeObject * type,
    PyObject     * args,
    PyObject     * kwds)
{
    lvqClusteringStatisticsObject_t * py_lvq_stats =
        reinterpret_cast<lvqClusteringStatisticsObject_t *>(type->tp_alloc(type, 0));

    if (NULL == py_lvq_stats) return NULL;

    liblvq__lvq__clustering_statistics__create(py_lvq_stats, args, kwds);

    return reinterpret_cast<PyObject *>(py_lvq_stats);
}

/** \cond */
static PyObject * BINDING_IDENT(liblvq__lvq__clustering_statistics__new)(
    PyTypeObject * type,
    PyObject     * args,
    PyObject     * kwds)
{
    return wrap_X((PyObject *)NULL, liblvq__lvq__clustering_statistics__new, type, args, kwds);
}
/** \endcond */


/**
 *  \brief  Constructor (__init__)
 *
 *  \param  self  Python LQV clustering statistics object
 *  \param  args  Arguments
 *  \param  kwds  Keywords
 *
 *  \return 0
 */
static int liblvq__lvq__clustering_statistics__init(
    PyObject * self,
    PyObject * args,
    PyObject * kwds)
{
    lvqClusteringStatisticsObject_t * py_lvq_stats =
        reinterpret_cast<lvqClusteringStatisticsObject_t *>(self);

    liblvq__lvq__clustering_statistics__destroy(py_lvq_stats);

    liblvq__lvq__clustering_statistics__create(py_lvq_stats, args, kwds);

    return 0;
}

/** \cond */
static int BINDING_IDENT(liblvq__lvq__clustering_statistics__init)(
    PyObject * self,
    PyObject * args,
    PyObject * kwds)
{
    return wrap_X((int)1, liblvq__lvq__clustering_statistics__init, self, args, kwds);
}
/** \endcond */


/**
 *  \brief  ml::lvq::clustering_statistics::avg_error binding
 */
static PyObject * liblvq__lvq__clustering_statistics__avg_error(
    PyObject * self, PyObject * args)
{
    // Get arguments
    int    c1ass = -1;
    parse_args(args, "|I", &c1ass);

    // Call implementation
    double avg_error = c1ass < 0
        ? python2lvq_clustering_stats(self)->avg_error()
        : python2lvq_clustering_stats(self)->avg_error(c1ass);

    // Transform result
    return Py_BuildValue("d", avg_error);
}

BINDING_INST(liblvq__lvq__clustering_statistics__avg_error)


//
// Module state
//

struct module_state {
    PyObject * error;
};


static int liblvq_traverse(PyObject * m, visitproc visit, void * arg) {
    Py_VISIT(((struct module_state *)PyModule_GetState(m))->error);
    return 0;
}


static int liblvq_clear(PyObject * m) {
    Py_CLEAR(((struct module_state *)PyModule_GetState(m))->error);
    return 0;
}


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
        "set_random",
        BINDING_IDENT(liblvq__lvq__set_random),
        METH_VARARGS,
        "Set cluster representant(s) randomly"
    },
    {
        "train1_supervised",
        BINDING_IDENT(liblvq__lvq__train1_supervised),
        METH_VARARGS,
        "Supervised training step"
    },
    {
        "train1_unsupervised",
        BINDING_IDENT(liblvq__lvq__train1_unsupervised),
        METH_VARARGS,
        "Unsupervised training step"
    },
    {
        "train_supervised",
        BINDING_IDENT(liblvq__lvq__train_supervised),
        METH_VARARGS,
        "Train LVQ model (supervised training)"
    },
    {
        "train_unsupervised",
        BINDING_IDENT(liblvq__lvq__train_unsupervised),
        METH_VARARGS,
        "Train LVQ model (unsupervised training)"
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
        "test_classifier",
        BINDING_IDENT(liblvq__lvq__test_classifier),
        METH_VARARGS,
        "Test LVQ classifier"
    },
    {
        "test_clustering",
        BINDING_IDENT(liblvq__lvq__test_clustering),
        METH_VARARGS,
        "Test LVQ clusteringmodel"
    },
    {
        "store",
        BINDING_IDENT(liblvq__lvq__store),
        METH_VARARGS,
        "Store lvq instance to a file"
    },
    {
        "load",
        BINDING_IDENT(liblvq__lvq__load),
        METH_VARARGS | METH_CLASS,
        "Load lvq instance from a file"
    },

    { NULL, NULL, 0, NULL }  // sentinel
};  // end of lvqObject_methods


/** LVQ classifier statistics data members */
static PyMemberDef lvqClassifierStatisticsObject_attrs[] = {
    { NULL, 0, 0, 0, NULL }  // sentinel
};  // end of lvqClassifierStatisticsObject_members


/** LVQ classifier statistics member functions */
static PyMethodDef lvqClassifierStatisticsObject_methods[] = {
    {
        "accuracy",
        BINDING_IDENT(liblvq__lvq__classifier_statistics__accuracy),
        METH_VARARGS,
        "Get accuracy"
    },
    {
        "precision",
        BINDING_IDENT(liblvq__lvq__classifier_statistics__precision),
        METH_VARARGS,
        "Get precision for class"
    },
    {
        "recall",
        BINDING_IDENT(liblvq__lvq__classifier_statistics__recall),
        METH_VARARGS,
        "Get recall for class"
    },
    {
        "F_beta",
        BINDING_IDENT(liblvq__lvq__classifier_statistics__F_beta),
        METH_VARARGS,
        "Get F_beta score"
    },
    {
        "F",
        BINDING_IDENT(liblvq__lvq__classifier_statistics__F),
        METH_VARARGS,
        "Get F (i.e. F_1) score"
    },

    { NULL, NULL, 0, NULL }  // sentinel
};  // end of lvqClassifierStatisticsObject_methods


/** LVQ clustering statistics data members */
static PyMemberDef lvqClusteringStatisticsObject_attrs[] = {
    { NULL, 0, 0, 0, NULL }  // sentinel
};  // end of lvqClusteringStatisticsObject_members


/** LVQ clustering statistics member functions */
static PyMethodDef lvqClusteringStatisticsObject_methods[] = {
    {
        "avg_error",
        BINDING_IDENT(liblvq__lvq__clustering_statistics__avg_error),
        METH_VARARGS,
        "Get average error"
    },

    { NULL, NULL, 0, NULL }  // sentinel
};  // end of lvqClusteringStatisticsObject_methods


/** LVQ Python type */
static PyTypeObject lvqType = {
    PyObject_HEAD_INIT(NULL)

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

static PyTypeObject * get_lvqType() { return &lvqType; }


/** LVQ classifier statistics Python type */
static PyTypeObject lvqClassifierStatisticsType = {
    PyObject_HEAD_INIT(NULL)

    /* tp_name          */  "liblvq.lvq.classifier_statistics",
    /* tp_basicsize     */  sizeof(lvqClassifierStatisticsObject_t),
    /* tp_itemsize      */  0,
    /* tp_dealloc       */  (destructor)BINDING_IDENT(liblvq__lvq__classifier_statistics__destroy),
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
    /* tp_doc           */  "lvq classifier statistics objects",
    /* tp_traverse      */  0,
    /* tp_clear         */  0,
    /* tp_richcompare   */  0,
    /* tp_weaklistoffset*/  0,
    /* tp_iter          */  0,
    /* tp_iternext      */  0,
    /* tp_methods       */  lvqClassifierStatisticsObject_methods,
    /* tp_members       */  lvqClassifierStatisticsObject_attrs,
    /* tp_getset        */  0,
    /* tp_base          */  0,
    /* tp_dict          */  0,
    /* tp_descr_get     */  0,
    /* tp_descr_set     */  0,
    /* tp_dictoffset    */  0,
    /* tp_init          */  BINDING_IDENT(liblvq__lvq__classifier_statistics__init),
    /* tp_alloc         */  0,
    /* tp_new           */  BINDING_IDENT(liblvq__lvq__classifier_statistics__new),

};  // end of lvqClassifierStatisticsType

static PyTypeObject * get_lvqClassifierStatisticsType() {
    return &lvqClassifierStatisticsType;
}


/** LVQ clustering statistics Python type */
static PyTypeObject lvqClusteringStatisticsType = {
    PyObject_HEAD_INIT(NULL)

    /* tp_name          */  "liblvq.lvq.clustering_statistics",
    /* tp_basicsize     */  sizeof(lvqClusteringStatisticsObject_t),
    /* tp_itemsize      */  0,
    /* tp_dealloc       */  (destructor)BINDING_IDENT(liblvq__lvq__clustering_statistics__destroy),
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
    /* tp_doc           */  "lvq clustering statistics objects",
    /* tp_traverse      */  0,
    /* tp_clear         */  0,
    /* tp_richcompare   */  0,
    /* tp_weaklistoffset*/  0,
    /* tp_iter          */  0,
    /* tp_iternext      */  0,
    /* tp_methods       */  lvqClusteringStatisticsObject_methods,
    /* tp_members       */  lvqClusteringStatisticsObject_attrs,
    /* tp_getset        */  0,
    /* tp_base          */  0,
    /* tp_dict          */  0,
    /* tp_descr_get     */  0,
    /* tp_descr_set     */  0,
    /* tp_dictoffset    */  0,
    /* tp_init          */  BINDING_IDENT(liblvq__lvq__clustering_statistics__init),
    /* tp_alloc         */  0,
    /* tp_new           */  BINDING_IDENT(liblvq__lvq__clustering_statistics__new),

};  // end of lvqClusteringStatisticsType

static PyTypeObject * get_lvqClusteringStatisticsType() {
    return &lvqClusteringStatisticsType;
}


/** Module member functions */
static PyMethodDef liblvq_methods[] = {
    {
        "rng_seed",
        BINDING_IDENT(rng_seed),
        METH_VARARGS,
        "Seed RNG"
    },

    { NULL, NULL, 0, NULL }  // sentinel
};  // end of liblvq__methods


/** Module definition */
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "liblvq",
    NULL,
    sizeof(struct module_state),
    liblvq_methods,
    NULL,
    liblvq_traverse,
    liblvq_clear,
    NULL
};


/** Module initialiser */
PyMODINIT_FUNC PyInit_liblvq(void) {
    if (PyType_Ready(&lvqType)                     < 0) return NULL;
    if (PyType_Ready(&lvqClassifierStatisticsType) < 0) return NULL;
    if (PyType_Ready(&lvqClusteringStatisticsType) < 0) return NULL;

    PyObject * module = PyModule_Create(&moduledef);
    if (NULL == module) return NULL;

    Py_INCREF(&lvqType);
    PyModule_AddObject(module, "lvq", (PyObject *)&lvqType);

    Py_INCREF(&lvqClassifierStatisticsType);
    PyModule_AddObject(module, "lvq.classifier_statistics",
        (PyObject *)&lvqClassifierStatisticsType);

    Py_INCREF(&lvqClusteringStatisticsType);
    PyModule_AddObject(module, "lvq.clustering_statistics",
        (PyObject *)&lvqClusteringStatisticsType);

    return module;
}
