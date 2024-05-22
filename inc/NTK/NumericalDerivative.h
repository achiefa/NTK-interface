#include <fstream>
#include <functional>
#include <algorithm>
#include <iostream>

/**
 * @brief First order derivative with finite difference
 *
 * Implement the numerical differentiation for the first order
 * derivative. In particular, given a function f(x), the
 * derivative is computed as
 *
 *      f'(x) = (f(x + h) - f(x - h)) / 2h.
 *
 * The output is a two-dimensional std::vector where the first
 * index runs over the parameters respect which the derivative
 * is computed. The second index runs over the possible entries
 * of f, which can be a n-dimensional vector.
 *
 * For instance, this function is used to compute numerically
 * the second derivative of the NNAD neural network starting
 * from the exact first order derivative. In this case, the
 * method `derive` returns a std::vector where the first
 * entries are the function evaluations for a given data point
 * x, whereas the other entries are the derivatives of each
 * output nodes of the NN (in case output were n-dimensional).
 * This method will then return a matrix where the first index
 * selects the parameter respect which the finite difference is
 * computed, whereas the internal index runs over the vector
 * returned by `derive`. In that particular case, `FiniteDifference`
 * will also compute the first numerical derivatives.
 *
 * @param f The std::function the we want to compute the derivative of. It must
 *          a function of the input vector `x` and the parameter vector.
 *          It returns a std::vector whose size is dim(f).
 * @param parameters Vector of parameters w.r.t we want to differentiate. This
 *                   will be passed in the std::function f.
 * @param x The data point at which we want to evaluate the derivative.
 * @param eps The 'small' increment in the finite difference.
 * @return vector<vector>> where the first index runs over the parameters and
 *         the second one runs over the dimension of f.
 * @todo The sta::function f has the input and the parameters as argument. This
 *       implicitly assumes that we are dealing with a parametrised function.
 *       However, this function can in principle be applied to plain function,
 *       namely those that only depends on the point x and whose parameters
 *       are fixed. It would be better if the signature of f were to contain
 *       only one single argument.
*/
namespace NTK
{
  std::vector<std::vector<double>> FiniteDifference (
    std::function<std::vector<double> (std::vector<double> const&, std::vector<double>)> f,
    std::vector<double> parameters, 
    std::vector<double> const& x, 
    double const& eps);

  /**
   * @brief First order derivative with finite difference - single vector
   *
   * This is the same function as `FiniteDifference`, but returns the derivatives
   * as a single vector object. The ordering of the vector is as follows:
   *
   *    results_jk = results[ j + k * dim(f) ]   Col-Major
   *
   * where
   *
   *      j = 0, ..., dim(f) - 1 and k = 0, ..., dim(np) - 1.
   *
   * It may happen that the function "f" is already a vectorial representation
   * of some tensor. For instance, f can represent the vector of first order
   * derivatives of a NN (and evaluations from the back-propagation). In this
   * case, f is a vector that represents a two-rank tensor, where the indices
   * run over the outputs and the parameters (remember that in the case of NNAD,
   * the method `derive` also provides the evaluation of the NN). Then, the
   * the actual dimension of f is the products of the two sub-dimensions that
   * build up f, namely dim(f) = dim(f_1) * dim(f_2). Moreover, also the index
   * linked to f can actually be splitted into two separates indices that run
   * over the two dimensions of f respectively. Assuming also f is in col-major
   * order, we have j -> i + j * dim(f_1), where
   *            i = 0, ..., dim(f_1) - 1  and  j = 0, ..., dim(f_2) - 1,
   * and the 3-rank tensor can then be written as
   *
   *     results_ijk = results[ i + j * dim(f_1) + k * dim(f) * dim(f_2)]   Col-Major
  */
  std::vector<double> FiniteDifferenceVec (
    std::function<std::vector<double> (std::vector<double> const&, std::vector<double>)> f,
    std::vector<double> parameters,
    std::vector<double> const& x,
    int const& size_f,
    double const& eps);
}