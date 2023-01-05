work on the case where there are real roots in the denominator, and numerator is apolynomial with degree < number of iterations

try redoing $1/\lambda^2$ in a different norm... hasn't worked so far
look up Anne Greenbaum / Leonid proof of Lanczos for exp(A)x

also look into $\frac1{A^2 + cI}$

functions $\lambda^{\alpha}$ for $\alpha \in (-1,1)$ or $\log(\lambda)$ or $\log(1+\lambda)$

Instead of computing minimax approximation to $R$, we could just compute the Chebyshev interpolation. The wikipedia page for Lebesgue constant shows that this is close to the optimal polynomial interpolant. 

Actually if we want a polynomial of degree $k$, just uniformly sample a million points (actually need at least $k^2$) in the range and do regression on the Chebyshev basis of size $k$.
(or instead of uniformly sampling them, Chebyshev sample them)
But this will not really be the same as the l-infinity problem

ALSO do interpolant of $k+1$ Chebyshev points 

Ask Anne to order textbooks ... Kari Schwartz

clean up the lanczos function

either
- can do "regular" Gram Schmdit twice in the inner loop of lanczos. instead of modified gram schmdit, just copy the "unmodified" gram schmdit line twice lol
- use the idea of modified gram schmidt inside the lanczos function

October 3
Try doing this again for higher precision so things don't just flattern at 10^-16
MPmath for arbitrary precision
- make sure NOT to use their sparse matrix class. just represent a diagonal matrix as a vector

Transfer the proof from hackmd to overleaf, for $1/[(x-z_1)(x-z_2)\cdots(x-z_q)]$
assuming the $z$'s aren't in the range of eigenvectors

Try to tighten $1/x^2$ using some other norm. 

Instead of Tyler's conditions for sqare root on $[1, k]$, maybe try the regime $[0, k]$ since in this case polynomials converge algebraically, not even exponentially.

Oct 17
For explicit form of rational approx to sqaure root, use Tyler's implementation of this logic
http://guettel.com/rktoolbox/examples/html/example_zolotarev.html#8
which is based on matrix sign
he sent his in slack
also a sidford paper that has an explicit form (search "Zolaterov")
is there even a paper analyzing the rate of convergence

Nov 8
Try gradient descent to find hard spectra or hard b's for a given spectrum
Notice that we pulled out a matrix of norm 1. But this matrix might actually help bring down the norm of what's inside there
Tyler notes:
Michael Overton and Anne maybe used a similar idea of optimization to try to find a worst case example for some NLA problem
"Local minimizers of the Crouzeix ratio: a nonsmooth optimization case study"
"GMRES vs Ideal GMRES"

Nov 21
$X^{\alpha}$ has nice ML applications. 
- [illegible] sign function do $\sqrt{A^2}$
If eigenvectors of $QQ^TAQQ^T$ nearly converged to those of $A$

Go back to the paper Anne sent to try to construct the hard instance
- Cauchy interlacing theorem for tridiagonal matric

in exact arithmetic, run Lanczos to the end... then the tridiagonal matrix is $QQ^TAQQ^T$ with $Q \in \R^{n \times n}$. This has the same eigenvalues as $A$
- now just think about the submatrices of $QQ^TAQQ^T$, which is $Q_k Q_k^T A Q_k Q_k^T$

using this paper and Cauchy interlacing, make the Ritz values equally spaced between the eigenvalues of $A$

could also try to look at eigenvalues of $Q_k Q_k^T A Q_k Q_k^T$ as $k$ increases and see how they evolve

write up full proof for poly(A) / poly(A) or poly(A) / $(A+z_1I)(A+z_2I)$

Lanczos error at step $m$ is $\leq C \cdot \mathrm{opt}_{m - q_k}$ where $m > k$ and $q_k$ is degree of denominator
assume poles are real and outside the range of eigenvalues for now

For the future:
- Anne's work states: if you have a Ritz value near the pole of the rational function, you'll get terrible error
- but in the next iteration, it will do well because guaranteed not to be close to the pole

Yujia Jin, Sidford, NeurIPS 2019
"PC Projection and regression in nearly linear time".
ML cares b/c principle component regression. Also for sampling from high dimensional Gaussian with covariance $C$, need $\sqrt{C}$
often what we actually know is $C^{-1}$. In that case we need $C^{-1/2}$ - e.g. for Gaussian graphical models

Dec 6
next, tackle product
$$(A - \lambda_1I)^{-1}(A - \lambda_2I)^{-1}\cdot (A - \lambda_qI)^{-1}$$

MUSCO Dec 13
at the n-1th step, make the ritz values be right in between each pair of eigenvalues
in that case, QT^{-1}Q^T should commute poorly with A^{-1}
then just check experimentally if there's a lot of error in earlier iterations (earlier than n-1)
it might be unstable / hard to implement :(

what matters for A^{-1}QT^{-1}Q^Tb ... depends on the small eigenvalues of A

experiment:
create a small animation where we run Lanczos and see how the Ritz values evolve
Hypothesis is that they converge to outer eigenvalues first (biggest and largest first)
see theorem 3.2
and the example right after it which shows how 
you can make Lambda whatever you want. so which one should we do?
Well if you make the Lambda's to be the Chebyshev nodes, then the poly that interpolates at the Lambdas should also be a good uniform approximation on the range...
try making Lambda: uniform, or two widely separated clusters

an important example:
sign function / step function where your eigenvalues are in [-1, 1]
and eigenvalues are very well behaved, clustered around -1 and 1
then Lanczos may do much worse than the best approximating polynomial
because you'll sometimes get ritz values very close to 0, where the approximating polynomial is bad
because recall: the polynomial that Lanczos applies is exactly the interpolating polynomial at the ritz values. and the polynomial that interpolates ritz values will be much worse than that which interpolates the real values because the ritz values include points near zero, where the steep step is. if all your interpolating points are near -1 and 1 then you can approximate step function with a low degree polynomial


another example spectrum that could be hard for A^{-2}b
a few eigenvalues that are big, say in the range 1,2
one cluster that's *very* tightly concentrated around 0.1


FOR THE FUNCTION x^{-2}
it seems like the hard case is
it seems like if you have a range of values where the eigenvalues are even
and then one eigenvalue at a point much SMALLER (closer to zero) than those
because the function value at the small point is very LARGE (it's one over (something small)^2)
whereas if we had made that new point much bigger than the uniform eigenvalues,
it wouldn't be that hard to nail it by just adding on a Lagrange basis polynomial over the set {chebyshev points over the range of dense eigenvalues} UNION {the outlying point}
and since the coefficient of the lagrange basis polynomial would be f(new point) which is very small, adding that new polynomial won't screw up your approximation at within the range of the dense eigenvalues

see picture on my phone from today

## Jan 4 2023
after refactoring convergence.ipynb, reran some tests on the function $x^{-2}$ with condition number 1000 and 100 dimensions, and $b = 1$
- flipped model spectrum: Lanczos does extremely close to optimal, but the uniform bound sucks. even with the $\kappa$ factor, our bound beats the old one after 10 iterations. except at the very end, around when the krylov optimal flatlines, the lanczos method flatlines a bit higher
- one cluster of 3 points around 1, the other 97 points around 1000. same story as reverse model spectrum but even more extreme. new bound is winning after 5 iters.
    - when you now pick b so that mu is evenly spaced between eigenvalues, convergence is superfast. machine precision in just a few iters. of course the uniform bound is extremely terrible, same as before
- uniform distribution: lanczos still performs optimally, but much gentler slope compared to the uniform bound. our bound wins at about 55 iters.
    - what are the ritz values in the n-1 th iteration for b is all ones? those at the beginning and end of the range are very accurate, those in the middle are not
- model spectrum: similar to uniform distribution but even more extreme. especially in the lower iterations, there's almost no gap between everything
- 95 eigenvalues clustered near 1, and a few evenly spaced between 500 and 1000: again, lanczos is basically optimal. a straight line down, our bound beats old one at 20


### idea
why are we using $b$, the vector we want to apply, as the vector to build the Krylov subspace
maybe we could use b for a few iterations to get an initial estimate and then use that estimate to pick a different vector to build a new krylov subspace with
