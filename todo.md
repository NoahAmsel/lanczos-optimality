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

