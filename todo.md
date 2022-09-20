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