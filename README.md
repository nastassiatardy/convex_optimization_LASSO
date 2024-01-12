# convex_optimization_LASSO

Given $x_1, ..., x_n \in \mathbb{R}^d$ data vectors and $y_1, ..., y_n \in \mathbb{R}$ observations, we are searching for regression parameters $w \in \mathbb{R}^d$ which fit data inputs to observations $y$ by minimizing their squared difference. In a high dimensional setting (when $n << l$) a $l_1$ norm penalty is often used on the regression coefficients $w$ in order to enforce sparsity of the solution (so that $w$ will only have a few non-zeros entries). Such penalizaion has well known statistical properties, and makers the model both more interpretable and faster at test time.

From an optimization point of view we want to solve the following problem called LASSO (Least Absolute Shrinkage Operator and Selection Operator):
$`
\text{minimize} \frac{1}{2} ||Xw - y||_2^2 + \lambda ||w||_1
`$


in the variable $w \in \mathbb{R}^d$, where $X = (x_1^T, ..., x_n^T) \in \mathbb{R}^{n\text{x}d}$, $y=(y_1, ..., y_n) \in \mathbb{R}^n$ and $\lambda > 0$ is a regularization parameter.  

# Formalize the LASSO problem as a general Quadratic Problem

We note $z=Xw-y \in \mathbb{R}^n$. Then (LASSO) is equivalent to 
$`
\text{min}_{w\in \mathbb{R}^d, z\in \mathbb{R}^n, z=Xw-y} \frac{1}{2}||z||_2^2 + \lambda ||w||_1
`$

The corresponding Lagrangian writes, for $\nu \ in \ \mathbb{R}^n$:
$`
L(w, z, \nu) = \frac{1}{2} ||z||_2^2 + \lambda ||w||_1 + \nu^T(Xw - y -z)
`$

Therefore, 
$`
\begin{aligned} g(\nu) &= \text{inf}_{w\in\mathbb{R}^d, z\in\mathbb{R}^n} \left[  \frac{1}{2} |||z||_2^2 + \lambda ||w||_1 + \nu^T(Xw-y-z)\right] \\
                       &= \text{inf}_z \left[ \frac{1}{2} ||z||_2^2 - \nu^Tz\right] + \text{inf}_w \left[ \lambda ||w||_1 + \nu^TXw \right]- \nu^Ty
\end{aligned}
`$

We note $T_1(\nu) = \text{inf}_z \left[\frac{1}{2}||z||_2^2 - \nu^Tz\right]$ and $T_2(\nu) = \text{inf}_w \left[ \lambda ||w||_1 + \nu^TXw \right]$ and we solve the two problems separately.

## Solving $T_{1}$

$\nabla_z \left(\frac{1}{2} ||z||_2^2 - \nu^Tz \right) = \nabla_z \left( \frac{1}{2} z^Tz - \nu^Tz \right) = z - \nu$
therefore, $\nabla_z \left(\frac{1}{2} ||z||_2^2 - \nu^Tz \right) = 0$ if and only if $z=\nu$.

Injecting this result in $T_1$ we get:
$` \begin{aligned}
\text{inf}_z \left( \frac{1}{2} ||z||_2^2 - \nu^Tz \right) &= \frac{1}{2}||\nu||_2^2 - \nu^T\nu \\
                                                           &= -\frac{1}{2} ||\nu||_2^2
\end{aligned}
`$


## Solving $T_{2}$

To solve $T_2$ we use the following result: \
$` \begin{aligned}
|| x||_1^* &= \text{sup}_y (x^T y - ||y||^1) = -\text{inf}_y \left( ||y||_1 - x^Ty \right) \\

            &= \begin{cases}
                        + \infty \ \text{if} \ ||x||_\infty > 1\\
                        0 \ \text{if} \ ||x||_\infty \leq 1
                \end{cases}
\end{aligned}
`$

Therefore, we have 
$` \begin{aligned}
T_2(\nu) &= \text{inf}_w \left(\lambda ||w||_1 + \nu^TXw \right)
         &= \lambda \text{inf}_w \left( ||w||_1 - \left( -\frac{\nu^T X}{\lambda} \right) w \right) 
                    
         &= \begin{cases}
                        0 \ \text{if} \ ||\frac{\nu^T X}{\lambda}||_\infty \leq 1\\
                        - \infty \ \text{otherwise}
            \end{cases}
\end{aligned}
`$
