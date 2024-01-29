# The barrier method for LASSO problem

Given $x_1, ..., x_n \in \mathbb{R}^d$ data vectors and $y_1, ..., y_n \in \mathbb{R}$ observations, we are searching for regression parameters $w \in \mathbb{R}^d$ which fit data inputs to observations $y$ by minimizing their squared difference. In a high dimensional setting (when $n << d$) a $l_1$ norm penalty is often used on the regression coefficients $w$ in order to enforce sparsity of the solution (so that $w$ will only have a few non-zeros entries). Such penalization has well known statistical properties, and makes the model both more interpretable and faster at test time.

From an optimization point of view we want to solve the following problem called LASSO (Least Absolute Shrinkage Operator and Selection Operator): 

$`
\text{minimize} \ \frac{1}{2} ||Xw - y||_2^2 + \lambda ||w||_1
`$


in the variable $w \in \mathbb{R}^d$, where $X = (x_1^t, ..., x_n^t) \in \mathbb{R}^{n \times d}$, $y=(y_1, ..., y_n) \in \mathbb{R}^n$ and $\lambda > 0$ is a regularization parameter.  

# Formalize the LASSO problem as a general Quadratic Problem

We note $z=Xw-y \in \mathbb{R}^n$. Then (LASSO) is equivalent to 
$`
\text{min}_{w \in \mathbb{R}^d, z\in \mathbb{R}^n, z=Xw-y} \frac{1}{2}||z||_2^2 + \lambda ||w||_1
`$

The corresponding Lagrangian writes, for $\nu \in \mathbb{R}^n$:
$`
L(w, z, \nu) = \frac{1}{2} ||z||_2^2 + \lambda ||w||_1 + \nu^t(Xw - y -z)
`$ 

Therefore, the Lagrange dual function $`g`$ writes:

$`
\begin{aligned} g(\nu) &= \text{inf}_{w\in\mathbb{R}^d, z\in\mathbb{R}^n} \left[  \frac{1}{2} |||z||_2^2 + \lambda ||w||_1 + \nu^t(Xw-y-z)\right] \\
                       &= \text{inf}_z \left[ \frac{1}{2} ||z||_2^2 - \nu^tz\right] + \text{inf}_w \left[ \lambda ||w||_1 + \nu^tXw \right]- \nu^ty
\end{aligned}
`$

We note $T_1(\nu) = \text{inf}_z \left[\frac{1}{2}||z||_2^2 - \nu^tz\right]$ and $T_2(\nu) = \text{inf}_w \left[ \lambda ||w||_1 + \nu^tXw \right]$ and we solve the two problems separately.

## Solving $T_{1}$

$\nabla_z \left(\frac{1}{2} ||z||_2^2 - \nu^tz \right) = \nabla_z \left( \frac{1}{2} z^tz - \nu^tz \right) = z - \nu$
therefore, $\nabla_z \left(\frac{1}{2} ||z||_2^2 - \nu^tz \right) = 0$ if and only if $z=\nu$. Injecting this result into $T_1$ we get:

$` \begin{aligned}
\text{inf}_z \left( \frac{1}{2} ||z||_2^2 - \nu^tz \right) &= \frac{1}{2}||\nu||_2^2 - \nu^t\nu \\
                                                           &= -\frac{1}{2} ||\nu||_2^2
\end{aligned}
`$


## Solving $T_{2}$

To solve $T_2$ we use the following result, with $`||x||_1^*`$ being the conjugate of $`||x||_1`$ :

 $`
||x||_1^* := \text{sup}_y (x^t y - ||y||_1) = -\text{inf}_y \left( ||y||_1 - x^ty \right) \\
           = \begin{cases}
                        + \infty \ \text{if} \ ||x||_\infty \gt 1 \\
                        0 \ \text{if} \ ||x||_\infty \leq 1
               \end{cases}
`$

This result can be shown easily considering that $`||x||_1 = \max_{||p||_\infty \leq 1} x^tp `$.


Therefore, we have 

$` \begin{aligned}
T_2(\nu) &= \text{inf}_w \left(\lambda ||w||_1 + \nu^tXw \right) \\
         &= \lambda \ \text{inf}_w \left( ||w||_1 - \left( -\frac{\nu^t X}{\lambda} \right) w \right) 
\end{aligned}
`$



$`T_2(\nu)         = \begin{cases}
                        0 \ \text{if} \ ||\frac{\nu^t X}{\lambda}||_\infty \leq 1\\
                        - \infty \ \text{otherwise}
            \end{cases}
`$

We get the following dual problem for (LASSO): 

$`
\text{max}_{||\frac{\nu^tX}{\lambda} \||_\infty \leq 1} -\frac{||\nu||_2^2}{2} -\nu^ty
\iff
\text{min}_{X^t\nu \leq \lambda} \ \frac{1}{2} \nu^t\nu \ + \ y^t \nu
`$

This last problem rewrites as the Quadratic Problem:

$`
\min_{A\nu \leq b} \ \nu^tQ\nu + p^t \nu
`$

with noting 
$`\begin{cases}
Q = \frac{1}{2}I_{n,n}\\
p = y \\
A = X^t \\
b = \lambda \left( \begin{array}\\
1 \\
\vdots \\
1 
\end{array}\right) 
\end{cases}`$

