import matplotlib.pyplot as plt
import numpy as np
import sympy

# create a bunch of symbols
a, b, c, d, x, alpha, beta = sympy.symbols('a b c d x alpha beta')

# create a polynomial function f(x)
f = a*x**3 + b*x**2 + c*x + d

# get its derivative f'(x)
fp = f.diff(x)

# evaluate both at x=0 and x=1
f0 = f.subs(x, 0)
f1 = f.subs(x, 1)
fp0 = fp.subs(x, 0)
fp1 = fp.subs(x, 1)

# we want a, b, c, d such that the following conditions hold:
#
#  f(0) = 0
#  f(1) = 0
#  f'(0) = alpha
#  f'(1) = beta

S = sympy.solve([f0, f1, fp0-alpha, fp1-beta], [a, b, c, d])

# print the analytic solution and plot a graphical example
coeffs = []

num_alpha = 0.3
num_beta = 0.03

for key in [a, b, c, d]:
    print key, '=', S[key]
    coeffs.append(S[key].subs(dict(alpha=num_alpha,
                                   beta=num_beta)))

xvals = np.linspace(0, 1, 101)
yvals = np.polyval(coeffs, xvals)

plt.plot(xvals, yvals)
plt.show()

