# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import numpy as np
import matplotlib.pyplot as plt

slopes = [
    (0.0, 0.0),
    (0.1, -0.1),
    (0.2,  0.2),
    (0.5,  0.0),
    (0.5,  0.2),
]

x = np.linspace(0, 1, 100)

for alpha, beta in slopes:
    poly = np.array([
        alpha + beta,
        -2*alpha - beta,
        alpha,
        0])
    y = np.polyval(poly, x)
    plt.plot(x, y, label=r'$\alpha$={:.1f}, $\beta$={:.1f}'.format(alpha, beta))

plt.legend()
plt.grid('on')
plt.xlabel('Horizontal page coordinate $x$')
plt.ylabel('Vertical displacement $z$')
plt.title('Cubic curves constrained to zero at endpoints')
plt.savefig('cubic_splines.png')
plt.show()
