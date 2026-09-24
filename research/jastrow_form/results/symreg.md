| term | channel | closed form (no free parameters on the scaled axis) | rel. weighted RMS |
|---|---|---|---|
| u | ud | exp(-x) | 2.000 |
| u | ud | 1/(1+x) - pade | 2.131 |
| u | uu | exp(-x) | 1.972 |
| u | uu | 1/(1+x) - pade | 2.114 |
| chi | ud | gauss exp(-ln2 x^2) | 0.032 |
| chi | ud | lorentz 1/(1+x^2) | 0.052 |
| chi | ud | sech | 0.039 |
| eta | ud | exp(-ln2 x) | 1.192 |
| eta | ud | gauss | 1.221 |

| term | channel | parsimony | expression (X0 = scaled r) | length | rel. weighted RMS | profiles pooled |
|---|---|---|---|---|---|---|
| u | ud | 0.01 | `div(-1.003, exp(X0))` | 4 | 0.064 | 22 |
| u | ud | 0.001 | `div(-1.003, exp(X0))` | 4 | 0.064 | 22 |
| u | ud | 0.0001 | `div(-1.003, exp(X0))` | 4 | 0.064 | 22 |
| u | uu | 0.01 | `div(-1.029, exp(X0))` | 4 | 0.056 | 19 |
| u | uu | 0.001 | `div(-1.029, exp(X0))` | 4 | 0.056 | 19 |
| u | uu | 0.0001 | `div(exp(sub(-0.991, X0)), -0.363)` | 6 | 0.056 | 19 |
| chi | ud | 0.01 | `add(mul(X0, 0.458), sub(1.057, X0))` | 7 | 0.062 | 24 |
| chi | ud | 0.001 | `exp(mul(mul(-0.673, X0), X0))` | 6 | 0.032 | 24 |
| chi | ud | 0.0001 | `exp(mul(mul(-0.673, X0), X0))` | 6 | 0.032 | 24 |
| eta | ud | 0.01 | `div(sub(X0, 1.788), exp(X0))` | 6 | 0.883 | 22 |
| eta | ud | 0.001 | `mul(exp(mul(mul(mul(X0, -1.980), mul(X0, exp(X0))), mul(X0, exp(X0)))), -1.877)` | 16 | 0.859 | 22 |
| eta | ud | 0.0001 | `div(div(sub(exp(sub(-1.366, -0.490)), sub(mul(add(X0, X0), sub(0.270, X0)), add(X0, -1.877))), exp(add(mul(X0, mul(mul(X0, X0), mul(X0, add(add(X0, X0), X0)))), sub(-1.325, 0.719)))), add(mul(add(0.207, 1.828), sub(1.900, sub(X0, X0))), div(add(0.447, X0), div(add(X0, X0), exp(X0)))))` | 56 | 0.857 | 22 |
