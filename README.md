# Beautiful_Spherical_Harmonics

A Simple demo of Spherical Harmonics to get directional reflection decomposition from complex scattering coefficients.
Use of Complex Fourier Orthonormal Basis as opposed to Real bases for RGB (color) case. 

## Spherical Harmonics

```math
Y_{\ell}^m(\theta,\varphi)=\sqrt{\frac{(2\ell+1)(\ell-m)!}{4\pi(\ell+m)!}}P_{\ell}^m(\cos\theta)e^{im\varphi}
```

This version of the code uses homogeneous polynomials of degree 6. Thus, the number of learnable coefficients is (6+1)^2 = 49.
