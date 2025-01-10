import torch
import math

# Complex constants for spherical harmonics
C0 = torch.tensor(0.28209479177387814, device='cuda')  # (1/2) * sqrt(1/pi)
C1 = torch.tensor([
    0.4886025119029199,   # sqrt(3/(4pi))
    0.4886025119029199,
    0.4886025119029199
    ], device='cuda')
C2 = torch.tensor([
    0.5462742152960396,   # 1/2 * sqrt(15/pi)
    1.0925484305920792,   # sqrt(15/pi)
    0.31539156525252005,  # 1/4 * sqrt(5/pi)
    1.0925484305920792,
    0.5462742152960396
    ], device='cuda')
C3 = torch.tensor([
    0.5900435899266435,   # sqrt(35/(16pi))
    2.890611442640554,    # sqrt(105/(4pi))
    0.4570457994644658,   # sqrt(21/(32pi))
    0.3731763325901154,   # sqrt(7/(16pi))
    0.4570457994644658,   
    1.445305721320277,    # sqrt(21/(32pi))
    0.5900435899266435
    ], device='cuda')
C4 = torch.tensor([
    0.6258357354491761,   # (3/8) * sqrt(35/pi)
    2.5033429417967046,   # (3/4) * sqrt(35/pi)
    0.9461746957575601,   # (3/8) * sqrt(5/pi)
    0.6690465435572892,   # (3/4) * sqrt(5/pi)
    0.10578554691520431,  # (3/16) * sqrt(1/pi)
    0.6690465435572892,
    0.47308734787878004,  # (3/8) * sqrt(5/pi)
    2.5033429417967046,
    0.6258357354491761
    ], device='cuda')

C5 = torch.tensor([
    0.6563820568,   # (some factor) for m=-5
    3.2819102840,   # (some factor) for m=-4
    2.1685288640,   # ...
    0.9003163161,
    0.2102610435,
    0.1230486685,   # for m= 0
    0.2102610435,
    0.9003163161,
    2.1685288640,
    3.2819102840,
    0.6563820568
], device='cuda')

C6 = torch.tensor([
    0.6831841051,   # for m=-6
    3.5449077018,   # for m=-5
    2.5033429418,   # for m=-4
    1.0206207261,   # for m=-3
    0.2561551815,   # for m=-2
    0.1467802647,   # for m=-1
    0.0945061722,   # for m=0
    0.1467802647,   # for m=1
    0.2561551815,   # for m=2
    1.0206207261,   # for m=3
    2.5033429418,   # for m=4
    3.5449077018,   # for m=5
    0.6831841051    # for m=6
], device='cuda')


def eval_complex_sh(deg, sh, dirs):
    """
    Evaluate complex spherical harmonics at unit directions
    using hardcoded SH polynomials.
    Works with torch tensors. Assumes dirs are normalized.
    
    Args:
        deg: int SH deg. Currently, 0-4 supported
        sh: complex tensor of SH coeffs [..., C, (deg + 1) ** 2]
        dirs: tensor of normalized directions [..., 3]
    Returns:
        complex tensor [..., C]
    """
    assert deg <= 6 and deg >= 0
    coeff = (deg + 1) ** 2
    assert sh.shape[-1] >= coeff

    # Convert directions to spherical coordinates
    x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
    # Since dirs are normalized, z = cos(theta) directly
    cos_theta = z
    theta = torch.arccos(cos_theta).to(sh.dtype)
    phi = torch.atan2(y, x).to(sh.dtype)
    
    result = C0 * sh[..., 0]  # l = 0
    
    if deg > 0:
        # l = 1
        sin_theta = torch.sqrt(1 - cos_theta * cos_theta)  # More efficient than sin(arccos(z))
        exp_phi = torch.exp(1j * phi)
        
        result = (result + 
                 C1[0] * exp_phi.conj() * sin_theta * sh[..., 1] +
                 C1[1] * cos_theta * sh[..., 2] +
                 -C1[2] * exp_phi * sin_theta * sh[..., 3])

        if deg > 1:
            # l = 2
            sin_theta_sq = sin_theta * sin_theta
            exp_2phi = exp_phi * exp_phi
            cos_theta_sq = cos_theta * cos_theta
            
            result = (result +
                     C2[0] * exp_2phi.conj() * sin_theta_sq * sh[..., 4] +
                     C2[1] * exp_phi.conj() * sin_theta * cos_theta * sh[..., 5] +
                     C2[2] * (3 * cos_theta_sq - 1) * sh[..., 6] +
                     -C2[3] * exp_phi * sin_theta * cos_theta * sh[..., 7] +
                     C2[4] * exp_2phi * sin_theta_sq * sh[..., 8])

            if deg > 2:
                # l = 3
                exp_3phi = exp_2phi * exp_phi
                cos_theta_3 = cos_theta_sq * cos_theta
                
                result = (result +
                         C3[0] * exp_3phi.conj() * sin_theta * sin_theta_sq * sh[..., 9] +
                         C3[1] * exp_2phi.conj() * sin_theta_sq * cos_theta * sh[..., 10] +
                         C3[2] * exp_phi.conj() * sin_theta * (5 * cos_theta_sq - 1) * sh[..., 11] +
                         C3[3] * (5 * cos_theta_3 - 3 * cos_theta) * sh[..., 12] +
                         -C3[4] * exp_phi * sin_theta * (5 * cos_theta_sq - 1) * sh[..., 13] +
                         C3[5] * exp_2phi * sin_theta_sq * cos_theta * sh[..., 14] +
                         -C3[6] * exp_3phi * sin_theta * sin_theta_sq * sh[..., 15])

                if deg > 3:
                    # l = 4
                    exp_4phi = exp_2phi * exp_2phi
                    cos_theta_4 = cos_theta_sq * cos_theta_sq
                    
                    result = (result +
                             C4[0] * exp_4phi.conj() * sin_theta_sq * sin_theta_sq * sh[..., 16] +
                             C4[1] * exp_3phi.conj() * sin_theta * sin_theta_sq * cos_theta * sh[..., 17] +
                             C4[2] * exp_2phi.conj() * sin_theta_sq * (7 * cos_theta_sq - 1) * sh[..., 18] +
                             C4[3] * exp_phi.conj() * sin_theta * (7 * cos_theta_3 - 3 * cos_theta) * sh[..., 19] +
                             C4[4] * (35 * cos_theta_4 - 30 * cos_theta_sq + 3) * sh[..., 20] +
                             -C4[5] * exp_phi * sin_theta * (7 * cos_theta_3 - 3 * cos_theta) * sh[..., 21] +
                             C4[6] * exp_2phi * sin_theta_sq * (7 * cos_theta_sq - 1) * sh[..., 22] +
                             -C4[7] * exp_3phi * sin_theta * sin_theta_sq * cos_theta * sh[..., 23] +
                             C4[8] * exp_4phi * sin_theta_sq * sin_theta_sq * sh[..., 24])
                    
                    if deg > 4:
                        # l = 5
                        exp_5phi = exp_4phi * exp_phi
                        cos_theta_5 = cos_theta_sq * cos_theta_3
                        
                        result = (result +
                                 C5[0] * exp_5phi.conj() * sin_theta * sin_theta_sq * sin_theta_sq * sh[..., 25] +
                                 C5[1] * exp_4phi.conj() * sin_theta_sq * sin_theta_sq * cos_theta * sh[..., 26] +
                                 C5[2] * exp_3phi.conj() * sin_theta * sin_theta_sq * (9 * cos_theta_sq - 1) * sh[..., 27] +
                                 C5[3] * exp_2phi.conj() * sin_theta_sq * (9 * cos_theta_3 - 3 * cos_theta) * sh[..., 28] +
                                 C5[4] * exp_phi.conj() * sin_theta * (9 * cos_theta_4 - 6 * cos_theta_sq + 1) * sh[..., 29] +
                                 C5[5] * (63 * cos_theta_5 - 70 * cos_theta_3 + 15 * cos_theta) * sh[..., 30] +
                                 -C5[6] * exp_phi * sin_theta * (9 * cos_theta_4 - 6 * cos_theta_sq + 1) * sh[..., 31] +
                                 C5[7] * exp_2phi * sin_theta_sq * (9 * cos_theta_3 - 3 * cos_theta) * sh[..., 32] +
                                 -C5[8] * exp_3phi * sin_theta * sin_theta_sq * (9 * cos_theta_sq - 1) * sh[..., 33] +
                                 C5[9] * exp_4phi * sin_theta_sq * sin_theta_sq * cos_theta * sh[..., 34] +
                                 -C5[10] * exp_5phi * sin_theta * sin_theta_sq * sin_theta_sq * sh[..., 35])
                        
                        if deg > 5:
                            # l = 6
                            exp_6phi = exp_5phi * exp_phi
                            cos_theta_6 = cos_theta_5 * cos_theta
                            
                            result = (result +
                                    C6[0] * exp_6phi.conj() * sin_theta_sq * sin_theta_sq * sin_theta_sq * sh[..., 36] +
                                    C6[1] * exp_5phi.conj() * sin_theta * sin_theta_sq * sin_theta_sq * cos_theta * sh[..., 37] +
                                    C6[2] * exp_4phi.conj() * sin_theta_sq * sin_theta_sq * (11 * cos_theta_sq - 1) * sh[..., 38] +
                                    C6[3] * exp_3phi.conj() * sin_theta * sin_theta_sq * (11 * cos_theta_3 - 3 * cos_theta) * sh[..., 39] +
                                    C6[4] * exp_2phi.conj() * sin_theta_sq * (11 * cos_theta_4 - 6 * cos_theta_sq + 1) * sh[..., 40] +
                                    C6[5] * exp_phi.conj() * sin_theta * (11 * cos_theta_5 - 10 * cos_theta_3 + 5 * cos_theta) * sh[..., 41] +
                                    C6[6] * (231 * cos_theta_6 - 315 * cos_theta_4 + 105 * cos_theta_sq - 5) * sh[..., 42] +
                                    -C6[7] * exp_phi * sin_theta * (11 * cos_theta_5 - 10 * cos_theta_3 + 5 * cos_theta) * sh[..., 43] +
                                    C6[8] * exp_2phi * sin_theta_sq * (11 * cos_theta_4 - 6 * cos_theta_sq + 1) * sh[..., 44] +
                                    -C6[9] * exp_3phi * sin_theta * sin_theta_sq * (11 * cos_theta_3 - 3 * cos_theta) * sh[..., 45] +
                                    C6[10] * exp_4phi * sin_theta_sq * sin_theta_sq * (11 * cos_theta_sq - 1) * sh[..., 46] +
                                    -C6[11] * exp_5phi * sin_theta * sin_theta_sq * sin_theta_sq * cos_theta * sh[..., 47] +
                                    C6[12] * exp_6phi * sin_theta_sq * sin_theta_sq * sin_theta_sq * sh[..., 48])



    if result.dim() > 0:
        result = result.squeeze(-1)

    return result

def scatter2ComplexSH(scattering_coeff):
    """
    Convert scattering coefficients to complex SH coefficients
    
    Args:
        scattering_coeff: complex tensor of scattering coefficients
    Returns:
        complex tensor of SH coefficients
    """
    return scattering_coeff / C0

def ComplexSH2scatter(sh):
    """
    Convert complex SH coefficients back to scattering coefficients
    
    Args:
        sh: complex tensor of SH coefficients
    Returns:
        complex tensor of scattering coefficients
    """
    return sh * C0