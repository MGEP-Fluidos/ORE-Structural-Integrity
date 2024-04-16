# -*- coding: utf-8 -*-
"""
Module Name: Spectral methods
Description: This module defines the spectral models for fatigue assessment.
"""

import numpy as np
import scipy.stats as st
import scipy.signal as sig
from scipy import integrate, special
from scipy.optimize import fsolve


def per_spectral_mom(f, Pxx, n):
    '''
    Function that calculates the spectral moments
    '''
    # Compute the spectral moments using Simpson's rule
    Mo = np.zeros_like(n, dtype=float)
    for i, ni in enumerate(n):
        integrand = (2 * np.pi * f)**ni * Pxx
        Mo[i] = integrate.simps(integrand, f)
    return Mo

def spectral_moments(f, Pxx):
    '''
    Function that returns the first five spectral moments
    '''
    Mo = per_spectral_mom(f, Pxx, n=[0, 1, 2, 3, 4])
    return Mo


def gamma_func(z):
    '''
    Function that returns the gamma function
    '''
    gamma = integrate.quad(lambda alfa: alfa**(z - 1)
                           * np.exp(-alfa), 0, np.inf)
    return gamma[0]

def get_mu(f, Pxx, k):
    landa = per_spectral_mom(f, Pxx, [0.01, k+0.01, 2*k+0.01])
    mu = landa[1] / np.sqrt(landa[0]*landa[2])
    return mu


def K_func(k):
    '''
    Function that returns the K function
    '''
    K = integrate.quad(lambda sigma: sigma**(1 + k)
                       * np.exp(-sigma**2 / 2), 0, np.inf)
    return K[0]

def P_pdf(Mo_1, Mo_2):
    
    m0 = Mo_1[0] + Mo_2[0]
    m0L = Mo_1[0]/m0
    m0H = Mo_2[0]/m0

    def pdf(x):
        q = x / np.sqrt(m0)
        pdf =  (m0L*q*np.exp(- q**2 /(2*m0L)) + m0H*q*np.exp(- q**2 /(2*m0H)) \
                    + np.sqrt(2*np.pi*m0L*m0H) * (q**2-1) * np.exp(-q**2/2) \
                    * (st.norm.cdf(np.sqrt(m0L/m0H)*q) + st.norm.cdf(np.sqrt(m0H/m0L)*q) - 1)) / np.sqrt(m0)
        return pdf
    
    return pdf


###############################################################################
'                               Spectral methods                              '
###############################################################################


def Narrowband(k, C, Mo, ZUCF):
    return ZUCF / C * (2 * np.sqrt(2 * Mo[0]))**k * gamma_func(1 + k / 2)

###############################################################################

# a)	Narrowband correction factor


def Wirsching_Light(k, Mo):
    a = 0.926 - 0.033*k
    b = 1.587*k - 2.323
    alfa_2 = Mo[2] / np.sqrt(Mo[0] * Mo[4])
    eps = np.sqrt(1-alfa_2**2)
    rho_WL = a + (1 - a) * (1 - eps)**b
    return rho_WL

def Tovo_Benasciutti(Mo, k):
    '''
    Function that calculates correction coefficients to apply the Tovo & Benasciutti
    approach with the nonlinear coefficient proposed by Cianetti
    '''
    alfa_1 = Mo[1] / np.sqrt(Mo[0] * Mo[2])
    alfa_2 = Mo[2] / np.sqrt(Mo[0] * Mo[4])
    b = ((alfa_1 - alfa_2) * (1.112 * (1 + alfa_1 * alfa_2 - (alfa_1 + alfa_2))
         * np.exp(2.11 * alfa_2) + (alfa_1 - alfa_2))) / (alfa_2 - 1)**2
    rho_TB = (b + (1 - b) * alfa_2**(k - 1))
    return rho_TB


def alfa075(Mo, f, Pxx):
    landa = per_spectral_mom(f, Pxx, n=[0.75, 2*0.75])
    a075 = landa[0] / np.sqrt(Mo[0] * landa[1])
    rho075 = a075**2
    return rho075

def Ortiz_Chen(k, Mo, f, Pxx):
    landa = per_spectral_mom(f, Pxx, n=[2 / k, 2 / k + 2])
    betha = np.sqrt((Mo[2] * landa[0]) / (Mo[0] * landa[1]))
    alfa2 = Mo[2] / np.sqrt(Mo[0] * Mo[4])
    rho_OC = betha**k / alfa2
    return rho_OC

def Modified_Tovo_Benasciutti(Mo, k, S):
    '''
    Function that calculates correction coefficients to apply the Tovo & Benasciutti
    approach with the nonlinear coefficient proposed by Cianetti
    '''
    def P_values(k):
        if k == 3:
            P = [1.021, 0.815, 2.038, 1.003, -0.274, -0.019, 0.212, 0.048]
        elif k == 4:
            P = [0.975, 1.591, 2.923, 1.020, -0.287, -0.024, 0.262, 0.052]
        elif k == 5:
            P = [0.938, 2.235, 3.661, 1.032, -0.293, -0.029, 0.311, 0.054]
        elif k == 6:
            P = [0.913, 2.820, 4.325, 1.036, -0.280, -0.035, 0.357, 0.054]
        return P
    P = P_values(k)
    alfa_1 = Mo[1] / np.sqrt(Mo[0] * Mo[2])
    alfa_2 = Mo[2] / np.sqrt(Mo[0] * Mo[4])
    g3 = st.skew(S)
    g4 = st.kurtosis(S, fisher=False)
    b = ((alfa_1 - alfa_2) * (P[0] + P[1]*alfa_2 - P[2]*alfa_1*alfa_2)) / (1 - P[3]*alfa_2)**2 * (P[4]*g3**2 + P[5]*g4**2 + P[6]*g4 + P[7]*g3**2*g4 + 1 - 9*P[5] - 3*P[6])
    rho_MTB = (b + (1 - b) * alfa_2**(k - 1))
    return rho_MTB


###############################################################################

# b)	RFC PDF approximation


def Dirlik(Mo, k, C):
    vp = (1 / (2 * np.pi)) * np.sqrt(Mo[4] / Mo[2])
    alfa2 = Mo[2] / np.sqrt(Mo[0] * Mo[4])
    xm = Mo[1] / Mo[0] * np.sqrt(Mo[2] / Mo[4])
    D1 = 2 * (xm - alfa2**2) / (1 + alfa2**2)
    R = (alfa2 - xm - D1**2) / (1 - alfa2 - D1 + D1**2)
    D2 = (1 - alfa2 - D1 + D1**2) / (1 - R)
    D3 = 1 - D1 - D2
    Q = (1.25 * (alfa2 - D3 - D2 * R)) / D1
    d = vp * (2 * np.sqrt(Mo[0]))**k / C * (D1 * Q**k * gamma_func(1 + k) + \
              np.sqrt(2)**k * gamma_func(1 + k / 2) * (D2 * np.abs(R)**k + D3))
    return d


def Park(Mo, k, C, f, Pxx):
    vp = (1 / (2 * np.pi)) * np.sqrt(Mo[4] / Mo[2])
    alfa2 = Mo[2] / np.sqrt(Mo[0] * Mo[4])
    alfa0_95 = per_spectral_mom(f, Pxx, [0.95])[0] / np.sqrt(Mo[0] * per_spectral_mom(f, Pxx, [2 * 0.95])[0])
    alfa1_97 = per_spectral_mom(f, Pxx,[1.97])[0] / np.sqrt(Mo[0] * per_spectral_mom(f, Pxx, [2 * 1.97])[0])
    alfa0_54 = per_spectral_mom(f, Pxx, [0.54])[0] / np.sqrt(Mo[0] * per_spectral_mom(f, Pxx, [2 * 0.54])[0])
    alfa0_93 = per_spectral_mom(f, Pxx, [0.93])[0] / np.sqrt(Mo[0] * per_spectral_mom(f, Pxx, [2 * 0.93])[0])
    alfa1_95 = per_spectral_mom(f, Pxx, [1.95])[0] / np.sqrt(Mo[0] * per_spectral_mom(f, Pxx, [2 * 1.95])[0])
    MRR1 = sR1 = alfa2
    MRR2 = alfa0_95 * alfa1_97 
    MRR3 = alfa0_54 * alfa0_93 * alfa1_95
    cR1 = (MRR2 - MRR3) / (sR1**2 * (1 - sR1))
    cR2 = (-sR1 * MRR2 + MRR3) / (1 - sR1)
    cG = 1 - cR1 - cR2
    sG = np.sqrt(np.pi) * gamma_func(1.5) / \
        (cG * gamma_func(1)) * (MRR1 - cR1 * sR1 - cR2)
    d = vp * (2*np.sqrt(2 * Mo[0]))**k / C * (cG / np.sqrt(np.pi) * sG**k * gamma_func(
        (1 + k) / 2) + cR1 * sR1**k * gamma_func(1 + k / 2) + cR2 * gamma_func(1 + k / 2))
    if d < 0:
        d = 0
    return d


def Jun_Park(k, C, Mo, f, Pxx):
    vp = (1 / (2 * np.pi)) * np.sqrt(Mo[4] / Mo[2])
    
    alfa1 = Mo[1] / np.sqrt(Mo[0] * Mo[2])
    alfa2 = Mo[2] / np.sqrt(Mo[0] * Mo[4])
    delta_alfa = alfa1 - alfa2
    
    rho = alfa1**1.1 * alfa2**0.9
    
    mu1 = get_mu(f, Pxx, 1)
    mu052 = get_mu(f, Pxx, 0.52)
    
    MRR1 = rho * mu1**-0.96
    MRR2 = rho * mu1**-0.02
    MRR3 = rho * mu052
    
    sR = alfa2
    
    D1 = 2*(alfa1*alfa2 - alfa2**2)/(1 + alfa2**2)
    D2 = (MRR2 - MRR3)/(sR**2*(1 - sR))
    D3 = (-sR*MRR2 + MRR3)/(1 - sR)
    D4 = 1 - D1 - D2 - D3
    
    A1 = gamma_func(2) / (np.sqrt(2)*gamma_func(1+1/2))
    B1 = gamma_func(1) / (np.sqrt(np.pi)*gamma_func(1+1/2))
    
    sH = 1/(B1 * D4) * (MRR1 - D1**2 - D2*sR - D3)
    sE = 1/(A1 * D1) * (MRR1 - D2*sR - D3 - B1*D4*sH)
    
    # # Correction factor Qc is validated under following conditions
    if not 0 <= alfa1 - alfa2 <= 1 and 0 <= alfa2 <= 1 and np.sqrt(1-alfa1**2) > 0.3:
        print('Correction factor Qc is not validated for given alfa1 and alfa2. Results should be evaluated carefully.')
    
    Qc = 0.903 - 0.28*delta_alfa + 4.448*delta_alfa**2 - 15.739*delta_alfa**3 + 19.57*delta_alfa**4 - 8.054*delta_alfa**5 + 1.013*alfa2 - 4.178*alfa2**2 + 8.362*alfa2**3 - 7.993*alfa2**4 + 2.886*alfa2**5

    d = Qc*vp*(2*np.sqrt(2*Mo[0]))**k / C * (D1/(np.sqrt(2)**k)*sE**k*gamma_func(1+k) + D2*sR**k*gamma_func(1+k/2) + D3*gamma_func(1+k/2) + D4/(np.sqrt(np.pi))*sH**k*gamma_func((1+k)/2))
            
    return d


def Zhao_Baker(k, C, Mo, Pxx, f):
    landa = per_spectral_mom(f, Pxx, n=[0.75, 2 * 0.75])
    alfa2 = Mo[2] / np.sqrt(Mo[0] * Mo[4])
    alfa075 = landa[0] / np.sqrt(Mo[0] * landa[1])
    if alfa2 < 0.9:
        b = 1.1
    else:
        b = 1.1 + 9 * (alfa2 - 0.9)
    if k == 3:
        if alfa075 >= 0.5:
            rho_ZB = -0.4154 + 1.392 * alfa075
        else:
            rho_ZB = 0.28

        def eq(d):
            return gamma_func(1 + 3 / b) * (1 - alfa2) * d**3 + 3 * gamma_func(1 + 1 / b) * (
                rho_ZB * alfa2 - 1) * d + 3 * np.sqrt(np.pi / 2) * alfa2 * (1 - rho_ZB)
        try:
            root = fsolve(eq, 0)
        except BaseException:
            root = fsolve(eq, np.random.rand() * 5.0)
        a = root**(-b)
    else:
        a = 8 - 7 * alfa2

    w = (1 - alfa2) / (1 - np.sqrt(2 / np.pi)
                       * gamma_func(1 + 1 / b) * a**(-1 / b))
    vp = (1 / (2 * np.pi)) * np.sqrt(Mo[4] / Mo[2])
    d = vp * (2 * np.sqrt(Mo[0]))**k / C * (w * a**(-k / b) * gamma_func(
        1 + k / b) + (1 - w) * np.sqrt(2)**k * gamma_func(1 + k / 2))
    return d[0]


def wu(k, C, Mo):
    
    def get_a_value(a, k):
        table = {
            'a00': {3: 29.7, 4: 124, 5: 570, 7: 15000},
            'a10': {3: 0.2, 4: -43, 5: -510, 7: -34000},
            'a11': {3: -10, 4: -19, 5: 100, 7: 17000},
            'a20': {3: -198.3, 4: -1146, 5: -6100, 7: -172000},
            'a21': {3: 200, 4: 1323, 5: 7960, 7: 283000},
            'a22': {3: -34.2, 4: -315, 5: -2270, 7: -103000},
            'a30': {3: -52.1, 4: -371, 5: -2300, 7: -84000},
            'a31': {3: 294, 4: 1872, 5: 10700, 7: 344000},
            'a32': {3: -265.4, 4: -1774, 5: -10590, 7: -367000},
            'a33': {3: 36, 4: 348, 5: 2430, 7: 101000}
        }
        
        return table.get(a, {}).get(k)
    
    alfa_1 = Mo[1] / np.sqrt(Mo[0] * Mo[2])
    alfa_2 = Mo[2] / np.sqrt(Mo[0] * Mo[4])
    
    e1 = np.sqrt(1 - alfa_1**2)
    e2 = np.sqrt(1 - alfa_2**2)
    
    Mk = 0
    
    for i in range(0, 4):
        for j in range(0, i+1):
            a = get_a_value(f'a{i}{j}', k)
            Mk += a * e1**(i-j) * e2**j
    
    d = (Mo[0])**(k/2) / (C*2*np.pi) * np.sqrt(Mo[4]/Mo[2]) * Mk 
    
    return d


###############################################################################

# c)	Combined fatigue damage – cycle type damage combination

def Jiao_Moan(m, v0, Mo_w, Mo_l, v0_l, v0_w, full_integration = False, C = 0):
    '''
    Function that returns the fatigue damage according to the Jiao & Moan approach
    [Probabilistic analysis of fatigue due to Gaussian load processes
    doi = {10.1016/0266-8920(90)90010-H}]
    '''
    d_w = np.sqrt(1 - Mo_w[1]**2 / (Mo_w[0] * Mo_w[2])
                  )  # bandwidth parameter that may taken equal to 0.1
    m1 = Mo_l[0] / (Mo_l[0] + Mo_w[0])
    m2 = Mo_w[0] / (Mo_l[0] + Mo_w[0])
    
    v0_p = m1 * v0_l * np.sqrt(1 + m2 / m1 * (v0_w / v0_l * d_w)**2)
    
    if full_integration == False:
        rho = v0_p / v0 * (m1**(m / 2 + 2) * (1 - np.sqrt(m2 / m1)) + np.sqrt(np.pi * m1 * m2)
                           * (m * gamma_func(m / 2 + 1 / 2)) / (gamma_func(m / 2 + 1))) + v0_w / v0 * m2**(m / 2)
        return rho
    else:
        # Damage due to small cycles (HF component):
        d_small = Narrowband(m, C, Mo_w, v0_w)
        
        # Damage due to large cycles (HF + LF component):
        pdf_P = P_pdf(Mo_l, Mo_w)
        I_large = integrate.quad(lambda x:  x**m * pdf_P(x), 0, np.inf)[0]
    
        d_large = 2**m * v0_p * I_large / C
                
        return d_small + d_large


def dual_narrowband(m, v0, Mo_w, Mo_l, v0_l, v0_w):
    '''
    Function that returns the fatigue damage according to the Jiao & Moan approach
    [Probabilistic analysis of fatigue due to Gaussian load processes
    doi = {10.1016/0266-8920(90)90010-H}]
    '''
    d_w = 0.1  # bandwidth parameter that may taken equal to 0.1
    m1 = Mo_l[0] / (Mo_l[0] + Mo_w[0])
    m2 = Mo_w[0] / (Mo_l[0] + Mo_w[0])
    v0_p = m1 * v0_l * np.sqrt(1 + m2 / m1 * (v0_w / v0_l * d_w)**2) # Peak frequency
    rho = v0_p / v0 * (m1**(m / 2 + 2) * (1 - np.sqrt(m2 / m1)) + np.sqrt(np.pi * m1 * m2)
                       * (m * gamma_func(m / 2 + 1 / 2)) / (gamma_func(m / 2 + 1))) + v0_w / v0 * m2**(m / 2)
    return rho

def Sakai_Okamura(k, C, Mo_l, Mo_w, ZUCF_l, ZUCF_w):
    
    d = 2**k*2**(k/2) / (2*np.pi*C) * gamma_func(k/2 + 1) * (Mo_l[0]**((k-1)/2)*Mo_l[2]**(1/2) + Mo_w[0]**((k-1)/2)*Mo_w[2]**(1/2))
    
    return d


def numerical_integration_trapezoidal(f, a, b, n = 1000):
    h = (b - a) / n
    integral = 0.5 * (f(a) + f(b))
    for i in range(1, n):
        x_i = a + i * h
        integral += f(x_i)
    integral *= h
    return integral


def Fu_Cebon(k, C, Mo_w, Mo_l, v0_w, v0_l, solution = 'Binomial expansion'):
    
    # Damage due to small cycles (HF component):
    d_small = Narrowband(k, C, Mo_w, v0_w - v0_l)
    
    # Damage due to large cycles (LF and HF components):
    if solution == 'Full integration':
        V = 1 / Mo_w[0]
        U = 1 / (2*Mo_l[0]) + 1 / (2*Mo_w[0])
        
        def integrand_y(y, S, U, V):
            # return np.log(S * y - y**2) - U * y**2 + V * S * y
            return (S * y - y**2) * np.exp(-U * y**2 + V * S * y)
        
        def integrand_S(S, Mo_w, k, U, V):
            integrand_y_partial = lambda y: integrand_y(y, S, U, V)
            result = integrate.quad(integrand_y_partial, 0, S)[0]
            return np.exp(-S**2 / (2 * Mo_w[0]**2)) * S**(k) * result
        
        # Peak distribution of the large amplitude cycles
        I_large = integrate.quad(lambda S: integrand_S(S, Mo_w, k, U, V), 0, np.inf)[0]
        I_large = I_large/(Mo_l[0] * Mo_w[0])
        d_large = 2**k * v0_l /C * I_large
        
    elif solution == 'Binomial expansion':
        std_HF = np.sqrt(Mo_w[0])
        std_LF = np.sqrt(Mo_l[0])
        
        # Low proposed a solution for d_large based on a finite binomial series expansion, where each of the 
        # raw moments have analytical solutions.
        # For noninteger k, the resulting binomial expansion has an infinite number of terms, 
        #but the series may be suitably truncated.
        J  = 0
        for j in range(int(k)+1):
            J += special.binom(k, j) * std_HF**j * std_LF**(k-j) * gamma_func(1 + j/2) * gamma_func(1 + (k-j)/2)
            
        d_large = 2**(3*k/2) * v0_l / C * J    
        
    elif solution == 'JM approximation':
        pdf_P   = P_pdf(Mo_l, Mo_w)
        I_large = integrate.quad(lambda x:  x**k * pdf_P(x), 0, np.inf)[0]
        d_large = 2**k * v0_l /C * I_large

    # Total damage
    d = d_small +  d_large
    return d

def Modified_Fu_Cebon(k, C, Mo_w, Mo_l, v0_w, v0_l, solution = 'Binomial expansion'):
    # Benasciutti and Tovo proposed a modification of the Fu Cebon method
    # by using v0_p (assumed as the correct frequency for large cycles) from Jiao Moan's
    # methods instead of v0_l
    d_w = np.sqrt(1 - Mo_w[1]**2 / (Mo_w[0] * Mo_w[2])
                  )  # bandwidth parameter that may taken equal to 0.1
    m1 = Mo_l[0] / (Mo_l[0] + Mo_w[0])
    m2 = Mo_w[0] / (Mo_l[0] + Mo_w[0])
    v0_p = m1 * v0_l * np.sqrt(1 + m2 / m1 * (v0_w / v0_l * d_w)**2)
    
    # Damage due to small cycles (HF component):
    d_small = Narrowband(k, C, Mo_w, v0_w - v0_p)
    
    # Damage due to large cycles (LF and HF components):
    if solution == 'Full integration':      
        V = 1 / Mo_w[0]
        U = 1 / (2*Mo_l[0]) + 1 / (2*Mo_w[0])
        
        def integrand_y(y, S, U, V):
            return (S * y - y**2) * np.exp(-U * y**2 + V * S * y)
    
        def integrand_S(S, Mo_w, k, U, V):
            integrand_y_partial = lambda y: integrand_y(y, S, U, V)
            result = integrate.quad(integrand_y_partial, 0, S)[0]
            return np.exp(-S**2 / (2 * Mo_w[0]**2)) * S**(k) * result
        
        # Peak distribution of the large amplitude cycles
        I_large = integrate.quad(lambda S: integrand_S(S, Mo_w, k, U, V), 0, np.inf)[0] / (Mo_l[0] * Mo_w[0])
        
        d_large = 2**k * v0_l /C * I_large
        
    elif solution == 'Binomial expansion':
        std_HF = np.sqrt(Mo_w[0])
        std_LF = np.sqrt(Mo_l[0])
        
        # Low proposed a solution for d_large based on a finite binomial series expansion, where each of the 
        # raw moments have analytical solutions.
        # For noninteger k, the resulting binomial expansion has an infinite number of terms, 
        #but the series may be suitably truncated.
        J  = 0
        for j in range(int(k)+1):
            J += special.binom(k, j) * std_HF**j * std_LF**(k-j) * gamma_func(1 + j/2) * gamma_func(1 + (k-j)/2)
            
        d_large = 2**(3*k/2) * v0_l / C * J   
        
    elif solution == 'JM approximation':
        pdf_P = P_pdf(Mo_l, Mo_w)
        I_large = integrate.quad(lambda x:  x**k * pdf_P(x), 0, np.inf)[0]
        d_large = 2**k * v0_p * I_large / C
    
    # Total damage
    d = d_small + d_large
    return d
    
def LowBimodal(k, C, v0_w, v0_l, Mo_w, Mo_l):
    beta = v0_w / v0_l

    def eps(r_l, theta, beta): 
        return np.pi / (2*beta) * r_l * np.abs(np.sin(theta))

    def pdf_r(r, M0): 
        return st.rayleigh.pdf(r, scale = np.sqrt(M0))

    def pdf_theta(theta, beta): 
        return st.uniform.pdf(theta, loc= np.pi/(4*beta), scale= np.pi/2 - np.pi/(4*beta))
    
    def rho(r_l, r_h, beta, j):
        c = beta * r_h / (r_l + beta**2 * r_h)
        return r_l * c**j + r_h * (beta * c - 1)**j / special.factorial(j)

    def integral_small(theta, r_l): 
        return integrand(k, eps(r_l, theta, beta), Mo_w[0]) * pdf_theta(theta, beta) * pdf_r(r_l, Mo_l[0])

    def integral_large(r_h, r_l): 
        return integrand(k, r_l, r_h, beta) * pdf_r(r_h, Mo_w[0]) * pdf_r(r_l, Mo_l[0])

    def integrand(k, *args):
        if len(args) == 2:  # Small cycles
            eps_val = args[0]
            M0_w = args[1]
            # To simplify the evaluation of the triple integral, the innermorst integral is evaluated analytically
            # The binomial coefficients (Ik) are computed first:
            Ik = []
            Ik.append(np.exp(-eps_val**2 / (2 * M0_w)))
            Ik.append(eps_val * Ik[0] + np.sqrt(2 * np.pi) * np.sqrt(M0_w) * st.norm.cdf(-eps_val / np.sqrt(M0_w)))
            for i in range(2, int(k) + 1):
                Ik.append(eps_val**i * np.exp(-eps_val**2 / (2 * M0_w)) + i * M0_w * Ik[i-2])
            
            # The analytical expression of the innermost integral is then computed (based on a binomial series expansion)
            J  = 0
            for j in range(int(k)+1):
                J += special.binom(k, j) * (-eps_val)**(k - j) * Ik[j]
            return J
        else:  # Large cycles
            # An explicit solution for the innermost integral is used. The integral is
            # expanded  through a McLaurin series and then a binomial expansion is
            # performed retaining terms up to sixth order. This ensures sufficient 
            # engineering precision (within 0.5%) in the damage estimate for up to k = 6.
            r_l, r_h, beta = args
            r = r_l + r_h
            rho_2 = rho(r_l, r_h, beta, 2)
            rho_4 = rho(r_l, r_h, beta, 4)
            rho_6 = rho(r_l, r_h, beta, 6)
            J = r**k * (np.pi - k * rho_2 * np.pi**3 / (3*r) + k / (5*r) * (rho_4 + (k - 1) * rho_2**2 / (2*r)) * np.pi**5
                              - k / (7*r) * (rho_6 + (k - 1) * rho_2 * rho_4 / r + (k - 1) * (k - 2) * rho_2**3 / (6 * r**2)) * np.pi**7)
            return J

    Is, _ = integrate.dblquad(integral_small, 0, np.inf, lambda r_l: np.pi/(4*beta), lambda r_l: np.pi / 2)
    Il = integrate.dblquad(integral_large, 0, np.inf, lambda r_l: 0, lambda r_l: np.inf)[0] / np.pi

    v0_small = v0_w - v0_l

    ds = v0_small * 2**k/ C * Is
    dl = v0_l * 2**k / C * Il

    return ds + dl


def Low(k, C, Mo, Mo_l, Mo_w, ZUCF, ZUCF_l, ZUCF_w):
    
    # Ratio between high and low frequencies
    beta = ZUCF_w/ZUCF_l
    
    # Normalised moment of the HF component
    sH = np.sqrt(Mo_w[0] / (Mo_l[0] + Mo_w[0]))
    
    # Low frequency damage ratio
    b1 = (1.111 + 0.7421*k - 0.0724*k**2) * beta**(-1) + (2.403 - 2.483*k) * beta**(-2)
    b2 = (-10.45 + 2.65*k ) * beta**(-1) + (2.607 + 2.63*k - 0.0133*k**2) * beta**(-2)
    
    L = (b1*sH + b2*sH**2 - (b1+b2)*sH**3 + sH**k) * (beta - 1) + 1
    
    # Narrowband damage
    d_NB = Narrowband(k, C, Mo, ZUCF)
    
    # Total damage
    d = L / np.sqrt(1 - sH**2 + beta**2*sH**2) * d_NB

    return d

def Han_Ma(k, C, v0, Mo_w, Mo_l, v0_l, v0_w):
    '''
    Function that returns the fatigue damage according to the Han-Ma approach
    [A practical method for combination of fatigue damage subjected to 
    low–frequency and high–frequency Gaussian random processes
    doi = {10.1016/j.apor.2016.08.007}]
    '''   
    # LF Narrowband damage
    d_NB_lf = Narrowband(k, C, Mo_l, v0_l)
    
    # HF Narrowband damage
    d_NB_hf = Narrowband(k, C, Mo_w, v0_w)
    
    # Damage combination
    d = (d_NB_lf**(2/k) + d_NB_hf**(2/k))**(k/2)
    
    return d

def Gao_Moan(k, C, v0, Mo_h, Mo_l, v0_l, v0_h, Mo_m = 0, v0_m = 0, signal = 'Trimodal'):
    
    if signal == 'Bimodal':
        d = Jiao_Moan(k, v0, Mo_h, Mo_l, v0_l, v0_h, True, C)
        return d
    else:
        # Damage due to the HF process:
        d_h = Narrowband(k, C, Mo_h, v0_h)
        
        # Damage due to the MF process (HF + MF components):
        delta_h = np.sqrt(1 - Mo_h[1]**2 / (Mo_h[0] * Mo_h[2]))  # bandwidth parameter that may taken equal to 0.1
        v0_p = Mo_m[0] * v0_m * np.sqrt(1 + Mo_h[0] / Mo_m[0] * (v0_h / v0_m * delta_h)**2)
        pdf_P = P_pdf(Mo_m, Mo_h)
        I_p = integrate.quad(lambda x:  x**k * pdf_P(x), 0, np.inf)[0]    
        d_p = 2**k * v0_p * I_p / C
        
        # Damage due to the LF process (HF + MF + LF components):
        delta_m = np.sqrt(1 - Mo_m[1]**2 / (Mo_m[0] * Mo_m[2]))
        v0_q = np.sqrt(Mo_h[2]*delta_h**2 + Mo_m[2]*delta_m**2 + Mo_l[2]) \
            * (2*np.sqrt(Mo_l[0]*(Mo_h[0]+Mo_m[0]+Mo_l[0])) - np.pi*np.sqrt(Mo_h[0]*Mo_m[0]) \
               + 2*np.sqrt(Mo_h[0]*Mo_m[0])*np.arctan(np.sqrt(Mo_h[0]*Mo_m[0]/Mo_l[0]) \
                                                      /np.sqrt(Mo_h[0]+Mo_m[0]+Mo_l[0]))) \
                / (4*np.pi*(np.sqrt(Mo_h[0]+Mo_m[0]+Mo_l[0]))**3)
        lf_rayleigh = lambda x: st.rayleigh.pdf(x/np.sqrt(Mo_l[0])) / np.sqrt(Mo_l[0])
        pdf_Q = lambda x: np.convolve(pdf_P, lf_rayleigh)
        I_q = integrate.quad(lambda x:  x**k * pdf_Q(x), 0, np.inf)[0] 
        d_q = 2**k * v0_q * I_q / C
        return d_h + d_p + d_q


def Double_frequency_coupled(k, C, Mo_w, Mo_l, v0_w, v0_l, Pxx_w, f_w):
    """
    Function to calculate double frequency coupled damage.

    Input Parameters:
    - k: Slope of the S-N curve. Taken form DNV specification (k=3)
    - C: Intercept parameter of the S-N curve. Taken from DNV specification (C=1.2E11)
    - The S-N curve is defined as N = C*S^(-k) where S is the stress range
    - Mo_w: List with spectral moments (from 0 to 4th) of the wave frequency (high frequency) component
    - Mo_l: List with spectral moments (from 0 to 4th) of the low frequency component
    - v0_w: Mean zero up crossing frequency of the wave frequency component
    - v0_l: Mean zero up crossing frequency of the low frequency component
    - Pxx_w: Power spectral density of the wave frequency component
    - f_w: Frequency range of the wave frequency component
    """
   
    # Damage due to small cycles:
        
    def pdf_rs(Rs, sigma_delta, sigma_HF):
        prefactor = np.sqrt(2) / (np.sqrt(np.pi) * sigma_delta * sigma_HF**2)
        exponent = - Rs**2 / (2 * sigma_HF**2)
        
        def integrand(y, Rs, sigma_delta, sigma_HF):
            term1 = np.exp(-(1 / (2 * sigma_delta**2) + 1 / (2 * sigma_HF**2)) \
                            * y**2 - 1 / (sigma_HF**2) * y * Rs)
            return (y + Rs) * term1

        pdf = integrate.quad(integrand, 0, Rs, args=(Rs, sigma_delta, sigma_HF))[0] \
            * prefactor * np.exp(exponent)
        return pdf
    
    sigma_HF = np.sqrt(Mo_w[0])
    mu_hf = v0_w
    sigma_delta = np.sqrt(2*Mo_l[0] - 2 * integrate.simps(Pxx_w * np.cos(f_w * mu_hf/2), f_w))
    
    small_integral = integrate.quad(lambda Rs:  Rs**k \
                                    * pdf_rs(Rs, sigma_delta, sigma_HF), 0, np.inf)[0]
    
    d_small = 2**k * (v0_w - v0_l) / C * small_integral
    
    
    # Damage due to large cycles:
        
    def pdf_rl(RL, sigma_delta, sigma_HF, sigma_LF):
        prefactor = np.sqrt(2) / (np.sqrt(np.pi) * sigma_delta * sigma_HF**2 * sigma_LF**2)
        exponent = - RL**2 / (2 * sigma_LF**2)
        
        def integrand(z, y, RL, sigma_LF, sigma_HF, sigma_delta):
            term1 = (RL - z) * (y + z)
            term2 = np.exp((-(1 / (2 * sigma_LF**2) + 1 / (2 * sigma_HF**2)) * z**2) \
                            + (1 / sigma_LF**2) * RL * z - ((1 / (2 * sigma_delta**2)) \
                            + (1 / (2 * sigma_HF**2))) * y**2 - (1 / (sigma_HF**2)) * y * RL)

            return term1 * term2
    

        pdf = integrate.dblquad(integrand, 0, RL, 0, lambda z: z, 
                                args=(RL, sigma_LF, sigma_HF, sigma_delta))[0] \
            * prefactor * np.exp(exponent)            
        return  pdf
    
    sigma_LF = np.sqrt(Mo_l[0])
    
    large_integral = integrate.quad(lambda Rl:  Rl**k * pdf_rl(Rl, sigma_delta,
                                                                sigma_HF, sigma_LF), 0, np.inf)[0]
    d_large = 2**k * v0_l / C * large_integral

    # Total damage
    d = d_small + d_large
    return d


###############################################################################

# d)	Combined fatigue damage – narrowband damage combination


def Lotsberg(k, C, Mo_w, Mo_l, ZUCF_w, ZUCF_l):
    D_NB_HF = Narrowband(k, C, Mo_w, ZUCF_w)
    D_NB_LF = Narrowband(k, C, Mo_l, ZUCF_l)
    d = D_NB_HF * (1 - ZUCF_l / ZUCF_w) + ZUCF_l * \
        ((D_NB_HF / ZUCF_w)**(1 / k) + (D_NB_LF / ZUCF_l)**(1 / k))**k
    return d


def Huang_Moan(k, C, Mo_w, Mo_l, ZUCF_w, ZUCF_l):
    D_NB_HF = Narrowband(k, C, Mo_w, ZUCF_w)
    D_NB_LF = Narrowband(k, C, Mo_l, ZUCF_l)
    d = ((D_NB_HF / ZUCF_w)**(2 / k) + (D_NB_LF / ZUCF_l)**(2 / k))**((k - 2) / 2) \
        * (ZUCF_w**2 * (D_NB_HF / ZUCF_w)**(2 / k) + ZUCF_l**2 \
           * (D_NB_LF / ZUCF_l)**(2 / k))**(3 / 2) / (ZUCF_w**4 * (D_NB_HF / ZUCF_w)**(2 / k) \
           + ZUCF_l**4 * (D_NB_LF / ZUCF_l)**(2 / k))**(1 / 2)
    return d


def SingleMoment(k, C, f, Pxx):
    m_2k = per_spectral_mom(f, Pxx, n = [2/k])[0]
    d = 2**(k/2)/(2*np.pi*C)*2**k*m_2k**(k/2)*gamma_func(1+k/2)
    return d


def BandsMethod(k, C, tension, fs, nbands, cor = False):
    beta = -1/k; alfa = 1/2*C**(-beta); K = K_func(k);
    # Compute the PSD using Welch's method
    f, Pxx = sig.welch(tension, fs=fs, nperseg=100 * nbands-1)
    f_ref = f[len(f)//2]
    if cor:
        Sm  = tension.sum() / len(tension)
        Sa = (max(tension)-min(tension))/2
        Sar = np.sqrt(Sa*(Sm+Sa))
        K_swt = Sar / Sa
        Pxx = K_swt**2 * Pxx
    f_area = len(f)//nbands
    j = 0; m0 = 0
    for i in range(1,nbands+1): 
        f_i = f[j:j+f_area]; Pxx_i = Pxx[j:j+f_area]
        m0_i = per_spectral_mom(f_i, Pxx_i, n = [0])[0]
        m0_i_ref = (f_ref/f_i[len(f_i)//2])**(2*beta)*m0_i
        m0 += m0_i_ref
        j += f_area
        
    d = (f_ref * K * (2*alfa)**(1/beta)) / ((2*np.sqrt(m0))**(1/beta))
    return d



def Gao_Zheng(k, C, f_l, Pxx_l, f_w, Pxx_w, v0_w, v0_l, v0):
    def get_a_values(P, gamma):
        if gamma >= 2 and gamma <= 4:
            a_values = {
                'P1': (-14.5495, -13.6189, -1.7305),
                'P2': (46.3878, 27.1178, 2.5926),
                'P3': (-10.851, 7.0653, -0.0151),
                'P4': (-14.2695, -13.4855, -0.7394),
                'P5': (0.1915, -0.0838, -0.0002),
                'P6': (13.335, -9.663, 0.1395),
                'P7': (203.4813, -3.2088, 6.4657),
                'P8': (-0.0486, -11.2238, 3.1149),
                'P9': (20.8779, -18.3121, -1.0359),
                'P10': (10.325, 1.2937, 0.7107),
                'P11': (56.3776, 7.792, -2.2805)
            }
        elif gamma > 4:
            a_values = {
                'P1': (0.1312, -0.0105, -0.0031),
                'P2': (-0.1205, 0.0166, 0.0052),
                'P3': (-0.0132, -0.0384, 0.0086),
                'P4': (0.0442, -0.0133, -0.001),
                'P5': (-0.01, 0.0095, -0.0008),
                'P6': (-0.003, 0.0265, -0.0067),
                'P7': (-0.6535, -0.1117, 0.0142),
                'P8': (-0.1837, -0.0632, 0.0055),
                'P9': (0.2871, -0.008, 0.0034),
                'P10': (-0.0419, 0.0227, 0.0008),
                'P11': (0.0393, 0.0577, -0.0033)
            }
        else:
            print('The xi coefficient is not empirically fitted for the given gamma value')
            return (0,0,0)
        return a_values.get(P)

    def xi_function(gamma, beta, k):
        P = []
        for n in range(1, 12):
            a0, a1, a2 = get_a_values(f'P{n}', gamma)
            P.append(a0 + a1 * k + a2 * k**2)

        log_gamma, log_beta = np.log(gamma), np.log(beta)
        xi = (np.dot(P[:6], [1, log_gamma, log_beta, log_gamma**2, log_beta**2, log_gamma*log_beta]) /
              (1 + np.dot(P[6:], [log_gamma, log_beta, log_gamma**2, log_beta**2, log_gamma*log_beta])))
        return xi

    def spectralmom_sum(k, Pxx, f):
        df = np.diff(f, prepend=f[0])
        return np.sum(f[1:]**(2/k) * Pxx[1:] * df[1:])

    def combined_spectralmom_sum(k, Pxx_1, f_1, Pxx_2, f_2, v0_1, v0_2):
        beta = integrate.simps(Pxx_2, f_2) / integrate.simps(Pxx_1, f_1)
        gamma = v0_1 / v0_2
        xi = xi_function(gamma, beta, k)
        df_1, df_2 = np.diff(f_1, prepend=f_1[0]), np.diff(f_2, prepend=f_2[0])
        lamba_2_k = np.sum(xi * np.sqrt((f_1[1:]**(2/k) * Pxx_1[1:] * df_1[1:]) *
                                        (f_2[1:]**(2/k) * Pxx_2[1:] * df_2[1:])))
        return lamba_2_k

    lamba_2_k_lf = spectralmom_sum(k, Pxx_l, f_l)
    lamba_2_k_hf = spectralmom_sum(k, Pxx_w, f_w)
    lamba_2_k_lf_hf = combined_spectralmom_sum(k, Pxx_l, f_l, Pxx_w, f_w, v0_w, v0_l)

    d = (2*np.sqrt(2))**k / C * gamma_func(1 + k/2) * (lamba_2_k_lf + lamba_2_k_hf + lamba_2_k_lf_hf)**(k/2)

    return d



def ImprovedBandsMethod(k, C, tension_low, tension_high, fs, nbands, Mo_w, Mo_l):
    # Compute the PSD using Welch's method
    f_l, Pxx_l = sig.welch(tension_low, fs=fs, nperseg=100 * nbands - 1)
    f_h, Pxx_h = sig.welch(tension_high, fs=fs, nperseg=100 * nbands - 1)
    f_l_r, f_h_r = f_l[len(f_l)//2], f_h[len(f_h)//2]
   
    d_w = np.sqrt(1 - Mo_w[1]**2 / (Mo_w[0] * Mo_w[2]))  # Vanmarcke bandwidth parameter
    
    f_area = len(f_l) // nbands
    m0_r_lf, m0_r_hf = 0, 0
    for j in range(0, len(f_l), f_area):
        f_i_l, Pxx_i_l = f_l[j:j+f_area], Pxx_l[j:j+f_area]
        m0_l_i = per_spectral_mom(f_i_l, Pxx_i_l, n=[0])[0]
        m0_l_i_ref = (f_l_r / f_i_l[len(f_i_l)//2])**(-2/k) * m0_l_i
        m0_r_lf += m0_l_i_ref
        
        f_i_h, Pxx_i_h = f_h[j:j+f_area], Pxx_h[j:j+f_area]
        m0_h_i = per_spectral_mom(f_i_h, Pxx_i_h, n=[0])[0]
        m0_h_i_ref = (f_h_r / f_i_h[len(f_i_h)//2])**(-2/k) * m0_h_i
        m0_r_hf += m0_h_i_ref

    
    def get_P_values_for_k(k, vanmarcke_hf):
        
        if vanmarcke_hf <= 0.1:
            P_values = {
                'P1': {3: 1.118, 4: 1.112, 5: 1.108},
                'P2': {3: -0.5818, 4: -0.6308, 5: -0.6301},
                'P3': {3: 0.1947, 4: 0.1213, 5: 0.06409},
                'P4': {3: 0.1369, 4: 0.1372, 5: 0.1274},
                'P5': {3: 0.05659, 4: 0.05294, 5: 0.0442},
                'P6': {3: 0.03104, 4: 0.03733, 5: 0.03809},
                'P7': {3: -0.4979, 4: -0.5555, 5: -0.5622},
                'P8': {3: 0.2341, 4: 0.1689, 5: 0.1178},
                'P9': {3: 0.1242, 4: 0.1282, 5: 0.121},
                'P10': {3: 0.0525, 4: 0.04615, 5: 0.03535},
                'P11': {3: 0.01831, 4: 0.02246, 5: 0.02295}
            }
        else:
            P_values = {
                'P1': {3: 1.021, 4: 0.9841, 5: 0.927},
                'P2': {3: -0.289, 4: -0.2251, 5: -0.09982},
                'P3': {3: 0.1204, 4: 0.1132, 5: 0.1266},
                'P4': {3: 0.1964, 4: 0.1524, 5: 0.1536},
                'P5': {3: 0.07312, 4: 0.08618, 5: 0.1189},
                'P6': {3: 0.1692, 4: 0.1541, 5: 0.1874},
                'P7': {3: -0.3553, 4: -0.3301, 5: -0.2593},
                'P8': {3: 0.1524, 4: 0.1477, 5: 0.1458},
                'P9': {3: 0.2125, 4: 0.1759, 5: 0.1813},
                'P10': {3: 0.06772, 4: 0.07799, 5: 0.1097},
                'P11': {3: 0.1667, 4: 0.1591, 5: 0.2111}
            }
        
        return [P_values[f'P{i}'][k] for i in range(1, 12)]
    
    P = get_P_values_for_k(k, d_w)
    gamma, beta = f_h_r / f_l_r, Mo_w[0] / Mo_l[0]
    
    log_gamma, log_beta = np.log(gamma), np.log(beta)
    mu_numerator = P[0] + P[1]*log_gamma + P[2]*log_beta + P[3]*log_gamma**2 + P[4]*log_beta**2 + P[5]*log_gamma*log_beta
    mu_denominator = 1 + P[6]*log_gamma + P[7]*log_beta + P[8]*log_gamma**2 + P[9]*log_beta**2 + P[10]*log_gamma*log_beta
    mu = mu_numerator / mu_denominator
    
    m0_total = m0_r_lf + gamma**(2/k) * m0_r_hf * mu
    d = f_l_r / C * (2 * np.sqrt(2 * m0_total))**k * gamma_func(k / 2 + 1)
    return d


