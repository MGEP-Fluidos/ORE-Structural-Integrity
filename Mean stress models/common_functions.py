import numpy as np

def Properties(grade, size):
    Z = size**2 * (44 - 0.08 * size)
    grades = {
        "R5": {"Su": 1000, "Sy": 760, "MBL": 0.032 * Z},
        "R4S": {"Su": 960, "Sy": 700, "MBL": 0.0304 * Z},
        "R4": {"Su": 860, "Sy": 580, "MBL": 0.0274 * Z},
        "R3S": {"Su": 770, "Sy": 490, "MBL": 0.0249 * Z},
        "R3": {"Su": 690, "Sy": 410, "MBL": 0.0223 * Z},
    }
    return grades.get(grade, {"Su": None, "Sy": None, "MBL": None})


def Smith_Watson_Topper(Sa, Smax):
    return np.sqrt(Sa*Smax) / Sa

def Walker(Sa, Smax, gamma):
    return Sa**(gamma)*Smax**(1-gamma) / Sa

def Goodman(Sm, Su):
    return 1 / (1 - Sm/Su)

def Gerber(Sm, Su):
    return 1 / (1 - (Sm/Su)**2)

def Soderberg(Sm, Sy):
    return 1 / (1 - Sm/Sy)

def Morrow(Sm, Sf):
    # The Morrow equation was originally developed to use with Sf,
    # the monotonic tension test value of true fracture stress.
    # However, some authors have replaced it by the fatigue strength coefficient.
    return 1 / (1 - Sm/Sf)

def Kwofie(Sm, Su, alphak):
    return 1 / (np.exp(-alphak*(Sm/Su)))


def para_DNV(component):
    '''
    Function that gives the ad, m and gamma parameters according to DNV rule
    for the selected chain type depending on the nominal diameter
    '''
    if component == "stud chain":
        ad = 1.2E11
        m = 3.0
        gamma = 1.3293
    elif component == "studless chain":
        ad = 6E10
        m = 3.0
        gamma = 1.3293
    elif component == "six strand rope":
        ad = 3.4E14
        m = 4
        gamma = 2
    elif component == "spiral strand rope":
        ad = 1.7E17
        m = 4.8
        gamma = 2.9812
    else:
        raise ValueError("Invalid component type")
    return m, gamma, ad