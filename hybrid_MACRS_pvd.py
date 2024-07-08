# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 2024

@author: lpmclau
"""
import numpy_financial as npf
def hybrid_MACRS_pvd(input_year, WACC_n):
    """
    This model calculates the MACRS Present Value of Depreciation (PVD) for a specified input year.

    Parameters:
    - input_year (int): The depreciation period (Yrs).
    - WACC_n (float): The Weighted Average Cost of Capital, nominal rate (frac)

    Returns:
    - PVD (float): The PVD for the specified input year (frac)
    
    Dependencies:
    - numpy_financial for financial functions
    """
    macrs_rates = {
    3: [0.3333, 0.4445, 0.1481, 0.0741],  # 3-year property
    5: [0.2000, 0.3200, 0.1920, 0.1152, 0.1152, 0.0576],  # 5-year property
    7: [0.1429, 0.2449, 0.1749, 0.1249, 0.0893, 0.0892, 0.0893, 0.0446]  # 7-year property
}

    if input_year in macrs_rates:
        macrs_schedule = macrs_rates[input_year]
        return sum(rate * npf.pv(rate=WACC_n, nper=year, pmt=0, fv=-1) for year, rate in enumerate(macrs_schedule, start=1))
