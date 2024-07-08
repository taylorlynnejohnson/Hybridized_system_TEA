# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 2024

@author: lpmclau
"""
import numpy as np
import pandas as pd
import numpy_financial as npf
from hybrid_aug_model_photon import hybrid_aug_model_photon
from hybrid_MACRS_pvd import hybrid_MACRS_pvd

def hybrid_tea_model_photon(row,C_OM_TES,C_OM_PV,C_OM_CSP,n,L,esc,DF,I,tax,COE,inflation,ITC,property_tax,insurance,MACRS_yrs,CAPEX):
    """ 
    This model performs a comprehensive Techno-Economic Analysis (TEA) for a hybrid renewable energy system. This function 
    integrates with an augmentation model to evaluate the lifetime cost of various renewable energy assets, including solar PV,
    CSP, TES, and battery storage.The methodology for the LCOE calculation provided here is predominantly adapted from Short et al. [1]. The LCOE
    is determined as the average $/kWh value that energy discharged from the energy system must be sold at to recover total project 
    revenue requirements over the analysis period. The LCOE is setup to be calculated as the "real" LCOE by default. 
    
    [1] W. Short, D. J. Packey, and T. Holt, "A Manual for the Economic Evaluation of Energy Efficiency and Renewable Energy Technologies,"
    National Renewable Energy Laboratory, Golden, Colorado, 1995. [Online]. Available: https://www.nrel.gov/docs/legosti/old/5173.pdf

    Parameters:
    
    - row (pd.Series): Data row containing parameters for the calculation, including system capacities, power ratings, and fractions of different components.
    - C_OM_TES, C_OM_Battery, C_OM_PV, C_OM_CSP (float): O&M costs for TES, Battery, PV, and CSP, respectively ($/kw-year)
    - n (int): Analysis period (Yrs)
    - L (int): Operational life (Yrs)
    - esc (float): Escalation rate for O&M costs (frac)
    - DF (float): Debt fraction (0 < DF <= 1)
    - I (float): Interest rate for the debt of the project (frac)
    - tax (float): Tax rate (frac)
    - COE (float): Cost of Equity (frac)
    - inflation (float): Inflation rate (frac)
    - ITC (float): Investment Tax Credit rate (0 < ITC <= 1)
    - property_tax, insurance (float): Rates for property tax and insurance (frac)
    - MACRS_yrs (float): MACRS depreciation period to determine PVD
    - DOD_i, DOD_f (float): Initial and final depth of discharge for batteries (0 < DOD_x <= 1)
    - batt_RTE (float): Battery Round-Trip Efficiency (0 < RTE <= 1)
    - N_60, N_80 (int): Number of cycles to 60% and 80% depth of discharge (#)
    - C_rod, C_controller, C_shell (float): Costs of rod, controller, and shell components for heaters ($/kW)
    - sb_batt (float): Storabe block augmentation cost ($/kWh/MW)
    
    Returns: 
        
        - A pandas Series containing calculated financial metrics and costs, including:            
            - LCOE_real_USD_kWh (float): Levelized cost of electricity on real basis ($/kWh)
            - CPX_yr1_battery_USD (float): initial costs for the battery ($)
            - CPX_yr1_heater_USD (float): initial costs of the particle heater (shell, rods, controller) ($)
    
    Dependencies:
    - numpy for numerical calculations 
    - numpy_financial for financial functions
    - pandas for data manipulation
    - aug_model for augmentation cost and schedule 
    - MACRS_pvd for PVD determination
            
    Assumptions:
        - Financial inputs are generic inputs following [1]
        - Energy production is assumed to occur in year 1 (no construction delay)
        """

    # Setup TEA calculation values: WACC, CRF, FCR, PVD
    WACC_n = DF*I*(1-tax) + (1-DF)*COE # use in LCOE calcs for "nominal" LCOE
    WACC_r = ((1+WACC_n)/(1+inflation))-1 # use in LCOE calcs for "real" LCOE
    PVD = hybrid_MACRS_pvd(MACRS_yrs, WACC_n)
    CRF = WACC_r/(1-(1+WACC_r)**(-n))
    FCR =((CRF* ((1 -(tax* PVD)*(1-ITC/2) - ITC)) ) + property_tax + insurance )/ (1 - tax)
    
    # Setup Proper O&M Cost based on active systems in [$] and proper heater cost assignment (yes or no)
    annual_OM_cost = C_OM_TES*row['P_block'] + C_OM_PV*row['PV_AC_Rating'] + C_OM_CSP*row['P_block']
    annual_OM_cost = 0
    
    
    # Determine augmentation schedule and costs 
    aug_htr = hybrid_aug_model_photon(L) 
    
    # Correct for Net Present Value and Include Augmentations 
    OM_NPV_arr=[]
    annual_renewables_NPV_arr=[]
    aug_htr_NPV_arr=[]
    for v in range(L):
        v=v+1
        OM_esc = annual_OM_cost * ((1 + esc)**(v-1))
        val = OM_esc/((1+WACC_r)**v)
        OM_NPV_arr.append(val)
        val = row['PO']/((1+WACC_r)**v)
        annual_renewables_NPV_arr.append(val)
        val = aug_htr[v-1]/((1+WACC_r)**v)
        aug_htr_NPV_arr.append(val)
        
        sum_NPV_htr_ARMO_N = np.sum(aug_htr_NPV_arr[0:n])
        sum_NPV_OM_N = np.sum(OM_NPV_arr[0:n])
        sum_NPV_renewables_N = np.sum(annual_renewables_NPV_arr[0:n])
        
        sum_NPV_htr_ARMO_L = np.sum(aug_htr_NPV_arr)
        sum_NPV_OM_L = np.sum(OM_NPV_arr)
        sum_NPV_renewables_L = np.sum(annual_renewables_NPV_arr)

    annualized_CAPEX = row['CAPEX'] * FCR
    annualized_OM = sum_NPV_OM_N*CRF  # [$/yr] Annualized OM Costs
    annualized_ARMO = (sum_NPV_htr_ARMO_N)*CRF # [$/yr] Annualized Augmentation & Replacement Costs
    annual_ARR = annualized_CAPEX + annualized_OM + annualized_ARMO # [$] Total ARR over project 
    
    # Calculate residual value for longer project life compared to analysis period (e.g. = 0 for n=L)
    Rv=((((1+WACC_r)**n)*((1-sum_NPV_renewables_N/sum_NPV_renewables_L)*(row['CAPEX'] *(1 -(tax*PVD)*(1-ITC/2) - ITC))+(sum_NPV_OM_N+sum_NPV_htr_ARMO_N)-(sum_NPV_renewables_N/sum_NPV_renewables_L)*(sum_NPV_OM_L+sum_NPV_htr_ARMO_L))))/((1+WACC_r)**n)
    
    cash_flow = np.ones_like(range(n))*annual_ARR  # Cash flow schedule in Analysis Period
    NPV_ARR = npf.npv(WACC_r,cash_flow)  # [$] NPV of ARR over Analysis Period 
    
    LCOE_real_USD_kWh = (NPV_ARR-Rv)/(sum_NPV_renewables_N) # [$/kWhe] Levelized cost of power from system (e or t basis depends on discharge eff)
    
    return pd.Series([LCOE_real_USD_kWh,annual_OM_cost])