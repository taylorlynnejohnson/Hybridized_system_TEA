import numpy as np
import pandas as pd
import numpy_financial as npf
from hybrid_aug_model_photon import hybrid_aug_model_photon
from hybrid_MACRS_pvd import hybrid_MACRS_pvd


def hybrid_tea_model_photon(row, grid_power_required, grid_contribution_to_ice_annual, gridtied, load_met_percentage, PV_AC_Rating, tes_capacity, maximum_hourly_tes_output, collector_area, PV_cc, TES_cc, PB_cc, CSP_cc, C_OM_TES, C_OM_PV, C_OM_CSP, C_OM_PB, n, L, esc, DF, I, tax, COE, inflation, ITC, property_tax, insurance, MACRS_yrs, CAPEX):
    C_grid = 0.03 # grid cost in $/kWh
    
    if gridtied: 
        grid_power_required = grid_power_required + grid_contribution_to_ice_annual
    else:
        grid_power_required = grid_contribution_to_ice_annual

    # TEA calculation values: WACC, CRF, FCR, PVD
    WACC_n = DF*I*(1-tax) + (1-DF)*COE # use in LCOE calcs for "nominal" LCOE
    WACC_r = ((1+WACC_n)/(1+inflation))-1 # use in LCOE calcs for "real" LCOE
    
    PVD = hybrid_MACRS_pvd(MACRS_yrs, WACC_n)
    CRF = WACC_r/(1-(1+WACC_r)**(-n))
    FCR =((CRF* ((1 -(tax* PVD)*(1-ITC/2) - ITC)) ) + property_tax + insurance )/ (1 - tax)
    
    # O&M costs
    annual_OM_cost = C_OM_TES*row['TES_capacity']/1000 + C_OM_PV*row['PV_AC_Rating']/1000 + C_OM_CSP*row['Collector_area'] + C_OM_PB*row['P_block']/1000
    
    # augmentation schedule 
    aug_htr = hybrid_aug_model_photon(L, TES_cc, PB_cc,CSP_cc)
    
    # correct for NPV, include augmentations 
    OM_NPV_arr=[]
    annual_renewables_NPV_arr=[]
    aug_htr_NPV_arr=[]
    grid_NPV_arr=[]   
    for v in range(L):
        v = v + 1
        OM_esc = annual_OM_cost * ((1 + esc)**(v-1))
        val = OM_esc / ((1 + WACC_r)**v)
        OM_NPV_arr.append(val)
        val = row['PO'] / ((1 + WACC_r)**v)  
        annual_renewables_NPV_arr.append(val)
        val = aug_htr[v-1] / ((1 + WACC_r)**v)
        aug_htr_NPV_arr.append(val)
        grid_val = C_grid * np.sum(grid_power_required) / ((1 + WACC_r)**v)  # Annualized grid cost
        grid_NPV_arr.append(grid_val)
        
        sum_NPV_htr_ARMO_N = np.sum(aug_htr_NPV_arr[0:n])
        sum_NPV_OM_N = np.sum(OM_NPV_arr[0:n])
        sum_NPV_renewables_N = np.sum(annual_renewables_NPV_arr[0:n])
        sum_NPV_grid_N = np.sum(grid_NPV_arr[0:n])
        
        sum_NPV_htr_ARMO_L = np.sum(aug_htr_NPV_arr)
        sum_NPV_OM_L = np.sum(OM_NPV_arr)
        sum_NPV_renewables_L = np.sum(annual_renewables_NPV_arr)
        sum_NPV_grid_L = np.sum(grid_NPV_arr)

    # annualized CAPEX and O&M costs
    annualized_CAPEX = row['CAPEX'] * FCR
    annualized_OM = sum_NPV_OM_N * CRF  # [$/yr] OM Costs
    annualized_ARMO = (sum_NPV_htr_ARMO_N) * CRF # [$/yr] Augmentation & Replacement Costs
    annualized_grid_cost = sum_NPV_grid_N * CRF  # [$/yr] Grid Cost
    annual_ARR = annualized_CAPEX + annualized_OM + annualized_ARMO + annualized_grid_cost  # Total ARR over project
    
    # Calculate residual value for longer project life compared to analysis period (e.g. = 0 for n=L)
    Rv = ((((1 + WACC_r) ** n) * ((1 - sum_NPV_renewables_N / sum_NPV_renewables_L) * (row['CAPEX'] * (1 - (tax * PVD) * (1 - ITC / 2) - ITC)) + (sum_NPV_OM_N + sum_NPV_htr_ARMO_N) - (sum_NPV_renewables_N / sum_NPV_renewables_L) * (sum_NPV_OM_L + sum_NPV_htr_ARMO_L)))) / ((1 + WACC_r) ** n)
    
    cash_flow = np.ones_like(range(n)) * annual_ARR  # Cash flow schedule in analysis period
    
    NPV_ARR = npf.npv(WACC_r, cash_flow)  # [$] NPV of ARR over analysis period 

    LCOE_real_USD_kWh = (NPV_ARR - Rv) / sum_NPV_renewables_N  # [$/kWh]  
    LCOE_real_USD_kWh = 0.5 if sum_NPV_renewables_N == 0 else LCOE_real_USD_kWh
    
    return pd.Series({
        'LCOE': LCOE_real_USD_kWh, 
        'OM': annual_OM_cost, 
        'C_OM_TES': C_OM_TES*row['TES_capacity']/1000, 
        'C_OM_PV': C_OM_PV*row['PV_AC_Rating']/1000, 
        'C_OM_CSP' : C_OM_CSP*row['Collector_area'], 
        'C_OM_PB': C_OM_PB*row['P_block']/1000, 
        'CAPEX': row['CAPEX'],
        'NPV' : NPV_ARR,
        'total_grid_cost': sum_NPV_grid_N 
    })




