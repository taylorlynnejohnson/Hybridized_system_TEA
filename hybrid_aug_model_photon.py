# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 2024

@author: lpmclau
"""
import numpy as np
def hybrid_aug_model_photon(years):

    """
    This model determines component augmentation schedules and costs. First, the model determines the energy capacity that must 
    augment the battery field on an annual basis to maintain a fixed dischargeable energy output. Electric particle heater replacment years
    are specified. The model then calculates the augmentation costs for batteries and heaters over time. Additional augmentation items can
    be added following the annualized approach. 
    
    The battery augmentation strategy is an adaption of the battery degradation rates and replacment schedule presented in the
    2022 Grid Energy Storage Technology Cost and Performance Assessment.
    
    Parameters:
    - dod_i (float): Initial depth of discharge (0 < dod_i <= 1)
    - dod_f (float): Final depth of discharge after augmentation (0 < dod_f <= 1)
    - sys_cap_mwh (float): Battery system capacity in MWh (MWh)
    - years (int): Number of project years (not analysis) for calculation (Yrs)
    - pr (float): Power rating for the batteries (MW)
    - rte (float): Battery round-trip efficiency (0 < rte <= 1)
    - n60, n80 (int): Number of cycles to 60% and 80% depth of discharge (# cycles)
    - cost_rod, cost_ctrl, cost_shell (float): Costs of rod, controller, and shell components ($/kW)
    - prh (float): Power rating for the heater (MW)

    Returns:
    - cpx_aug_annual (np.array): Annual augmentation costs for batteries ($)
    - CPX_yr1_battery_USD (float): Year 1 cost for the battery system ($)
    - cpx_heater_aug_annual (np.array): Annual augmentation costs for heaters ($)
    - CPX_yr1_heater_USD (float): Year 1 cost for heater components ($)
    
    Dependencies:
    - numpy for numerical calculations
    """
    prh = 0
    
    cpx_heater_aug_annual = np.zeros(years)
    for year in range(years):
        cpx_heater_aug_annual[year] = 0 if year in [10, 20] else 0
        
    return cpx_heater_aug_annual