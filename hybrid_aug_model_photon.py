# -*- coding: utf-8 -*-

import numpy as np

def hybrid_aug_model_photon(years, TES_cc, PB_cc,CSP_cc):
    aug_costs = np.zeros(years)
    
    collector_replacement_cost = 0.65*CSP_cc
    turbo_machinery_cost = PB_cc
    tes_media_cost = TES_cc

    turbo_machinery_lifespan = 40
    collector_lifespan = 20
    tes_media_lifespan = 30
    
    # schedule replacements
    for year in range(years):
        if (year + 1) % collector_lifespan == 0:  
            aug_costs[year] += collector_replacement_cost
        if (year + 1) % turbo_machinery_lifespan == 0: 
            aug_costs[year] += turbo_machinery_cost
        if (year + 1) % tes_media_lifespan == 0: 
            aug_costs[year] += tes_media_cost
    
    return aug_costs
