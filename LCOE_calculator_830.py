#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Photon Vault TEA Model: calculates LCOE for a hybridized PV/CSP/TES system, determining tilt + PV/CSP/TES capacities by minimizing LCOE and maximizing RES 


# In[ ]:

import time
import pyNSRDB
import geopy
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import requests
import pytz
from timezonefinder import TimezoneFinder
import pvlib
from pvlib import pvsystem, location, temperature, modelchain, solarposition
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from hybrid_tea_model_photon import hybrid_tea_model_photon
import warnings
from scipy.interpolate import interp1d
import math
from scipy.optimize import minimize
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.callbacks import CheckpointSaver
import plotly.express as px
import plotly.graph_objects as go
from itertools import product
import seaborn as sns
from matplotlib.colors import Normalize
warnings.filterwarnings('ignore')
from plotly.subplots import make_subplots
"""from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup"""

################################################################# Retrieve TMY data ###################################################################

#%% System Constants and Parameters 


class solar_data_fetcher:
    def __init__(self, plant_location="Page, Arizona", api_key="2owE0tA0Bs6DTaTM5Y9gRPUHZX8iOjpEqRjukHyX", interpolating=False):
        self.plant_location = plant_location
        self.api_key = api_key
        self.interpolating = interpolating
        
        if self.interpolating:
            self.no_ts = 4
            self.timeunit = '1/4 Hour'
            self.interval = 30
        else:
            self.no_ts = 1
            self.timeunit = 'Hour'
            self.interval = 60
            
    def get_location_coordinates(self):
        geolocator = Nominatim(user_agent="photonvault")
        try:
            location = geolocator.geocode(self.plant_location)
            return location.latitude, location.longitude
        except Exception as e:
            print(f"Error: Geocode failed for {self.plant_location}. Error message: {e}")
            return None, None
            
    def get_tmy_data(self, lat, lon, year = 2019):
        attributes = 'ghi,dhi,dni,wind_speed,air_temperature,solar_zenith_angle'
        leap_year = 'false'
        utc = 'true'
        your_name = 'NSTTF'
        reason_for_use = 'tea_analysis'
        your_affiliation = 'Sandia'
        your_email = 'johnsontaylor@ufl.edu'
        mailing_list = 'false'
        url = f'https://developer.nrel.gov/api/nsrdb/v2/solar/psm3-download.csv?wkt=POINT({lon}%20{lat})&names={year}&leap_day={leap_year}&interval={self.interval}&utc={utc}&full_name={your_name}&email={your_email}&affiliation={your_affiliation}&mailing_list={mailing_list}&reason={reason_for_use}&api_key={self.api_key}&attributes={attributes}'
        data = pd.read_csv(url, nrows=3)
        data = pd.read_csv(url, header=2)
        return data

    def get_timezone(self, latitude, longitude):
        tf = TimezoneFinder()
        timezone_str = tf.timezone_at(lat=latitude, lng=longitude)
        return timezone_str
    
    def fetch_solar_data(self, year=2019):
        latitude, longitude = 36.9140854, -111.4551159  # TO-DO: replace once you figure out firewall issue
        data = self.get_tmy_data(latitude, longitude, year)
        tz = self.get_timezone(latitude, longitude)
        period = 24*365*self.no_ts # Number of time steps for simulation
        if data is None or data.empty:
            return None
        else:
            theta_z = data['Solar Zenith Angle']                                                               
            amb_temp = data['Temperature']   
            return data, self.no_ts, self.timeunit, latitude, longitude, theta_z, amb_temp, tz, period 

"""class ElectricityData:
    def __init__(self, location, year):
        self.location = location
        self.year = year
        self.url = f"https://www.gridstatus.io/map?ref=blog.gridstatus.io&color=rt_lmp&zoom=3.70&center={self.location}"
        
        self.options = Options()
        self.options.headless = True
        self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=self.options)
    
    def fetch_data(self):
        self.driver.get(self.url)
        time.sleep(15)
        soup = BeautifulSoup(self.driver.page_source, 'html.parser')
        
        data = []
        elements = soup.find_all('div', class_='some-class-name')

        for elem in elements:
            data.append(elem.get_text())

        data_array = np.array(data)
        
        return data_array
    
    def close(self):
        self.driver.quit()"""


# In[ ]:
#######################################################################################################################################################################################################################

class pv_system:
    def __init__(self, latitude, longitude, tz, no_ts):
        self.latitude = latitude
        self.longitude = longitude
        self.no_ts = no_ts
        self.tz = tz
        self.sandia_modules = pvlib.pvsystem.retrieve_sam('SandiaMod')
        self.module_parameters = self.sandia_modules['Canadian_Solar_CS5P_220M___2009_']

    
    def preprocess_tmy_data(self, tmy_data):
        DHI,DNI,GHI = tmy_data['DHI'],tmy_data['DNI'], tmy_data['GHI']  
        year,month,day,hour,minute = tmy_data['Year'],tmy_data['Month'], tmy_data['Day'], tmy_data['Hour'],tmy_data['Minute'] 
        tmy_data = pd.DataFrame({
            'dhi': DHI,
            'dni': DNI,
            'ghi': GHI,
            'Year': year,
            'Month': month,
            'Day': day,
            'Hour': hour,
            'Minute': minute
        })
        
        tmy_data['DateTime'] = pd.to_datetime(tmy_data[['Year', 'Month', 'Day', 'Hour', 'Minute']])
        tmy_data.set_index('DateTime', inplace=True)
        tmy_data.drop(columns=['Year', 'Month', 'Day', 'Hour', 'Minute'], inplace=True)
        if not tmy_data.index.tz:
            tmy_data.index = tmy_data.index.tz_localize('UTC').tz_convert(self.tz)
        return tmy_data
    
    def run_pv_simulation(self, tmy_data,PV_AC_Rating,tilt):
        period = 24*365*self.no_ts
        inverter_parameters = {
            'Paco': PV_AC_Rating,  # AC power output [W]
            'Pdco': PV_AC_Rating * 2,  # DC power input [W]
            'Vdco': 480,  # DC voltage [V]
            'Pnt': 0.5,  # Night time power [W]
            'Pso': 0,  # Self-consumption power [W]
            'C0': -0.00025,
            'C1': 0.0001,
            'C2': 0.02,
            'C3': -0.04
        }
        
        site = pvlib.location.Location(self.latitude, self.longitude, self.tz)
        
        system = pvlib.pvsystem.PVSystem(surface_tilt=tilt, surface_azimuth=180,
                                              module_parameters=self.module_parameters,
                                              modules_per_string=1,
                                              inverter_parameters=inverter_parameters,
                                              temperature_model_parameters=pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass'])
        mc = pvlib.modelchain.ModelChain(system, site, aoi_model='physical', spectral_model='no_loss', ac_model='sandia')
        tmy_data_ = self.preprocess_tmy_data(tmy_data)
        
        dni_data = tmy_data_['dni'][:period]  # Direct Normal Irradiance data [W/m^2]
        
        mc.run_model(tmy_data_)
        hourly_pv_output = mc.results.ac /1000 # Hourly PV output [kW]
        hourly_pv_output = hourly_pv_output[:period]
        hourly_pv_output = np.zeros(period) if PV_AC_Rating == 0 else hourly_pv_output 
        return hourly_pv_output, dni_data, tmy_data_


# In[ ]:
#######################################################################################################################################################################################################################

class csp_system:
    def __init__(self, latitude, longitude, theta_z):
        self.latitude = latitude
        self.longitude = longitude
        self.sigma = 5.67e-8
        self.emissivity = 0.07
        self.surface_temp = 4.93e2**4 
        self.theta_z = theta_z
        
        # optics parameters
        self.n1 = np.array([1,1.33]) # air, water
        self.n2 = 1.585 # polycarbonate 
        self.absorption_fp = 0.95
        self.reflection = 0.92
        self.optical_efficiency = 0.95
        self.transmission_glass = 0.92
        self.absorption_collector = 0.95
    
    def optimal_azimuth(self):
        if self.latitude > 0:
            return 180  # True south for Northern Hemisphere
        else:
            return 0    # True north for Southern Hemisphere
        
    def calculate_AOI(self, tmy_data_,tilt):
        azimuth = self.optimal_azimuth()
        solar_position = pvlib.solarposition.get_solarposition(tmy_data_.index, self.latitude, self.longitude)
        zenith_angle = solar_position['zenith']
        azimuth_angle = solar_position['azimuth']
        AOI = pvlib.irradiance.aoi(tilt, azimuth, zenith_angle, azimuth_angle)
        AOI = np.deg2rad(AOI)
        return AOI, zenith_angle
    
    def calculate_transmission_coefficient(self, AOI, n1, n2):
        trsmn1 = []
        trsmn2 = []
        
        for aoi in AOI:
            theta = aoi
            n1cos1 = n1[0] * np.cos(theta)
            n2sqrt1 = n2 * np.sqrt(1 - (n1[0]**2 / n2**2) * np.sin(theta)**2)
            Rs1 = ((n1cos1 - n2sqrt1) ** 2) / ((n1cos1 + n2sqrt1) ** 2)
            Rp1 = ((n1cos1 - n2sqrt1) ** 2) / ((n1cos1 + n2sqrt1) ** 2)
            Ravg1 = (Rs1 + Rp1) / 2
            transmission_fp1 = (1 - Ravg1)**2
            trsmn1.append(transmission_fp1)

            n1cos2 = n1[1] * np.cos(theta)
            n2sqrt2 = n2 * np.sqrt(1 - (n1[1]**2 / n2**2) * np.sin(theta)**2)
            Rs2 = ((n1cos2 - n2sqrt2) ** 2) / ((n1cos2 + n2sqrt2) ** 2)
            Rp2 = ((n1cos2 - n2sqrt2) ** 2) / ((n1cos2 + n2sqrt2) ** 2)
            Ravg2 = (Rs2 + Rp2) / 2
            transmission_fp2 = (1 - Ravg2)**2
            trsmn2.append(transmission_fp2)

        transmission_fp = np.array(trsmn1) * np.array(trsmn2)
        return transmission_fp
    
    def csp_performance_simulation(self, tmy_data_, dni_data, amb_temp, collector_area, tilt):
        pipe_transmission_losses = 22 * (250 - amb_temp) / 225 
        AOI, zenith_angle = self.calculate_AOI(tmy_data_,tilt)
        transmission_fp = self.calculate_transmission_coefficient(AOI, self.n1, self.n2)
        sol_zen_ang = zenith_angle
        acceptance_angle = np.abs(tilt - sol_zen_ang)
        acceptance_angle = np.where(acceptance_angle > 30, 0, 1)
    
        collection_fraction = (transmission_fp * self.absorption_fp * self.reflection * self.optical_efficiency *
                               self.transmission_glass * self.absorption_collector * acceptance_angle * np.cos(AOI))
     
        incident_power = collection_fraction * collector_area * dni_data
        energy_emitted_from_surface = self.sigma*self.emissivity*self.surface_temp**4 
        radiated_power = energy_emitted_from_surface * collector_area * (0.382 / 0.66)
        net_power_after_radlosses = incident_power - radiated_power
        net_power_received = [max(0, net - pipe) for net, pipe in zip(net_power_after_radlosses, pipe_transmission_losses)]
        net_power_received = np.array(incident_power)
        total_losses = np.sum(pipe_transmission_losses) + radiated_power
    
        if collector_area == 0:
            flux = np.zeros_like(net_power_received)
        else: 
            flux = net_power_received / collector_area

        csp_thermal_power = flux * collector_area / 1000  # kW
        return csp_thermal_power



# In[ ]:
#######################################################################################################################################################################################################################

class tes_system:
    def __init__(self, period, no_ts, load, amb_temp):
        
        self.period = period
        self.load = load 
        self.amb_temp = amb_temp
        self.no_ts = no_ts
        self.tes_charge_efficiency = 0.95  # TES charge efficiency
        self.tes_loss_rate = 0.020833333/100  # TES energy loss rate per hour
        self.tes_min_threshold = 20  # Minimum energy level in kWh to maintain in TES
        self.min_tes_output = 10  # Minimum TES output in kW_e 
        self.eff_particleheater = 0.99 # Particle Heater efficiency
        self.grid_power_required = np.zeros(period)
        self.yearly_tes_stored = np.zeros(period)
        self.excess_pv = np.zeros(period)
        self.pv_to_load = np.zeros(period)
        self.tes_to_load = np.zeros(period)
        self.CSP_gen = np.zeros(period)
        self.grid_contribution_to_ice_annual = np.zeros(period)
        self.pv_excess_contribution_to_ice_annual = np.zeros(period)
        self.leftover_PV_after_ice_ann = np.zeros(period)
    
    def th2e_efficiency(self, T):
        ambient_temps = [-5, 5, 10, 15, 20, 25, 30]
        conversion_rates = [0.27972028, 0.27972028, 0.27972028, 0.307692308, 0.323624595, 0.374531835, 0.374531835]
        
        if T <= ambient_temps[0]:
            return conversion_rates[0]
        elif T >= ambient_temps[-1]:
            return conversion_rates[-1]
        else: # interpolates
            for i in range(len(ambient_temps) - 1):
                if ambient_temps[i] <= T < ambient_temps[i + 1]:
                    eta_val = (conversion_rates[i] * (ambient_temps[i + 1] - T) +
                               conversion_rates[i + 1] * (T - ambient_temps[i])) / (ambient_temps[i + 1] - ambient_temps[i])
                    return eta_val
    
    def rte_value(self, T):
        ambient_temps = [-5, 5, 10, 15, 20, 25, 30]
        rte_values = [1.48, 1.48, 1.48, .95, .82, .66, .66]

        if T <= ambient_temps[0]:
            return rte_values[0]
        elif T >= ambient_temps[-1]:
            return rte_values[-1]
        else: # Interpolates
            for i in range(len(ambient_temps) - 1):
                if ambient_temps[i] <= T < ambient_temps[i + 1]:
                    rte_val = (rte_values[i] * (ambient_temps[i + 1] - T) +
                            rte_values[i + 1] * (T - ambient_temps[i])) / (ambient_temps[i + 1] - ambient_temps[i])
                    return rte_val
                
    def energybalance(self, hourly_pv_output, csp_thermal_power, tes_capacity, maximum_hourly_tes_output, gridtied):
        tes_stored = 0
        total_energy_supplied = 0
        total_energy_req_for_ice = np.zeros(self.period)
        excess_pv_to_tes_nc = np.zeros(self.period)
        curtailed_energy = np.zeros(self.period)
        load = self.load / 1000 # [kWh]
        load = [load] * 8760 * self.no_ts
        tes_capacity = tes_capacity / 1000 # [kWhth]
        maximum_hourly_tes_output = maximum_hourly_tes_output / 1000 # [kWh]
        curtailedpvafterchargingtes = []
        #energy_lost_to_surroundings = []
  
        for i in range(self.period):
            tempforhour = self.amb_temp[i]
            pv_generation = hourly_pv_output[i]
            charge_from_csp = csp_thermal_power[i]
            self.CSP_gen[i] = charge_from_csp

            # calculate pv available after meeting ice requirement
            current_rte = self.rte_value(tempforhour) 
            energy_req_for_ice = charge_from_csp*self.th2e_efficiency(tempforhour)/ current_rte
            if pv_generation - energy_req_for_ice > 0:
                pv_available = pv_generation - energy_req_for_ice
                energy_req_for_ice = 0
            else: 
                pv_available = 0
                energy_req_for_ice -= pv_generation
                heat_waste = energy_req_for_ice*current_rte/self.th2e_efficiency(tempforhour)
                charge_from_csp -= heat_waste 

            # Charging TES from CSP
            possible_charge = min(charge_from_csp, tes_capacity - tes_stored)
            tes_stored += possible_charge * self.tes_charge_efficiency
            ###curtailed_csp.append(charge_from_csp - possible_charge) ###
            ###energy_lost_to_surroundings[i] += (possible_charge * (1 - self.tes_charge_efficiency)) ###

            # Apply TES energy loss 
            tes_stored *= (1 - self.tes_loss_rate / self.no_ts) 
            ###energy_lost_to_surroundings[i] += tes_stored * (self.tes_loss_rate / self.no_ts) ###
            
            # Store the current TES level
            self.yearly_tes_stored[i] = tes_stored

            # TES discharge logic 
            discharge_efficiency = self.th2e_efficiency(tempforhour) 
            if tes_stored > self.tes_min_threshold + self.min_tes_output / discharge_efficiency / self.no_ts:
                max_discharge = (tes_stored - self.tes_min_threshold) * discharge_efficiency
                max_tes_output = min(max_discharge, maximum_hourly_tes_output) 
                ###energylost = (tes_stored - self.tes_min_threshold) * (1- self.tes_discharge_efficiency) if max_tes_output == max_discharge else (tes_stored - self.tes_min_threshold) - maximum_hourly_tes_output / self.no_ts
                ###energy_lost_to_surroundings[i] += energylost ###
                if pv_available + self.min_tes_output < load[i]:
                    tes_output = min(max(self.min_tes_output, load[i] - pv_available), max_tes_output)
                    pv_excess = 0
                    pv_to_load = pv_available
                else:
                    tes_output = self.min_tes_output
                    pv_excess = pv_available + self.min_tes_output - load[i]
                    pv_to_load = load[i] - self.min_tes_output
            else:
                tes_output = 0
                pv_excess = max(0, pv_available - load[i])
                pv_to_load = load[i] if pv_available > load[i] else pv_available
                ###energy_lost_to_surroundings[i] += (self.tes_min_threshold + self.min_tes_output / self.tes_discharge_efficiency - tes_stored) 
        
            self.tes_to_load[i] = tes_output
            self.pv_to_load[i] = pv_to_load
            self.excess_pv[i] = pv_excess

            # Adjust TES stored energy based on the output
            tes_stored -= tes_output / discharge_efficiency
            # energy_lost_to_surroundings[i] += tes_output / self.tes_discharge_efficiency - tes_output  

            # scenario 1: gridtied, updated TES logic (excess PV to 1/4 elec, 3/4 ice making)
            # scenario 2: isolated, updated TES logic (excess PV to 1/4 elec, 3/4 ice making)

            ############### logic for isolated system
            if gridtied == False: 

                # Charging TES from PV excess:
                tempforhour = self.amb_temp[i]
                current_rte = self.rte_value(tempforhour) 
                elec_ratio = current_rte/(1+current_rte) 
                charge_from_pv_excess = (1-elec_ratio)*self.excess_pv[i]*self.eff_particleheater # excess PV in thermal 
                possible_charge = min(charge_from_pv_excess, tes_capacity - tes_stored)
                tes_stored += possible_charge * self.tes_charge_efficiency
                curtailed_pv_excess = charge_from_pv_excess - possible_charge
                pv_excess_to_ice_production = elec_ratio*self.excess_pv[i]
                self.pv_excess_contribution_to_ice_annual[i] = pv_excess_to_ice_production
                total_energy_req_for_ice[i] = pv_excess_to_ice_production

           ################# logic for grid-tied system
            else:   
                # Charging TES from PV excess: 
                current_rte = self.rte_value(tempforhour) 
                charge_csp_elec = charge_from_csp*self.th2e_efficiency(tempforhour)
                energy_req_for_ice_from_csp = charge_csp_elec / current_rte
                ice_req_met_by_pv_excess = min(energy_req_for_ice_from_csp, self.excess_pv[i])
                pv_excess_leftover = max(0, self.excess_pv[i]-ice_req_met_by_pv_excess)
                # pv excess first goes to meeting ice requirement 
                grid_power_req_for_ice = energy_req_for_ice_from_csp - ice_req_met_by_pv_excess
                # leftover goes to charging TES and meeting corresponding ice req
                elec_ratio = current_rte/(1+current_rte)

                charge_from_pv_excess = (1 - elec_ratio)*pv_excess_leftover*self.eff_particleheater # thermal excess pv that can go to TES
                possible_charge = min(charge_from_pv_excess, tes_capacity - tes_stored)
                tes_stored += possible_charge * self.tes_charge_efficiency
                curtailed_pv_excess = charge_from_pv_excess - possible_charge
                pv_excess_to_ice_prod_for_pv_to_tes = elec_ratio*pv_excess_leftover
                self.pv_excess_contribution_to_ice_annual[i] = pv_excess_to_ice_prod_for_pv_to_tes + ice_req_met_by_pv_excess
                # append grid contribution and total energy req
                self.grid_contribution_to_ice_annual[i] = grid_power_req_for_ice
                total_energy_req_for_ice[i] += energy_req_for_ice_from_csp + pv_excess_to_ice_prod_for_pv_to_tes

                    

            # check if the combined power from PV and TES meets the load
            if self.pv_to_load[i] + tes_output >= load[i]:
                total_energy_supplied += load[i]
            else:
                total_energy_supplied += self.pv_to_load[i]  + tes_output
                self.grid_power_required[i] += load[i] - (self.pv_to_load[i]  + tes_output)
        
        
        load_met_percentage = (total_energy_supplied / np.sum(load)) * 100
        total_RES_power_to_load = (self.pv_to_load + self.tes_to_load)
        
        leftoverpvafterice = np.sum(self.leftover_PV_after_ice_ann)
        curtailedpvafterchargingtes = np.sum(curtailedpvafterchargingtes)
        curtailed_pv = curtailedpvafterchargingtes
    

        #curtailed_or_lost_energy = np.zeros(self.period)
        #curtailed_or_lost_energy += np.sum(curtailed_pv)              # excess PV that can't go to ice production/TES
        #curatiled_or_lost_energy += np.sum(curtailed_csp)                  # excess CSP that can't go to TES
        #curtailed_or_lost_energy += np.sum(energy_lost_to_surroundings)    # energy lost to surroundings (from TES)

        #total_energy_in = np.sum(csp_thermal_power) + np.sum(hourly_pv_output) # total energy in = pv + csp
        #total_energy_out = np.sum(self.pv_to_load) + np.sum(self.tes_to_load) + np.sum(curtailed_or_lost_energy) # energy out = pv/tes to load + curtailed/lost energy
        #print(f'energy diff {total_energy_in - total_energy_out}')

        return load_met_percentage, total_RES_power_to_load, load, self.yearly_tes_stored, self.CSP_gen, self.tes_to_load, self.pv_to_load, self.grid_power_required, self.excess_pv, self.grid_contribution_to_ice_annual, self.pv_excess_contribution_to_ice_annual, total_energy_req_for_ice, curtailed_pv
# In[ ]:
#######################################################################################################################################################################################################################

class TEA_Calculations:
    def __init__(self, load_met_percentage, PV_AC_Rating, tes_capacity, maximum_hourly_tes_output, collector_area, grid_power_required, grid_contribution_to_ice_annual, load,
                 C_PV=890, C_CSP=85, C_TES=3.7, C_PB=3000, C_grid=0.03):
        # System Params 
        self.load_met_percentage = load_met_percentage
        self.grid_power_required = grid_power_required
        self.PV_AC_Rating = PV_AC_Rating
        self.tes_capacity = tes_capacity
        self.maximum_hourly_tes_output =  maximum_hourly_tes_output # Maximum TES output in kW_e 
        self.collector_area = collector_area
        self.C_PV = C_PV # Cost of PV [$ / kW]
        self.C_CSP = C_CSP # Cost of CSP [$/m2]
        self.C_TES = C_TES # Cost per capacity of TES [$/kWh_th]
        self.C_PB = C_PB # Cost of power block [$/kW]
        self.C_grid = C_grid
        
        # O&M Params
        self.C_OM_TES = 0.0037 # [$/kW-year] TES O&M cost
        self.C_OM_CSP = 1 # [$/m2-year] CSP O&M cost
        self.C_OM_PV = 22 # [$/kW-year] PV O&M cost
        self.C_OM_PB = 45 # power block O&M cost [$/kW-year]

        # TEA Params
        self.n = 30  # [yrs] Analysis period
        self.L = 30  # [yrs] Operational Life
        self.esc = 0.02  # [frac] Escalation rate
        self.DF = 0.5  # [frac] Debt fraction
        self.I = 0.08  # [frac] Nominal Interest rate
        self.tax = 0.257  # [frac] State and federal tax rate
        self.COE = 0.13  # [frac] Cost of equity
        self.inflation = 0.028  # [frac] Inflation rate
        self.ITC = 0.0  # [frac] Internal tax credit
        self.property_tax = 0.0084  # [frac] Property Tax Rate
        self.insurance = 0.004  # [frac] Insurance rate
        self.MACRS_yrs = 7  # [yrs] MACRS Depreciation Period
        self.load = load

    def calculate_combined_power_to_load(self, pv_to_load, tes_to_load):
        combined_power_to_load = pv_to_load + tes_to_load 
        total_power_to_load = np.sum(combined_power_to_load)
        return total_power_to_load

    def calculate_CAPEX(self):
        CAPEX = self.C_PV * self.PV_AC_Rating / 1000 + self.C_TES * self.tes_capacity / 1000 + self.C_PB * self.maximum_hourly_tes_output / 1000 + self.C_CSP * self.collector_area
        return CAPEX

    def individual_CAPEX(self):
        PV_cc, TES_cc, PB_cc, CSP_cc = self.C_PV * self.PV_AC_Rating / 1000, self.C_TES * self.tes_capacity / 1000, self.C_PB * self.maximum_hourly_tes_output / 1000, self.C_CSP * self.collector_area
        return PV_cc, TES_cc, PB_cc, CSP_cc 

    def calculate_LCOE_OM(self, pv_to_load, tes_to_load, grid_power_required, gridtied, grid_contribution_to_ice_annual):

        load = self.load
        if gridtied: 
            PO = np.sum(load)
        else:
            PO = self.calculate_combined_power_to_load(pv_to_load, tes_to_load)

        PV_cc, TES_cc, PB_cc, CSP_cc = self.individual_CAPEX()

        data = {'PV_AC_Rating': [self.PV_AC_Rating], 
                'TES_capacity': [self.tes_capacity],
                'Collector_area' : [self.collector_area],
                'P_block': [self.maximum_hourly_tes_output],
                'CAPEX': [self.calculate_CAPEX()],
                'PO': [PO],
                'load_met_percentage': [self.load_met_percentage]}   
    
        df = pd.DataFrame(data)
        #print(grid_power_required, grid_contribution_to_ice_annual,PO)
        results = df.apply(hybrid_tea_model_photon, axis=1,
                                      args=(grid_power_required, grid_contribution_to_ice_annual, gridtied, self.load_met_percentage, 
                                            self.PV_AC_Rating, self.tes_capacity, self.maximum_hourly_tes_output, self.collector_area,
                                            PV_cc, TES_cc, PB_cc, CSP_cc, self.C_OM_TES, self.C_OM_PV, self.C_OM_CSP, self.C_OM_PB, self.n, self.L, self.esc, self.DF, self.I, self.tax, self.COE, 
                                            self.inflation, self.ITC, self.property_tax, self.insurance, self.MACRS_yrs, df['CAPEX']))
        
        result_series = results.iloc[0]

        LCOE = result_series['LCOE']
        C_OM_TES = result_series['C_OM_TES']
        C_OM_PV = result_series['C_OM_PV']
        C_OM_CSP = result_series['C_OM_CSP']
        C_OM_PB = result_series['C_OM_PB']
        CAPEX = result_series['CAPEX']
        NPV = result_series['NPV']
        totalgridcost = result_series['total_grid_cost']
        total_energy_prod = result_series['total_energy_prod']
    
        return LCOE, PV_cc, TES_cc, PB_cc, CSP_cc, C_OM_TES, C_OM_PV, C_OM_CSP, C_OM_PB, CAPEX, NPV, total_energy_prod, totalgridcost


# In[ ]:
#######################################################################################################################################################################################################################

def optimize_system(load, plant_location, minimum_load_met_percentage, gridtied, n_calls):
    
    solar_data_fetcher_ins = solar_data_fetcher(plant_location=plant_location, interpolating=False)
    tmy_data, no_ts, timeunit, latitude, longitude, theta_z, amb_temp, tz, period = solar_data_fetcher_ins.fetch_solar_data()
    
    results = []

    def objective_function(params, load=load, tmy_data=tmy_data, no_ts=no_ts, latitude=latitude, longitude=longitude, 
                       theta_z=theta_z, amb_temp=amb_temp, tz=tz, period=period, n_calls=n_calls):
        collector_area, tes_capacity, maximum_hourly_tes_output, tilt, PV_AC_Rating = params
        
        # PV System
        pv_sys = pv_system(latitude, longitude, tz, no_ts)
        hourly_pv_output, dni_data, tmy_data_ = pv_sys.run_pv_simulation(tmy_data, PV_AC_Rating, tilt=tilt)
        
        # CSP System
        csp_sys = csp_system(latitude, longitude, theta_z)
        csp_thermal_power = csp_sys.csp_performance_simulation(tmy_data_, dni_data, amb_temp, collector_area, tilt=tilt)
        
        # TES System
        tes_sys = tes_system(period, no_ts, load, amb_temp)
        load_met_percentage, total_RES_power_to_load, load, yearly_tes_stored, CSP_gen, tes_to_load, pv_to_load, grid_power_required, excess_pv, grid_contribution_to_ice_annual, pv_excess_contribution_to_ice_annual, total_energy_req_for_ice, curtailed_pv = tes_sys.energybalance(hourly_pv_output, csp_thermal_power, tes_capacity, maximum_hourly_tes_output, gridtied)
        
        # LCOE calculation
        tea_calc = TEA_Calculations(load_met_percentage, PV_AC_Rating, tes_capacity, maximum_hourly_tes_output, collector_area, grid_power_required, grid_contribution_to_ice_annual,  load)
        LCOE, PV_cc, TES_cc, PB_cc, CSP_cc, C_OM_TES, C_OM_PV, C_OM_CSP, C_OM_PB, CAPEX, NPV, total_energy_prod, totalgridcost = tea_calc.calculate_LCOE_OM(pv_to_load, tes_to_load, grid_power_required, gridtied, grid_contribution_to_ice_annual)
        
        results.append((LCOE, load_met_percentage, params))
        
        return LCOE

    lcoe_values = []
    iterations = []

    def monitor_optimization(res):
        iteration = len(res.x_iters)
        lcoe = res.func_vals[-1]
        lcoe_values.append(lcoe)
        iterations.append(iteration)
        #print(f"Iteration {iteration}: LCOE = {lcoe:.3f}")

    # Bounds for params
    space = [
        Integer(0, 7500, name='collector_area'),
        Real(0, 30*load, name='tes_capacity'),
        Real(0, 2*load, name='max_hourly_tes_output'),
        Real(35, 45, name='tilt'), 
        Real(0, 3*load, name='pv_capacity')
    ]

    # optimization
    result = gp_minimize(
        objective_function,
        space,
        n_calls=n_calls,  # Number of evaluations
        random_state=68,
        callback=[monitor_optimization]
    )

    # filter results based on minimum load met percentage threshold
    valid_results = [(lcoe, load_met, params) for lcoe, load_met, params in results if load_met >= minimum_load_met_percentage]
    
    if not valid_results:
        print("No solution found that meets the load met percentage requirement")
        return None

    # find the result with the minimum LCOE
    min_lcoe_result = min(valid_results, key=lambda x: x[0])
    min_lcoe, load_met_percentage, optimal_params = min_lcoe_result

    # plot for optimization progression
    #plt.figure(figsize=(10, 5))
    #plt.plot(iterations, lcoe_values, marker='o', linestyle='-', color='b')
    #plt.xlabel('Iteration')
    #plt.ylabel('LCOE')
    #plt.title('Optimization Progress')
    #plt.grid(True)
    #plt.show()

    print(f"Optimal Number of Collectors: {optimal_params[0]}")
    print(f"Optimal TES Capacity: {optimal_params[1]}")
    print(f"Optimal Max Hourly TES Output: {optimal_params[2]}")
    print(f"Optimal Tilt: {optimal_params[3]}")
    print(f"Optimal PV Capacity: {optimal_params[4]}")
    print(f"Minimum LCOE: {min_lcoe}")
    print(f"Load Met Percentage for Minimum LCOE: {load_met_percentage}")

    return optimal_params



# In[ ]:
#######################################################################################################################################################################################################################

def get_optimized_results(optimal_params, plant_location, load, gridtied):
    collector_area, tes_capacity, maximum_hourly_tes_output, tilt, pv_capacity = optimal_params

    solar_data_fetcher_ins = solar_data_fetcher(plant_location=plant_location, interpolating=False)
    tmy_data, no_ts, timeunit, latitude, longitude, theta_z, amb_temp, tz, period = solar_data_fetcher_ins.fetch_solar_data()
    
    # PV System
    pv_sys = pv_system(latitude, longitude, tz, no_ts)
    hourly_pv_output, dni_data, tmy_data_ = pv_sys.run_pv_simulation(tmy_data, pv_capacity, tilt=tilt)
    
    # CSP System
    csp_sys = csp_system(latitude, longitude, theta_z)
    csp_thermal_power = csp_sys.csp_performance_simulation(tmy_data_, dni_data, amb_temp, collector_area, tilt=tilt)
    
    # TES System
    tes_sys = tes_system(period, no_ts, load, amb_temp)
    load_met_percentage, total_RES_power_to_load, load, yearly_tes_stored, CSP_gen, tes_to_load, pv_to_load, grid_power_required, excess_pv, grid_contribution_to_ice_annual, pv_excess_contribution_to_ice_annual, total_energy_req_for_ice, curtailed_pv = tes_sys.energybalance(
            hourly_pv_output, csp_thermal_power, tes_capacity, maximum_hourly_tes_output, gridtied)
    # LCOE calculation
    tea_calc = TEA_Calculations(load_met_percentage, pv_capacity, tes_capacity, maximum_hourly_tes_output, collector_area, grid_power_required, grid_contribution_to_ice_annual,  load)
    LCOE, PV_cc, TES_cc, PB_cc, CSP_cc, C_OM_TES, C_OM_PV, C_OM_CSP, C_OM_PB, CAPEX, NPV, total_energy_prod, totalgridcost = tea_calc.calculate_LCOE_OM(pv_to_load, tes_to_load, grid_power_required, gridtied, grid_contribution_to_ice_annual)
    print(LCOE, PV_cc, TES_cc, PB_cc, CSP_cc, C_OM_TES, C_OM_PV, C_OM_CSP, C_OM_PB, CAPEX, NPV, total_energy_prod, totalgridcost)
    print(f'LCOE: {LCOE}')
    print(f'load met percentage : {load_met_percentage}')
    return no_ts, timeunit, amb_temp, load, period, hourly_pv_output, dni_data, csp_thermal_power, load_met_percentage, total_RES_power_to_load, yearly_tes_stored, CSP_gen, tes_to_load, pv_to_load, grid_power_required, excess_pv, grid_contribution_to_ice_annual, pv_excess_contribution_to_ice_annual, total_energy_req_for_ice


# In[ ]:
#######################################################################################################################################################################################################################

################################################################### Plotting ##########################################################################

#### Annual Performance Plots
def annual_performance_plots(gridtied, days, startday, no_ts, timeunit, amb_temp, period, hourly_pv_output, dni_data, csp_thermal_power, load_met_percentage, total_RES_power_to_load, load, yearly_tes_stored, CSP_gen, tes_to_load, pv_to_load, grid_power_required, excess_pv, grid_contribution_to_ice_annual, pv_excess_contribution_to_ice_annual, total_energy_req_for_ice): 
    display = 24*days*no_ts + startday*24
    hrs = np.arange(0, period, 1)
    starthr = startday*24
    time_del = 10

    # Energy generation from PV and CSP  
    plt.figure(figsize=(12, 6))
    plt.plot(hrs[starthr:display], CSP_gen[starthr:display], label='CSP Generation (kW)', color='blue')
    plt.plot(hrs[starthr:display], hourly_pv_output[starthr:display], label='PV Generation (kW)', color='green')
    #plt.plot(hrs[starthr:display], load[starthr:display], label='Load (kW)', color='red')
    plt.plot(hrs[starthr:display], dni_data[starthr:display], label='DNI (kW)', color='purple')
    #plt.plot(hrs[starthr:display], grid_power_required[0:display], label='Grid to Load(kW)', color='gray')
    #plt.plot(hrs[starthr:display], tes_to_load[0:display], label='TES to load (kW)', color='orange')
    plt.xlabel(timeunit)
    plt.ylabel('Power (kW)')
    plt.title('Energy Generation')
    plt.legend(loc = 'upper left')

    # Energy balance - what is going to load 
    plt.figure(figsize=(12, 6))
    plt.fill_between(hrs[starthr:display], 0, tes_to_load[starthr:display], color='red', alpha=0.5, label='TES to Load')
    plt.fill_between(hrs[starthr:display], tes_to_load[starthr:display], tes_to_load[starthr:display] + pv_to_load[starthr:display], color='blue', alpha=0.5, label='PV to Load')
    plt.fill_between(hrs[starthr:display], total_RES_power_to_load[starthr:display], total_RES_power_to_load[starthr:display] + grid_power_required[starthr:display], color='gray', alpha=0.5, label='Grid to Load (kW)')
    plt.plot(hrs[starthr:display], load[starthr:display], label='Load (kW)', color='black', linewidth=2)
    plt.xlabel(timeunit)
    plt.ylabel('Power (kW)')
    plt.title('Energy Supplied to the Load')
    plt.legend(loc = 'upper left')
    plt.show()

    # Ice production 
   
    plt.figure(figsize=(12, 6))
    color_excess_pv = 'purple'
    color_pv_to_ice = 'orange'
    color_grid_to_ice = 'yellow'
    color_ice_req = 'black'
    plt.fill_between(hrs[starthr:display], 0, excess_pv[starthr:display], color=color_excess_pv, alpha=0.3, label='PV excess (kW)')
    plt.fill_between(hrs[starthr:display], 0, pv_excess_contribution_to_ice_annual[starthr:display], color=color_pv_to_ice, alpha=1, label='PV excess going to ice production (kW)')
    plt.fill_between(hrs[starthr:display], pv_excess_contribution_to_ice_annual[starthr:display], pv_excess_contribution_to_ice_annual[starthr:display] + grid_contribution_to_ice_annual[starthr:display], color=color_grid_to_ice, alpha=0.8, label='Grid going to ice production (kW)')
    plt.plot(hrs[starthr:display], total_energy_req_for_ice[starthr:display], label='Ice Production Requirement (kW)', color=color_ice_req, linewidth=2)

    plt.xlabel(timeunit)
    plt.ylabel('Power (kW)')
    plt.title('Energy Supplied to Ice Production')
    plt.legend(loc = 'upper left')
    plt.show()

    # Energy stored in TES
    plt.figure(figsize=(12, 6))
    plt.fill_between(hrs[0:display], 0, yearly_tes_stored[0:display], color='red', alpha=0.5)
    plt.xlabel(timeunit)
    plt.ylabel('Power (kW)')
    plt.title('Stored Energy')
    plt.legend()
    plt.show()


    #print(f"Percentage of the load met by the PV and TES system: {load_met_percentage:.2f}%")


# In[ ]:
#######################################################################################################################################################################################################################

#### Parametric plots, LCOE vs capacity plots

def parametric_plots(load, plant_location, gridtied, n_data_points, csv_name):

            # scenario 1: gridtied, updated TES logic (excess PV to 1/4 elec, 3/4 ice making)
            # scenario 2: isolated, updated TES logic (excess PV to 1/4 elec, 3/4 ice making)

    collector_area_min, collector_area_max = 0, 25e3
    tes_capacity_min, tes_capacity_max = 1e3, 2e9
    pv_capacity_min, pv_capacity_max = 0, 10e6
    maximum_hourly_tes_output_min,maximum_hourly_tes_output_max = 1e3, load

    def run_simulation(params, tmy_data, no_ts, timeunit, latitude, longitude, theta_z, amb_temp, tz, period, load):
        collector_area, tes_capacity, PV_AC_Rating, maximum_hourly_tes_output = params
        
        # PV System
        pv_sys = pv_system(latitude, longitude, tz, no_ts)
        hourly_pv_output, dni_data, tmy_data_ = pv_sys.run_pv_simulation(tmy_data, PV_AC_Rating, tilt=latitude)
        
        # CSP System
        csp_sys = csp_system(latitude, longitude, theta_z)
        csp_thermal_power = csp_sys.csp_performance_simulation(tmy_data_, dni_data, amb_temp, collector_area, tilt=latitude)
        
        # TES System
        tes_sys = tes_system(period, no_ts, load, amb_temp)
        load_met_percentage, total_RES_power_to_load, load, yearly_tes_stored, CSP_gen, tes_to_load, pv_to_load, grid_power_required, excess_pv, grid_contribution_to_ice_annual, pv_excess_contribution_to_ice_annual, total_energy_req_for_ice, curtailed_pv = tes_sys.energybalance(
            hourly_pv_output, csp_thermal_power, tes_capacity, maximum_hourly_tes_output, gridtied)
        
        # LCOE calculation
        tea_calc = TEA_Calculations(load_met_percentage, PV_AC_Rating, tes_capacity, maximum_hourly_tes_output, collector_area, grid_power_required, grid_contribution_to_ice_annual, load)
        LCOE, PV_cc, TES_cc, PB_cc, CSP_cc, C_OM_TES, C_OM_PV, C_OM_CSP, C_OM_PB, CAPEX, NPV, total_energy_prod, totalgridcost = tea_calc.calculate_LCOE_OM(pv_to_load, tes_to_load, grid_power_required, gridtied, grid_contribution_to_ice_annual)
                                    
        return load_met_percentage, LCOE, PV_cc, TES_cc, PB_cc, CSP_cc, C_OM_TES, C_OM_PV, C_OM_CSP, C_OM_PB, CAPEX, NPV, total_energy_prod, curtailed_pv, totalgridcost

    
    def generate_and_plot_results(tmy_data, no_ts, timeunit, latitude, longitude, theta_z, amb_temp, tz, period, load, collector_area_array, tes_capacity_array, pv_capacity_array, max_hourly_tes_output_array):
        results_list = []
        for collector_area, tes_capacity, pv_capacity, maximum_hourly_tes_output in product(collector_area_array, tes_capacity_array, pv_capacity_array,max_hourly_tes_output_array):
            params = [collector_area, tes_capacity, pv_capacity, maximum_hourly_tes_output]
            load_met_percentage, LCOE, PV_cc, TES_cc, PB_cc, CSP_cc, C_OM_TES, C_OM_PV, C_OM_CSP, C_OM_PB, CAPEX, NPV, total_energy_prod, curtailed_pv, totalgridcost = run_simulation(params, tmy_data, no_ts, timeunit, latitude, longitude, theta_z, amb_temp, tz, period, load)
            
            results_list.append({
                'LCOE': LCOE,
                'load_met_percentage': load_met_percentage,
                'collector_area': collector_area,
                'tes_capacity': tes_capacity,
                'pv_capacity': pv_capacity,
                'maximum_hourly_tes_output' : maximum_hourly_tes_output,
                'PV_cc' : PV_cc,
                'TES_cc' : TES_cc,
                'PB_cc' : PB_cc,
                'CSP_cc' : CSP_cc,
                'C_OM_TES' : C_OM_TES,
                'C_OM_PV' : C_OM_PV, 
                'C_OM_CSP' : C_OM_CSP,
                'C_OM_PB' : C_OM_PB,
                'CAPEX' : CAPEX,
                'NPV' : NPV,
                'total_energy_prod' : total_energy_prod,
                'PV surplus' : curtailed_pv,
                'total grid cost' : totalgridcost,
            })

        results_df = pd.DataFrame(results_list)
        results_df = results_df.dropna(subset=['load_met_percentage', 'LCOE'])

        min_lcoe_idx = results_df.groupby('load_met_percentage')['LCOE'].idxmin()
        min_lcoe_df = results_df.loc[min_lcoe_idx]

        return results_df, min_lcoe_df

    def generate_parameter_arrays(n_data_points, collector_area_min, collector_area_max, tes_capacity_min, tes_capacity_max, pv_capacity_min, pv_capacity_max,maximum_hourly_tes_output_min,maximum_hourly_tes_output_max):
        collector_area_array = np.linspace(collector_area_min, collector_area_max, n_data_points).astype(int)
        tes_capacity_array = np.linspace(tes_capacity_min, tes_capacity_max, n_data_points).astype(int)
        pv_capacity_array = np.linspace(pv_capacity_min, pv_capacity_max, n_data_points).astype(int)
        max_hourly_tes_output_array = np.linspace(maximum_hourly_tes_output_min,maximum_hourly_tes_output_max, 3).astype(int)
        return collector_area_array, tes_capacity_array, pv_capacity_array, max_hourly_tes_output_array

    solar_data_fetcher_ins = solar_data_fetcher(plant_location=plant_location, interpolating=False)
    tmy_data, no_ts, timeunit, latitude, longitude, theta_z, amb_temp, tz, period = solar_data_fetcher_ins.fetch_solar_data()

    # parameter arrays
    collector_area_array, tes_capacity_array, pv_capacity_array, max_hourly_tes_output_array = generate_parameter_arrays(
        n_data_points, collector_area_min, collector_area_max, tes_capacity_min, tes_capacity_max, pv_capacity_min, pv_capacity_max, maximum_hourly_tes_output_min, maximum_hourly_tes_output_max)
    if gridtied:
        results_df, min_lcoe_df = generate_and_plot_results(tmy_data, no_ts, timeunit, latitude, longitude, theta_z, amb_temp, tz, period, load, collector_area_array, tes_capacity_array, pv_capacity_array, max_hourly_tes_output_array) 
        results_df['scenario'] = 'Charge TES with PV Excess = True'
        results_df = results_df
        min_lcoe_df = min_lcoe_df

        fig = make_subplots(rows=1, cols=1, subplot_titles=(
            'LCOE vs Load Met Percentage for Grid-tied System (Scenario 1)',
            ' '
        ))
        trace_true = go.Scatter(
            x=results_df['load_met_percentage'],
            y=results_df['LCOE'],
            mode='markers',
            name='Charge TES with PV Excess = True',
            marker=dict(color='blue'),
            text=(
                'Collector area (m2): ' + results_df['collector_area'].astype(str) + '<br>' +
                'TES Capacity (Whth): ' + results_df['tes_capacity'].astype(str) + '<br>' +
                'PV Capacity  (W): ' + results_df['pv_capacity'].astype(str) + '<br>' +
                'Max hourly TES output (Wh): ' + results_df['maximum_hourly_tes_output'].astype(str) + '<br>' +
                'LCOE: ' + results_df['LCOE'].astype(str) + '<br>' +
                'Load Met Percentage: ' + results_df['load_met_percentage'].astype(str)
            ),
        hoverinfo='text'
        )
        fig.add_trace(trace_true, row=1, col=1)
        fig.update_xaxes(title_text='Load Met Percentage', row=1, col=1)
        fig.update_yaxes(title_text='LCOE', row=1, col=1)

        fig.show()

    else:
        
         # get results for both scenarios
        results_df, min_lcoe_df = generate_and_plot_results(tmy_data, no_ts, timeunit, latitude, longitude, theta_z, amb_temp, tz, period, load, collector_area_array, tes_capacity_array, pv_capacity_array, max_hourly_tes_output_array)
        results_df['scenario'] = 'Charge TES with PV Excess = True'
        results_df = results_df
        min_lcoe_df = min_lcoe_df

        # subplots
        fig = make_subplots(rows=1, cols=1, subplot_titles=(
            'LCOE vs Load Met Percentage for Isolated System (Scenario 2)',
            ' '
        ))
        
        trace_true = go.Scatter(
            x=results_df['load_met_percentage'],
            y=results_df['LCOE'],
            mode='markers',
            name='Charge TES with PV Excess = True',
            marker=dict(color='blue'),
            text=(
                    'Collector area (m2): ' + results_df['collector_area'].astype(str) + '<br>' +
                    'TES Capacity (Whth): ' + results_df['tes_capacity'].astype(str) + '<br>' +
                    'PV Capacity  (W): ' + results_df['pv_capacity'].astype(str) + '<br>' +
                    'Max hourly TES output (Wh): ' + results_df['maximum_hourly_tes_output'].astype(str) + '<br>' +
                    'LCOE: ' + results_df['LCOE'].astype(str) + '<br>' +
                    'Load Met Percentage: ' + results_df['load_met_percentage'].astype(str)
                ),
            hoverinfo='text'
        )
        fig.add_trace(trace_true, row=1, col=1)
        
        fig.update_xaxes(title_text='Load Met Percentage', row=1, col=1)
        fig.update_yaxes(title_text='LCOE', row=1, col=1)

        fig.show()
    results_df.to_csv(csv_name)
    min_csv_name = csv_name.replace('.csv', '_min_values.csv')
    min_lcoe_df.to_csv(min_csv_name)

def plot_performance(load, plant_location, gridtied, collector_area, tes_capacity, pv_capacity,maximum_hourly_tes_output, days, startday):

    solar_data_fetcher_ins = solar_data_fetcher(plant_location=plant_location, interpolating=False)
    tmy_data, no_ts, timeunit, latitude, longitude, theta_z, amb_temp, tz, period = solar_data_fetcher_ins.fetch_solar_data()
    tilt = latitude
    results_list = []
    
    
    # PV System
    pv_sys = pv_system(latitude, longitude, tz, no_ts)
    hourly_pv_output, dni_data, tmy_data_ = pv_sys.run_pv_simulation(tmy_data, pv_capacity, tilt=tilt)
    
    # CSP System
    csp_sys = csp_system(latitude, longitude, theta_z)
    csp_thermal_power = csp_sys.csp_performance_simulation(tmy_data_, dni_data, amb_temp, collector_area, tilt=tilt)
    
    # TES System
    tes_sys = tes_system(period, no_ts, load, amb_temp)
    load_met_percentage, total_RES_power_to_load, load, yearly_tes_stored, CSP_gen, tes_to_load, pv_to_load, grid_power_required, excess_pv, grid_contribution_to_ice_annual, pv_excess_contribution_to_ice_annual, total_energy_req_for_ice, curtailed_pv = tes_sys.energybalance(
            hourly_pv_output, csp_thermal_power, tes_capacity, maximum_hourly_tes_output, gridtied)
    # LCOE calculation
    tea_calc = TEA_Calculations(load_met_percentage, pv_capacity, tes_capacity, maximum_hourly_tes_output, collector_area, grid_power_required, grid_contribution_to_ice_annual,  load)
    LCOE, PV_cc, TES_cc, PB_cc, CSP_cc, C_OM_TES, C_OM_PV, C_OM_CSP, C_OM_PB, CAPEX, NPV, total_energy_prod, totalgridcost = tea_calc.calculate_LCOE_OM(pv_to_load, tes_to_load, grid_power_required, gridtied, grid_contribution_to_ice_annual)
    
    # output LCOE
    print(f'LCOE: {round(LCOE,2)} $/kWh')
    print(f'load met percentage: {round(load_met_percentage,2)}')

    results_list.append({
                'LCOE': LCOE,
                'load_met_percentage': load_met_percentage,
                'collector_area': collector_area,
                'tes_capacity': tes_capacity,
                'pv_capacity': pv_capacity,
                'max_hourly_tes_output' : maximum_hourly_tes_output,
                'PV_cc' : PV_cc,
                'TES_cc' : TES_cc,
                'PB_cc' : PB_cc,
                'CSP_cc' : CSP_cc,
                'C_OM_TES' : C_OM_TES,
                'C_OM_PV' : C_OM_PV, 
                'C_OM_CSP' : C_OM_CSP,
                'C_OM_PB' : C_OM_PB,
                'CAPEX' : CAPEX,
                'NPV' : NPV,
                'total_energy_prod' : total_energy_prod,
                'PV surplus' : curtailed_pv,
                'total grid cost' : totalgridcost,
            })

    results_df = pd.DataFrame(results_list)
    results_df = results_df.dropna(subset=['load_met_percentage', 'LCOE'])


    # annual performance plots
    annual_performance_plots(gridtied, days, startday, no_ts, timeunit, amb_temp, period, hourly_pv_output, dni_data, csp_thermal_power, load_met_percentage, total_RES_power_to_load, load, yearly_tes_stored, CSP_gen, tes_to_load, pv_to_load, grid_power_required, excess_pv, grid_contribution_to_ice_annual, pv_excess_contribution_to_ice_annual, total_energy_req_for_ice)


 
# In[ ]:
#######################################################################################################################################################################################################################

#### Tornado sensitivity plot :

def generate_and_plot_tornado_sensitivities(load, load_met_percentage, tes_to_load, pv_to_load, grid_power_required, grid_contribution_to_ice_annual, gridtied, optimal_params):
    cost_parameter_ranges = {
        'pv_cost': [590, 890, 1200],
        'csp_cost': [800, 1275, 2000],
        'tes_cost': [1.7, 3.7, 5.7],
        'power_block_cost': [1000, 3000, 4400]
    }
    
    base_case_costs = {
        'pv_cost': 890,
        'csp_cost': 1275,
        'tes_cost': 3.7,
        'power_block_cost': 3000
    }

    collector_area, tes_capacity, PV_AC_Rating, maximum_hourly_tes_output = int(optimal_params[0]), optimal_params[1], optimal_params[4], optimal_params[2]


    # base case LCOE
    base_case_tea_calc = TEA_Calculations(
        load_met_percentage, PV_AC_Rating, tes_capacity, maximum_hourly_tes_output, collector_area, grid_power_required, grid_contribution_to_ice_annual, load,
        C_PV=base_case_costs['pv_cost'], C_CSP=base_case_costs['csp_cost'], C_TES=base_case_costs['tes_cost'], C_PB=base_case_costs['power_block_cost']
    )
    base_case_LCOE, PV_cc, TES_cc, PB_cc, CSP_cc, C_OM_TES, C_OM_PV, C_OM_CSP, C_OM_PB, CAPEX, NPV, total_energy_prod, totalgridcost = base_case_tea_calc.calculate_LCOE_OM(pv_to_load, tes_to_load, grid_power_required, gridtied, grid_contribution_to_ice_annual)
    print(f'Base Case LCOE: {base_case_LCOE:.2f}')

    parameter_names = []
    low_lcoe = []
    high_lcoe = []

    for param in cost_parameter_ranges:
        ranges = cost_parameter_ranges[param]
        lcoe_values = []

        for value in ranges:
            costs = base_case_costs.copy()
            costs[param] = value
            
            tea_calc = TEA_Calculations(
                load_met_percentage, PV_AC_Rating, tes_capacity, maximum_hourly_tes_output, collector_area, grid_power_required, load,
                C_PV=costs['pv_cost'], C_CSP=costs['csp_cost'], C_TES=costs['tes_cost'], C_PB=costs['power_block_cost']
            )
            LCOE, PV_cc, TES_cc, PB_cc, CSP_cc, C_OM_TES, C_OM_PV, C_OM_CSP, C_OM_PB, CAPEX, NPV, total_energy_prod, totalgridcost = tea_calc.calculate_LCOE_OM(pv_to_load, tes_to_load, grid_power_required, gridtied, grid_contribution_to_ice_annual)
            lcoe_values.append(LCOE)

        parameter_names.append(param)
        low_lcoe.append(min(lcoe_values))
        high_lcoe.append(max(lcoe_values))

    df = pd.DataFrame({
        'Parameter': parameter_names,
        'Low LCOE': low_lcoe,
        'High LCOE': high_lcoe
    })

    df['Total Impact Range'] = df['High LCOE'] - df['Low LCOE']
    df_sorted = df.sort_values(by='Total Impact Range', ascending=False)

   
    fig, ax = plt.subplots(figsize=(12, 8))

    #  low LCOE range (base vs low) in green
    bars_low = ax.barh(df_sorted['Parameter'], df_sorted['Low LCOE'] - base_case_LCOE, color='green', label='Low vs Base')

    #  high LCOE range (base vs high) in blue
    bars_high = ax.barh(df_sorted['Parameter'], df_sorted['High LCOE'] - base_case_LCOE, color='blue', left=df_sorted['Low LCOE'] - base_case_LCOE, label='High vs Base')

    for i, (low, high, param) in enumerate(zip(df_sorted['Low LCOE'], df_sorted['High LCOE'], df_sorted['Parameter'])):
        ax.text(df_sorted['Low LCOE'].iloc[i] - base_case_LCOE / 2, i, f'{df_sorted["Low LCOE"].iloc[i]:.2f}', va='center', ha='center', color='white')
        ax.text(df_sorted['Low LCOE'].iloc[i] + (df_sorted['High LCOE'].iloc[i] - df_sorted['Low LCOE'].iloc[i]) / 2, i, f'{df_sorted["High LCOE"].iloc[i]:.2f}', va='center', ha='center', color='white')

    ax.set_xlabel('LCOE Value')
    ax.set_title('Tornado Sensitivity Plot')
    ax.axvline(x=base_case_LCOE, color='black', linestyle='--', label='Base Case LCOE')  # Line at base case LCOE
    ax.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

 


# In[ ]:
#######################################################################################################################################################################################################################

#### Contour plots:

def contour_plots():

    def plot_contours(csp_fraction_values, tes_capacity_values, metric_values, metric_name):
        csp_fraction_values = np.array(csp_fraction_values)
        tes_capacity_values = np.array(tes_capacity_values)
        metric_values = np.array(metric_values)

        data = pd.DataFrame({
            'CSP Fraction': np.repeat(csp_fraction_values, len(tes_capacity_values)),
            'TES Capacity': np.tile(tes_capacity_values, len(csp_fraction_values)),
            'Metric': metric_values.flatten()
        })

        x = np.linspace(csp_fraction_values.min(), csp_fraction_values.max(), 100)
        y = np.linspace(tes_capacity_values.min(), tes_capacity_values.max(), 100)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)

        for i in range(len(data)):
            xi = np.searchsorted(x, data['CSP Fraction'].iloc[i])
            yi = np.searchsorted(y, data['TES Capacity'].iloc[i])
            Z[yi, xi] = data['Metric'].iloc[i]


        plt.figure(figsize=(12, 8))
        plt.contourf(X, Y, Z, levels=50, cmap='viridis', norm=Normalize(vmin=Z.min(), vmax=Z.max()))
        plt.colorbar(label=metric_name)
        plt.xlabel('CSP Fraction')
        plt.ylabel('TES Capacity (Wh)')
        plt.title(f'{metric_name} Contour Plot')
        plt.grid(True)
        plt.show()

    # LCOE contour
    plot_contours(csp_fraction_values, tes_capacity_values, lcoe_values, 'LCOE')

    # Percent Load Met contour
    plot_contours(csp_fraction_values, tes_capacity_values, load_met_percentages, 'Percent Load Met')


# In[ ]:




