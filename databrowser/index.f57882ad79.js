var qs=Object.defineProperty;var Fr=(e,t)=>()=>(e&&(t=e(e=0)),t);var zr=(e,t)=>{for(var a in t)qs(e,a,{get:t[a],enumerable:!0})};var Sn={};zr(Sn,{BUILTIN_METADATA:()=>Al});var Al,Cn=Fr(()=>{Al=Object.freeze({cmor_table:Object.freeze({"1day":"daily mean samples","1hr":"hourly mean samples","1hr-cf":"","1hr-era5":"","1hrpt":"sampled hourly, at specified time point within the time period","3hr":"3-hourly data","5min":"5 minutes mean samples","6hr":"6 hourly mean samples","6hrlev":"6-hourly data on atmospheric model levels","6hrplev":"6-hourly atmospheric data on pressure levels (time mean)","6hrplevpt":"6-hourly atmospheric data on pressure levels (instantaneous)","6hrpt":"sampled 6 hourly, at specified time point within the time period",aday:"daily atmospheric data",aerday:"Daily atmospheric chemistry and aerosol data",aerfx:"Fixed atmospheric chemistry and aerosol data",aerhr:"Hourly atmospheric chemistry and aerosol data",aermon:"Monthly atmospheric chemistry and aerosol data",aermonz:"Monthly atmospheric chemistry and aerosol data",aero:"Aerosol data",amon:"Monthly atmospheric data",cf3hr:"3-hourly associated with cloud forcing",cfday:"Daily data associated with cloud forcing",cfmon:"Monthly data associated with cloud forcing",cfoff:"CFMIP (Cloud Feedback Model Intercomparison Project) monthly offline",cfsites:"CFMIP (Cloud Feedback Model Intercomparison Project) high frequency Cloud Diagnostic Fields",cfsubhr:"Diagnostics for cloud forcing analysis at specific sites","day-fc":"",e1hr:"Hourly Atmospheric Data (extension)",e1hrclimmon:"Diurnal Cycle",e3hr:"3-hourly (time mean, extension)",e3hrpt:"3-hourly (instantaneous, extension)",e6hrz:"6-hourly Zonal Mean (extension)",eday:"Daily (time mean, extension)",edayz:"Daily Zonal Mean (extension)",efx:"Fixed (extension)",emon:"Monthly (time mean, extension)",emonz:"Monthly zonal means (time mean, extension)",esubhr:"Sub-hourly (extension)",eyr:"Daily (time mean, extension)",fx:"Fixed variables",ifxant:"Fixed fields on the Antarctic ice sheet",ifxgre:"Fixed fields on the Greenland ice sheet",imonant:"Monthly fields on the Antarctic ice sheet",imongre:"Monthly fields on the Greenland ice sheet",iyrant:"Annual fields on the Antarctic ice sheet",iyrgre:"Annual fields on the Greenland ice sheet",limon:"Monthly fields for the terrestrial cryosphere",lmon:"Monthly land surface and soil model fields",mon:"Monthly mean samples","mon-cf":"","mon-era5":"",oclim:"Monthly climatologies of ocean data",ocnbgchem:"Ocean Biogeochemistry",oday:"Daily ocean data",odec:"Decadal ocean data",ofx:"Fixed ocean data",oimon:"Monthly ocean ice data",olmon:"",omon:"Monthly ocean data",oyr:"Annual ocean variables",sem:"Weekly mean samples",siday:"Daily sea-ice data",simon:"Monthly sea-ice data",yr:"Annual mean samples"}),fs_type:Object.freeze({hsm:"StrongLink tape archive",posix:"normal unix filesystem",swift:"OpenStack cloud storage"}),grid_label:Object.freeze({gm:"global mean data",gn:"data reported on a model's native grid",gna:"data reported on a native grid in the region of Antarctica",gng:"data reported on a native grid in the region of Greenland",gnz:"zonal mean data reported on a model's native latitude grid",gr:"regridded data reported on the data provider's preferred target grid",gr1:"regridded data reported on a grid other than the native grid and other than the preferred target grid",gr1a:"regridded data reported in the region of Antarctica on a grid other than the native grid and other than the preferred target grid",gr1g:"regridded data reported in the region of Greenland on a grid other than the native grid and other than the preferred target grid",gr1z:"regridded zonal mean data reported on a grid other than the native latitude grid and other than the preferred latitude target grid",gr2:"regridded data reported on a grid other than the native grid and other than the preferred target grid",gr2a:"regridded data reported in the region of Antarctica on a grid other than the native grid and other than the preferred target grid",gr2g:"regridded data reported in the region of Greenland on a grid other than the native grid and other than the preferred target grid",gr2z:"regridded zonal mean data reported on a grid other than the native latitude grid and other than the preferred latitude target grid",gr3:"regridded data reported on a grid other than the native grid and other than the preferred target grid",gr3a:"regridded data reported in the region of Antarctica on a grid other than the native grid and other than the preferred target grid",gr3g:"regridded data reported in the region of Greenland on a grid other than the native grid and other than the preferred target grid",gr3z:"regridded zonal mean data reported on a grid other than the native latitude grid and other than the preferred latitude target grid",gr4:"regridded data reported on a grid other than the native grid and other than the preferred target grid",gr4a:"regridded data reported in the region of Antarctica on a grid other than the native grid and other than the preferred target grid",gr4g:"regridded data reported in the region of Greenland on a grid other than the native grid and other than the preferred target grid",gr4z:"regridded zonal mean data reported on a grid other than the native latitude grid and other than the preferred latitude target grid",gr5:"regridded data reported on a grid other than the native grid and other than the preferred target grid",gr5a:"regridded data reported in the region of Antarctica on a grid other than the native grid and other than the preferred target grid",gr5g:"regridded data reported in the region of Greenland on a grid other than the native grid and other than the preferred target grid",gr5z:"regridded zonal mean data reported on a grid other than the native latitude grid and other than the preferred latitude target grid",gr6:"regridded data reported on a grid other than the native grid and other than the preferred target grid",gr6a:"regridded data reported in the region of Antarctica on a grid other than the native grid and other than the preferred target grid",gr6g:"regridded data reported in the region of Greenland on a grid other than the native grid and other than the preferred target grid",gr6z:"regridded zonal mean data reported on a grid other than the native latitude grid and other than the preferred latitude target grid",gr7:"regridded data reported on a grid other than the native grid and other than the preferred target grid",gr7a:"regridded data reported in the region of Antarctica on a grid other than the native grid and other than the preferred target grid",gr7g:"regridded data reported in the region of Greenland on a grid other than the native grid and other than the preferred target grid",gr7z:"regridded zonal mean data reported on a grid other than the native latitude grid and other than the preferred latitude target grid",gr8:"regridded data reported on a grid other than the native grid and other than the preferred target grid",gr8a:"regridded data reported in the region of Antarctica on a grid other than the native grid and other than the preferred target grid",gr8g:"regridded data reported in the region of Greenland on a grid other than the native grid and other than the preferred target grid",gr8z:"regridded zonal mean data reported on a grid other than the native latitude grid and other than the preferred latitude target grid",gr9:"regridded data reported on a grid other than the native grid and other than the preferred target grid",gr9a:"regridded data reported in the region of Antarctica on a grid other than the native grid and other than the preferred target grid",gr9g:"regridded data reported in the region of Greenland on a grid other than the native grid and other than the preferred target grid",gr9z:"regridded zonal mean data reported on a grid other than the native latitude grid and other than the preferred latitude target grid",gra:"regridded data in the region of Antarctica reported on the data provider's preferred target grid",grg:"regridded data in the region of Greenland reported on the data provider's preferred target grid",grz:"regridded zonal mean data reported on the data provider's preferred latitude target grid"}),institute:Object.freeze({aer:"Research and Climate Group, Atmospheric and Environmental Research, 131 Hartwell Avenue, Lexington, MA 02421, USA","as-rcec":"Research Center for Environmental Changes, Academia Sinica, Nankang, Taipei 11529, Taiwan",atmos:"Deutscher Wetterdienst",awi:"Alfred Wegener Institute, Helmholtz Centre for Polar and Marine Research, Am Handelshafen 12, 27570 Bremerhaven, Germany",b1:"Max Planck Institute for Meteorology",bcc:"Beijing Climate Center(BCC),China Meteorological Administration,China",bnu:"GCESS,BNU,Beijing,China",cams:"Chinese Academy of Meteorological Sciences, Beijing 100081, China",cas:"Chinese Academy of Sciences, Beijing 100029, China",cccma:"CCCma (Canadian Centre for Climate Modelling and Analysis, Victoria, BC, Canada)","cccr-iitm":"Centre for Climate Change Research, Indian Institute of Tropical Meteorology Pune, Maharashtra 411 008, India",cclm:"COSMO-CLM - Climate Limited-area Modelling Community",cesm:"Community Earth System Model",clmcom:"Climate Limited-area Modelling Community (CLM-Community)","cm-saf":"EUMETSAT/CMSAF",cmcc:"Fondazione Centro Euro-Mediterraneo sui Cambiamenti Climatici, Lecce 73100, Italy",cnes:"Centre National d'\xC3\x89tudes Spatiales",cnrm:"Centre National de Recherches Meteorologiques","cnrm-cerfacs":"CNRM (Centre National de Recherches Meteorologiques, Toulouse 31057, France), CERFACS (Centre Europeen de Recherche et de Formation Avancee en Calcul Scientifique, Toulouse 31057, France)","cola-cfs":"COLA (Center for Ocean-Land-Atmosphere Studies, Calverton, MD)",cru:"Data held at British Atmospheric Data Centre, RAL, UK.",csiro:"Commonwealth Scientific and Industrial Research Organisation, Aspendale, Victoria 3195, Australia","csiro-arccss":"CSIRO (Commonwealth Scientific and Industrial Research Organisation, Aspendale, Victoria 3195, Australia), ARCCSS (Australian Research Council Centre of Excellence for Climate System Science). Mailing address: CSIRO, c/o Simon J. Marsland, 107-121 Station Street, Aspendale, Victoria 3195, Australia","csiro-bom":"CSIRO (Commonwealth Scientific and Industrial Research Organisation, Australia), and BOM (Bureau of Meteorology, Australia)","csiro-cosima":"CSIRO (Commonwealth Scientific and Industrial Research Organisation, Australia), COSIMA (Consortium for Ocean-Sea Ice Modelling in Australia). Mailing address: CSIRO, c/o Simon J. Marsland, 107-121 Station Street, Aspendale, Victoria 3195, Australia","csiro-qccce":"Australian Commonwealth Scientific and Industrial Research Organization (CSIRO) Marine and Atmospheric Research (Melbourne, Australia) in collaboration with the Queensland Climate Change Centre of Excellence (QCCCE) (Brisbane, Australia)",dhmz:"Meteorological and Hydrological Service of Croatia",dkrz:"Deutsches Klimarechenzentrum, Hamburg 20146, Germany",dlr:"Max-Planck-Institute for Meteorology",dmi:"Danish Meteorological Institute",dwd:"Deutscher Wetterdienst, Offenbach am Main 63067, Germany","e3sm-project":"LLNL (Lawrence Livermore National Laboratory, Livermore, CA 94550, USA); ANL (Argonne National Laboratory, Argonne, IL 60439, USA); BNL (Brookhaven National Laboratory, Upton, NY 11973, USA); LANL (Los Alamos National Laboratory, Los Alamos, NM 87545, USA); LBNL (Lawrence Berkeley National Laboratory, Berkeley, CA 94720, USA); ORNL (Oak Ridge National Laboratory, Oak Ridge, TN 37831, USA); PNNL (Pacific Northwest National Laboratory, Richland, WA 99352, USA); SNL (Sandia National Laboratories, Albuquerque, NM 87185, USA). Mailing address: LLNL Climate Program, c/o David C. Bader, Principal Investigator, L-103, 7000 East Avenue, Livermore, CA 94550, USA","ec-earth-consortium":"AEMET, Spain; BSC, Spain; CNR-ISAC, Italy; DMI, Denmark; ENEA, Italy; FMI, Finland; Geomar, Germany; ICHEC, Ireland; ICTP, Italy; IDL, Portugal; IMAU, The Netherlands; IPMA, Portugal; KIT, Karlsruhe, Germany; KNMI, The Netherlands; Lund University, Sweden; Met Eireann, Ireland; NLeSC, The Netherlands; NTNU, Norway; Oxford University, UK; surfSARA, The Netherlands; SMHI, Sweden; Stockholm University, Sweden; Unite ASTR, Belgium; University College Dublin, Ireland; University of Bergen, Norway; University of Copenhagen, Denmark; University of Helsinki, Finland; University of Santiago de Compostela, Spain; Uppsala University, Sweden; Utrecht University, The Netherlands; Vrije Universiteit Amsterdam, the Netherlands; Wageningen University, The Netherlands. Mailing address: EC-Earth consortium, Rossby Center, Swedish Meteorological and Hydrological Institute/SMHI, SE-601 76 Norrkoping, Sweden",ecad:"European Climate Assessment and Dataset",ecmwf:"European Centre for Medium-Range Weather Forecasts, Reading RG2 9AX, UK",essl:"Freie Universitaet Berlin - Institute of Meteorology",fio:"FIO(The First Institution of Oceanography,SOA,Qingdao,China)","fio-qlnm":"FIO (First Institute of Oceanography, Ministry of Natural Resources, Qingdao 266061, China), QNLM (Qingdao National Laboratory for Marine Science and Technology, Qingdao 266237, China)",fub:"Freie Universitaet Berlin, Institute of Meteorology","fub-dwd":"Freie Universitaet Berlin / Deutscher Wetterdienst","fub-ifm":"Freie Universitaet Berlin - Institute of Meteorology",geomar:"Max-Planck-Institute for Meteorology",gerics:"Helmholtz-Zentrum Geesthacht, Climate Service Center Germany","gfdl-esm":"Geophysical Fluid Dynamics Laboratory Earth System Model.",gfs:"Global Forecast System",gfz:"GFZ (German Research Centre for Geosciences, Potsdam, Germany)",hadobs:"Met Office Hadley Centre","hammoz-consortium":"ETH Zurich, Switzerland; Max Planck Institut fur Meteorologie, Germany; Forschungszentrum Julich, Germany; University of Oxford, UK; Finnish Meteorological Institute, Finland; Leibniz Institute for Tropospheric Research, Germany; Center for Climate Systems Modeling (C2SM) at ETH Zurich, Switzerland",hms:"Hungarian Meteorological Service",ichec:"EC-Earth (European Earth System Model)",ictp:"International Centre for Theoretical Physics",ifm:"Max Planck Institute for Meteorology",inm:"Institute for Numerical Mathematics, Russian Academy of Science, Moscow 119991, Russia",inpe:"INPE CCST, Av Astronautas 1758, Sao Jose dos Campos, SP, 12220-010, Brazil, (http://www.inpe.br)",ipsl:"Institut Pierre Simon Laplace, Paris 75252, France","ipsl-ineris":"IPSL (Institut Pierre Simon Laplace) and INERIS (Institut National de l Environnement industriel et des RISques)","jma-criepi":"Japan Meteorological Agency (JMA) and the Central Research Institute of Electric Power Industry (CRIEPI).",kiost:"Korea Institute of Ocean Science and Technology, Busan 49111, Republic of Korea",knmi:"Royal Netherlands Meteorological Institute, De Bilt, The Netherlands","lasg-cess":"IAP (Institute of Atmospheric Physics, Chinese Academy of Sciences, Beijing, China) and THU (Tsinghua University)","lasg-iap":"IAP(Institute of Atmospheric Physics),CAS(Chinese Academy of Sciences),Beijing,China",llnl:"Lawrence Livermore National Laboratory, Livermore, CA 94550, USA. Mailing address: LLNL Climate Program, c/o Stephen A. Klein, Principal Investigator, L-103, 7000 East Avenue, Livermore, CA 94550, USA",loa_ipsl:"Laboratoire d'Optique Atmosph\xC3\xA9rique - Institut Pierre-Simon Laplace (IPSL)","messy-consortium":"The Modular Earth Submodel System (MESSy) Consortium, represented by the Institute for Physics of the Atmosphere, Deutsches Zentrum fur Luft- und Raumfahrt (DLR), Wessling, Bavaria 82234, Germany","miklip-module-c":"IMK-TRO/KIT (Institut fuer Meteorologie und Klimaforschung/Karlsruhe Institut of Technology, Karlsruhe, Germany)",miroc:"JAMSTEC (Japan Agency for Marine-Earth Science and Technology, Kanagawa 236-0001, Japan), AORI (Atmosphere and Ocean Research Institute, The University of Tokyo, Chiba 277-8564, Japan), NIES (National Institute for Environmental Studies, Ibaraki 305-8506, Japan), and R-CCS (RIKEN Center for Computational Science, Hyogo 650-0047, Japan)",miub:"Max Planck Institute for Meteorology, Hamburg 20146, Germany","module-c":"IMK-TRO/KIT (Institut fuer Meteorologie und Klimaforschung/Karlsruhe Institut of Technology, Karlsruhe, Germany)",mohc:"Met Office Hadley Centre, Fitzroy Road, Exeter, Devon, EX1 3PB, UK, (http://www.metoffice.gov.uk)","mohc-cru":"Freie Universitaet Berlin - Institute of Meteorology",mpi:"Max Planck Institute for Meteorology","mpi-csc":"Helmholtz-Zentrum Geesthacht, Climate Service Center, Max Planck Institute for Meteorology","mpi-dwd":"Max Planck Institute for Meteorology","mpi-m":"Max Planck Institute for Meteorology, Hamburg 20146, Germany","mpi-m-1":"Max Planck Institute for Meteorology","mpi-m-2":"Max Planck Institute for Meteorology","mpi-m-3":"Max Planck Institute for Meteorology","mpi-m-4":"Max Planck Institute for Meteorology","mpi-m-5":"Max Planck Institute for Meteorology","mpim-dwd":"Freie Universitaet Berlin - Institute of Meteorology",mri:"Meteorological Research Institute, Tsukuba, Ibaraki 305-0052, Japan",nasa:"National Aeronautics and Space Administration","nasa-giss":"Goddard Institute for Space Studies, New York, NY 10025, USA","nasa-gmao":"Global Modeling and Assimilation Office, NASA Goddard Space Flight Center, Greenbelt, MD 20771","nasa-gsfc":"NASA Goddard Space Flight Center, Greenbelt, MD 20771, USA","nasa-jpl":"National Aeronautics and Space Administration, Jet Propulsion Laboratory","nasa-larc":"NASA Langley Research Center, Hampton, Va, USA",ncar:"National Center for Atmospheric Research, Climate and Global Dynamics Laboratory, 1850 Table Mesa Drive, Boulder, CO 80305, USA","ncar-ucar":" National Center for Atmospheric Research - University Corporation for Atmospheric Research",ncc:"NorESM Climate modeling Consortium consisting of CICERO (Center for International Climate and Environmental Research, Oslo 0349), MET-Norway (Norwegian Meteorological Institute, Oslo 0313), NERSC (Nansen Environmental and Remote Sensing Center, Bergen 5006), NILU (Norwegian Institute for Air Research, Kjeller 2027), UiB (University of Bergen, Bergen 5007), UiO (University of Oslo, Oslo 0313) and UNI (Uni Research, Bergen 5008), Norway. Mailing address: NCC, c/o MET-Norway, Henrik Mohns plass 1, Oslo 0313, Norway","ncep-ncar":"National Centers for Environmental Prediction",nerc:"Natural Environment Research Council, STFC-RAL, Harwell, Oxford, OX11 0QX, UK",nicam:"Nonhydrostatic Icosahedral Atmospheric Model (NICAM) Group (RIGC-JAMSTEC/AORI-U.Tokyo/AICS-RIKEN,Japan)","nimr-kma":"NIMR (National Institute of Meteorological Research, Seoul, South Korea)","nims-kma":"National Institute of Meteorological Sciences/Korea Meteorological Administration, Climate Research Division, Seoho-bukro 33, Seogwipo-si, Jejudo 63568, Republic of Korea",niwa:"National Institute of Water and Atmospheric Research, Hataitai, Wellington 6021, New Zealand",noaa:"National Oceanographic Data Center(NODC)","noaa-cires":"NOAA ESRL/PSD","noaa-gfdl":"National Oceanic and Atmospheric Administration, Geophysical Fluid Dynamics Laboratory, Princeton, NJ 08540, USA","noaa-ncdc":"","noaa-ncep":"NCEP (National Centers for Environmental Prediction, Camp Springs, MD)",noaancepcpc:"National Centers for Environmental Prediction","nsf-doe-ncar":"NSF/DOE NCAR (National Center for Atmospheric Research) Boulder, CO, USA",ntu:"National Taiwan University, Taipei 10650, Taiwan",nuist:"Nanjing University of Information Science and Technology, Nanjing, 210044, China",pcmdi:"Program for Climate Model Diagnosis and Intercomparison, Lawrence Livermore National Laboratory, Livermore, CA 94550, USA","pnnl-waccem":"PNNL (Pacific Northwest National Laboratory), Richland, WA 99352, USA",pr:"Max Planck Institute for Meteorology",remss:"Remote Sensing Systems","rmib-ugent":"Royal Meteorological Institute of Belgium and Ghent University","rte-rrtmgp-consortium":"AER (Atmospheric and Environmental Research, Lexington, MA 02421, USA); UColorado (University of Colorado, Boulder, CO 80309, USA). Mailing address: AER c/o Eli Mlawer, 131 Hartwell Avenue, Lexington, MA 02421, USA",rubisco:"ORNL (Oak Ridge National Laboratory, Oak Ridge, TN 37831, USA); ANL (Argonne National Laboratory, Argonne, IL 60439, USA); BNL (Brookhaven National Laboratory, Upton, NY 11973, USA); LANL (Los Alamos National Laboratory, Los Alamos, NM 87545); LBNL (Lawrence Berkeley National Laboratory, Berkeley, CA 94720, USA); NAU (Northern Arizona University, Flagstaff, AZ 86011, USA); NCAR (National Center for Atmospheric Research, Boulder, CO 80305, USA); UCI (University of California Irvine, Irvine, CA 92697, USA); UM (University of Michigan, Ann Arbor, MI 48109, USA). Mailing address: ORNL Climate Change Science Institute, c/o Forrest M. Hoffman, Laboratory Research Manager, Building 4500N Room F106, 1 Bethel Valley Road, Oak Ridge, TN 37831-6301, USA",smhi:"Swedish Meteorological and Hydrological Institute, Rossby Centre",snu:"Seoul National University, Seoul 08826, Republic of Korea",thu:"Department of Earth System Science, Tsinghua University, Beijing 100084, China",ua:"Department of Geosciences, University of Arizona, Tucson, AZ 85721, USA",ub:"University of Bremen",uci:"Department of Earth System Science, University of California Irvine, Irvine, CA 92697, USA",ucsb:"Bren School of Environmental Science and Management, University of California, Santa Barbara. Mailing address: c/o Samantha Stevenson, 2400 Bren Hall, University of California Santa Barbara, Santa Barbara, CA 93106, USA",uhh:"Universitat Hamburg, Hamburg 20148, Germany",uoc:"University of Cologne",uor:"TAMSAT Research Group, Meteorology Department, University of Reading, UK",zmaw:"Institute of Oceanography, University of Hamburg"}),model:Object.freeze({"access1-0":"ACCESS1-0 2011. Atmosphere: AGCM v1.0 (N96 grid-point, 1.875 degrees EW x approx 1.25 degree NS, 38 levels); ocean: NOAA/GFDL MOM4p1 (nominal 1.0 degree EW x 1.0 degrees NS, tripolar north of 65N, equatorial refinement to 1/3 degree from 10S to 10 N, cosine dependent NS south of 25S, 50 levels); sea ice: CICE4.1 (nominal 1.0 degree EW x 1.0 degrees NS, tripolar north of 65N, equatorial refinement to 1/3 degree from 10S to 10 N, cosine dependent NS south of 25S); land: MOSES2 (1.875 degree EW x 1.25 degree NS, 4 levels<br>See http://wiki.csiro.au/confluence/display/ACCESS/ACCESS+Publications","access1-3":"ACCESS1.3 2011. Atmosphere: AGCM v1.0 (N96 grid-point, 1.875 degrees EW x approx 1.25 degree NS, 38 levels); ocean: NOAA/GFDL MOM4p1 (nominal 1.0 degree EW x 1.0 degrees NS, tripolar north of 65N, equatorial refinement to 1/3 degree from 10S to 10 N, cosine dependent NS south of 25S, 50 levels); sea ice: CICE4.1 (nominal 1.0 degree EW x 1.0 degrees NS, tripolar north of 65N, equatorial refinement to 1/3 degree from 10S to 10 N, cosine dependent NS south of 25S); land: CABLE1.0 (1.875 degree EW x 1.25 degree NS, 6 levels<br>See http://wiki.csiro.au/confluence/display/ACCESS/ACCESS+Publications","bcc-csm1-1":"bcc-csm1-1:atmosphere: BCC_AGCM2.1 (T42L26); land: BCC_AVIM1.0;ocean: MOM4_L40 (tripolar, 1 lon x (1-1/3) lat, L40);sea ice: SIS (tripolar,1 lon x (1-1/3) lat)<br>","bcc-csm1-1-m":"bcc-csm1-1-m:atmosphere: BCC_AGCM2.2 (T106L26); land: BCC_AVIM1.1;ocean: MOM4_L40v2 (tripolar, 1 lon x (1-1/3) lat, L40);sea ice: SIS (tripolar,1 lon x (1-1/3) lat)<br>","bcc-csm2-mr":"BCC-CSM 2 MR (2017): aerosol: none atmos: BCC_AGCM3_MR (T106; 320 x 160 longitude/latitude; 46 levels; top level 1.46 hPa) atmosChem: none land: BCC_AVIM2 landIce: none ocean: MOM4 (1/3 deg 10S-10N, 1/3-1 deg 10-30 N/S, and 1 deg in high latitudes; 360 x 232 longitude/latitude; 40 levels; top grid cell 0-10 m) ocnBgchem: none seaIce: SIS2<br>Model described by Tongwen Wu et al. (JGR 2013; JMR 2014; submmitted to GMD,2018). Also see http://forecast.bcccsm.ncc-cma.net/htm","bcc-esm1":"BCC-ESM 1 (2017): aerosol: none atmos: BCC_AGCM3_LR (T42; 128 x 64 longitude/latitude; 26 levels; top level 2.19 hPa) atmosChem: BCC-AGCM3-Chem land: BCC_AVIM2 landIce: none ocean: MOM4 (1/3 deg 10S-10N, 1/3-1 deg 10-30 N/S, and 1 deg in high latitudes; 360 x 232 longitude/latitude; 40 levels; top grid cell 0-10 m) ocnBgchem: none seaIce: SIS2<br>Model described by Tongwen Wu et al. (JGR 2013; JMR 2014; submmitted to GMD,2018). Also see http://forecast.bcccsm.ncc-cma.net/htm","bnu-esm":"BNU-ESM<br>",canam4:"CanAM4 2010 atmosphere: CanAM4 (AGCM15i, T63L35) land: CLASS2.7 (Note: Adjusted Land Cover and soil albedo relative to that used in CanESM2 and CanCM4)<br>http://www.cccma.ec.gc.ca/models",cancm4:"CanCM4 2010 atmosphere: CanAM4 (AGCM15i, T63L35) ocean: CanOM4 (OGCM4.0, 256x192L40) sea ice: CanSIM1 (Cavitating Fluid, T63 Gaussian Grid) land: CLASS2.7<br>http://www.cccma.ec.gc.ca/models",canesm2:"CanESM2 2010 atmosphere: CanAM4 (AGCM15i, T63L35) ocean: CanOM4 (OGCM4.0, 256x192L40) and CMOC1.2 sea ice: CanSIM1 (Cavitating Fluid, T63 Gaussian Grid) land: CLASS2.7 and CTEM1<br>http://www.cccma.ec.gc.ca/models",canesm5:`CanESM5 (2017): 
<br>Geophysical Model Development Special issue on CanESM5 (https://www.geosci-model-dev.net/special_issues.html)`,"cccma-canesm2-gerics-remo2015-v1":"<br>","cccma-canesm2-smhi-rca4-v1":"<br>http://www.smhi.se/en/Research/Research-departments/climate-research-rossby-centre",cclm:"<br>","cclm-4-21":"CCLM-4-21<br>http//www.clm-community.eu, http://www.kit.edu","cclm-4-8":"CCLM-4-8<br>http//www.clm-community.eu, http://www.kit.edu","cclm-5-0-7":"CCLM-5-0-7<br>http//www.clm-community.eu, http://www.kit.edu","cclm-5-0-9":"CCLM-5-0-9<br>http//www.clm-community.eu, http://www.kit.edu","cclm-eur-22":"CCLM-5-0-7<br>http//www.clm-community.eu, http://www.kit.edu",cclm0440:"<br>http://www.clm-community.eu/",ccsm4:"CCSM4<br>Gent P. R., et.al. 2011: The Community Climate System Model version 4. J. Climate, doi: 10.1175/2011JCLI4083.1",cdas:"<br>Kruschke, T. et al., Evaluating decadal predictions of northern hemispheric cyclone frequencies, Tellus A 66, 22830, doi: 10.3402/tellusa.v66.22830; Murray, R.J. & Simmonds, I., A numerical scheme for tracking cyclone centres from digital data. Part I: Development and operation of the scheme, Aust. Meteor. Mag., 1991, 39, 155166; Pinto, J.G. et al., Sensitivities of a cyclone detection and tracking algorithm: individual tracks and climatology Meteor. Z., 2005, 14, 823-838",cesm:"CAM<br>","cesm-dple":"CCSM POP2, the CCSM Ocean Component<br>","cesm-le":"CCSM POP2, the CCSM Ocean Component<br>","cesm1-bgc":`CESM1-BGC<br>Lindsay K., et al.: Preindustrial Control and 20th Century Experiments with the Earth System Model CESM1-(BGC) (in preparation for Journal of Climate).
`,"cesm1-cam5":"CESM1-CAM5<br>Neale, R., et.al. 2012: Coupled simulations from CESM1 using the Community Atmosphere Model version 5: (CAM5). See also http://www.cesm.ucar.edu/publications","cesm1-cam5-1-fv2":`CESM1-CAM5.1-FV2<br>Neale, R. B. et al. (2011a) Description of the NCAR Community Atmosphere Model (CAM5), Technical Report NCAR/TN-486+STR.
`,"cesm1-fastchem":"CESM1-FASTCHEM<br>TBD","cesm1-waccm":"CESM1-WACCM<br>Marsh, D., et.al. 2012: WACCM4 simulations of atmospheric trends from 1850 to present. See also http://www.cesm.ucar.edu/publications",cesm2:"CESM2 (2017): atmosphere: CAM6 (0.9x1.25 finite volume grid; 288 x 192 longitude/latitude; 32 levels; top level 2.25 mb); ocean: POP2 (320x384 longitude/latitude; 60 levels; top grid cell 0-10 m); sea_ice: CICE5.1 (same grid as ocean); land: CLM5 0.9x1.25 finite volume grid; 288 x 192 longitude/latitude; 32 levels; top level 2.25 mb); aerosol: MAM4 (0.9x1.25 finite volume grid; 288 x 192 longitude/latitude; 32 levels; top level 2.25 mb); atmoschem: MAM4 (0.9x1.25 finite volume grid; 288 x 192 longitude/latitude; 32 levels; top level 2.25 mb); landIce: CISM2.1; ocnBgchem: MARBL (320x384 longitude/latitude; 60 levels; top grid cell 0-10 m)<br>","cesm2-waccm":"CESM2 (2017): atmosphere: CAM6 (0.9x1.25 finite volume grid; 288 x 192 longitude/latitude; 70 levels; top level 4.5e-6 mb); ocean: POP2 (320x384 longitude/latitude; 60 levels; top grid cell 0-10 m); sea_ice: CICE5.1 (same grid as ocean); land: CLM5 0.9x1.25 finite volume grid; 288 x 192 longitude/latitude; 70 levels; top level 4.5e-6 mb); aerosol: MAM4 (0.9x1.25 finite volume grid; 288 x 192 longitude/latitude; 70 levels; top level 4.5e-6 mb); atmosChem: WACCM (0.9x1.25 finite volume grid; 288 x 192 longitude/latitude; 70 levels; top level 4.5e-6 mb; landIce: CISM2.1; ocnBgchem: MARBL (320x384 longitude/latitude; 60 levels; top grid cell 0-10 m)<br>",cesm_cam5:"CAM<br>",cfs:"CFSR Reanalysis from 1979 to 2011. analysis: Data assimilation system used is the 3-D variational GSI (Gridded Statistical Interpolation) scheme. The resolution is spectral T382 with 64 sigma-hybrid pressure levels in the vertical. The background guess is generated with a fully coupled atmosphere-ocean-land-seaice forecast system. The atmosphere and ocean analysis are done independently every 6 hours, while the land assimilation is done every 24 hours. Satellite radiances are directly assimilated over the period from 1979 to the present.; atmosphere: GFS (cfs/gfs, T382L64); ocean: MOM4 (cfs/mom4, 40 levels .25 deg at eq. .50 deg near poles); sea ice: (cfsv2-seaice, 3 layer on mom4 grid using Winton thermodynamics and modified CICE dynamics with 5 category ice thickness ); land: NOAH (cfs/noah 2009, T382/L4)<br>The model is described by the paper: Suranjana Saha, Shrinivas Moorthi, Xingren Wu, Jiande Wang, Sudhir Nadiga, Patrick Tripp, Hua-Lu Pan, David Behringer, Yu-Tai Hou, Hui-ya Chuang, Mark Iredell, Michael Ek, Jesse Meng, Rongqian Yang, Huug van den Dool, Qin Zhang, Wanqiu Wang, Mingyue Chen, 2011 : The NCEP Climate Forecast System Version 2. (To be submitted to the Journal of Climate.) and also the paper: Saha, Suranjana, and Coauthors, 2010: The NCEP Climate Forecast System Reanalysis. Bull. Amer. Meteor. Soc., 91, 1015.1057. doi: 10.1175/2010BAMS3001.1. Also see http://cfs.ncep.noaa.gov","cfsv2-2011":"CFSv2-2011: atmosphere: GFS (cfs/gfs, T126L64); ocean: MOM4 (cfs/mom4, 40 levels .25 deg at eq. .50 deg near poles); sea ice: (cfsv2-seaice, 3 layer on mom4 grid, 5 category using Winton thermodynamics and modified CICE dynamics); land: NOAH (cfs/noah 2009, T126/L4)<br>Model described by the paper:Suranjana Saha, Shrinivas Moorthi, Xingren Wu, Jiande Wang, Sudhir Nadiga, Patrick Tripp, Hua-Lu Pan, David Behringer, Yu-Tai Hou, Hui-ya Chuang, Mark Iredell, Michael Ek, Jesse Meng, Rongqian Yang, Huug van den Dool, Qin Zhang, Wanqiu Wang, Mingyue Chen, 2011 :The NCEP Climate Forecast System Version 2. (To be submitted to the Journal of Climate.) and also the paper:Saha, Suranjana, and Coauthors, 2010:The NCEP Climate Forecast System Reanalysis. Bull. Amer. Meteor. Soc., 91, 1015.1057. doi: 10.1175/2010BAMS3001.1. Also see http://cfs.ncep.noaa.go","cm-saf":"HOAPS-S Hamburg Ocean-Atmosphere Parameters and fluxes from Satellite, March 2011, Release 4.0<br>http://dx.doi.org/10.5676/EUM_SAF_CM/HOAPS/V002","cmcc-cesm":"CMCC-CESM<br>model described in the documentation at http://www.cmcc.it/data-models/models","cmcc-cm":"CMCC-CM<br>model described in the documentation at http://www.cmcc.it/data-models/models","cmcc-cms":"CMCC-CMS<br>model described in the documentation at http://www.cmcc.it/data-models/models",cnes:"SSALTO/DUACS<br>http://www.aviso.oceanobs.com/en/data/product-information/duacs/","cnrm-cerfacs-cnrm-cm5-clmcom-cclm4-8-17-v1":"CLMcom-CCLM4-8-17<br>http://www.clm-community.eu/","cnrm-cerfacs-cnrm-cm5-cnrm-aladin52-v1":"<br>http://www.cnrm-game.fr/spip.php?rubrique69&lang=en","cnrm-cerfacs-cnrm-cm5-cnrm-aladin53-v1":"<br>http://www.cnrm-game.fr/spip.php?rubrique69&lang=en","cnrm-cerfacs-cnrm-cm5-gerics-remo2015-v1":"<br>","cnrm-cerfacs-cnrm-cm5-hms-aladin52-v1":"<br>http://www.met.hu/en/RCM","cnrm-cerfacs-cnrm-cm5-rmib-ugent-alaro-0-v1":"<br>http://dx.doi.org/10.5194/gmd-9-1143-2016","cnrm-cerfacs-cnrm-cm5-smhi-rca4-v1":"<br>http://www.smhi.se/en/Research/Research-departments/climate-research-rossby-centre","cnrm-cm5":"CNRM-CM5 2010 Atmosphere: ARPEGE-Climat (V5.2.1, TL127L31); Ocean: NEMO (nemo3.3.v10.6.6P, ORCA1degL42); Sea Ice: GELATO (V5.30); River Routing: TRIP (v1); Land: SURFEX (v5.1.c); Coupler : OASIS 3<br>See http://www.cnrm.meteo.fr/cmip5 - Follow model description link","cnrm-cm5-2":"CNRM-CM5-2 2010 Atmosphere: ARPEGE-Climat (V5.2.3i, TL127L31); Ocean: &<br>This is an updated version of the model used for CMIP5 and described at http://www.cnrm.meteo.fr/cmip5 ; changes are : i) real salt and water fluxes between SeaIce and Ocean, ii) increased ice depth on permanent ice sheets to avoid complete melt under extreme global warming, iii) using a 2012 version of Szopa et al. prescribed aerosols and iv) solar forcing is no more the TIM-corrected one (i.e. values are around 1365 W/m**2) . A few variants of physics are used, please see the metadata field named comment. Comparing to physics #1 : physics #2 adds melt ponds; physics #3 uses 60 atmospheric levels instead of 31; physics #4 also, and adds a stratospheric chemical scheme ; physics #5 integrates biophysical coupling between phytoplankton and light penetration into the ocean as described in Lengaigne et al;, 2009","cnrm-cm6-1":"CNRM-CM6-1 (2017): aerosol: prescribed monthly fields computed by TACTIC_v2 scheme atmos: Arpege 6.3 (T127; Gaussian Reduced with 24572 grid points in total distributed over 128 latitude circles (with 256 grid points per latitude circle between 30degN and 30degS reducing to 20 grid points per latitude circle at 88.9degN and 88.9degS); 91 levels; top level 78.4 km) atmosChem: OZL_v2 land: Surfex 8.0c ocean: Nemo 3.6 (eORCA1, tripolar primarily 1deg; 362 x 294 longitude/latitude; 75 levels; top grid cell 0-1 m) seaIce: Gelato 6.1<br>http://www.umr-cnrm.fr/cmip6/references","cnrm-esm2-1":"CNRM-ESM2-1 (2017): aerosol: TACTIC_v2 atmos: Arpege 6.3 (T127; Gaussian Reduced with 24572 grid points in total distributed over 128 latitude circles (with 256 grid points per latitude circle between 30degN and 30degS reducing to 20 grid points per latitude circle at 88.9degN and 88.9degS); 91 levels; top level 78.4 km) atmosChem: REPROBUS-C_v2 land: Surfex 8.0c ocean: Nemo 3.6 (eORCA1, tripolar primarily 1deg; 362 x 294 longitude/latitude; 75 levels; top grid cell 0-1 m) ocnBgchem: Pisces 2.s seaIce: Gelato 6.1<br>http://www.umr-cnrm.fr/cmip6/references","cosmo-clm":"CCLM: lmparbin, INT2LM: @{INT2LM_EXE}<br>http://www.clm-community.eu/","csiro-mk3-6-0":"CSIRO-Mk3-6-0 2010 atmosphere: AGCM v7.3.8 (T63 spectral, 1.875 degrees EW x approx. 1.875 degrees NS, 18 levels); ocean: GFDL MOM2.2 (1.875 degrees EW x approx. 0.9375 degrees NS, 31 levels)<br>a) Rotstayn, L., Collier, M., Dix, M., Feng, Y., Gordon, H., O\\'Farrell, S., Smith, I. and Syktus, J. 2010. Improved simulation of Australian climate and ENSO-related climate variability in a GCM with an interactive aerosol treatment. Int. J. Climatology, vol 30(7), pp1067-1088, DOI 10.1002/joc.1952 b) Please refer to online documentation at: http://cmip-pcmdi.llnl.gov/cmip5/","csiro-qccce-csiro-mk3-6-0-smhi-rca4-v1":"<br>http://www.smhi.se/en/Research/Research-departments/climate-research-rossby-centre",dwd:"surface observation<br>Datenlieferung2018_AP101b_HYRAS-PRE.pdf","e3sm-1-0":`E3SM 1.0 (2018): 
<br>Golaz, J.-C., P. M. Caldwell, L. P. Van Roekel and co-authors, 2019: The DOE E3SM coupled model version 1: Overviewand evaluation at standard resolution. JAMES, submitted; http://e3sm.org`,"ec-earth":"EC-EARTH 2.3 (2011); atmosphere: IFS (cy31R1+modifications, T159L62); ocean: NEMO (version2+modifications, ORCA1-42lev); sea ice: LIM2; land: HTessel<br>Model described by Hazeleger et al. (Bull. Amer. Meteor. Soc., 2010, 91, 1357-1363). Also see http://ecearth.knmi.nl.","ec-earth3":`EC-Earth3 (2018): 
<br>`,ecad:"<br>http://www.ecad.eu\\nhttp://www.ecad.eu/download/ensembles/ensembles.php\\nhttp://www.ecad.eu/download/ensembles/Haylock_et_al_2008.pdf",echam5:"ECHAM5.4<br>","echam6-lr":"ECHAM6<br>Kruschke, T. et al. 2015: Probabilistic evaluation of decadal predictions of Northern Hemisphere winter storms. Meteorol. Z. (early online release) DOI: 10.1127/metz/2015/0641","ecmwf-eraint-awi-hirham5-v1":"<br>","ecmwf-eraint-clmcom-cclm4-8-17-clm3-5-v1":"Climate Limited-area Modelling Community (CLM-Community)<br>http://cordex.clm-community.eu","ecmwf-eraint-clmcom-cclm4-8-17-v1":"CLMcom-CCLM4-8-17<br>http://www.clm-community.eu/","ecmwf-eraint-cnrm-aladin52-v1":"<br>http://www.cnrm-game.fr/spip.php?rubrique69&lang=en","ecmwf-eraint-cnrm-aladin53-v1":"<br>http://www.cnrm-game.fr/spip.php?rubrique69&lang=en","ecmwf-eraint-dhmz-regcm4-2-v1":"<br>http://gforge.ictp.it/gf/project/regcm/","ecmwf-eraint-dmi-hirham5-v1":"HIRHAM<br>","ecmwf-eraint-dmi-hirham5-v2":"<br>","ecmwf-eraint-hms-aladin52-v1":"<br>http://www.met.hu/en/RCM","ecmwf-eraint-ictp-regcm4-3-v1":"RegCM Model output file<br>http://gforge.ictp.it/gf/project/regcm","ecmwf-eraint-ipsl-ineris-wrf331f-v1":"WRF 3.3.1 F<br>Vautard, R. et al., 2012 : The simulation of European heat waves from an ensemble of 1 regional climate models within the EURO-CORDEX project. Climate Dynamics, doi:10.1007/s00382-01-1714-z Jacob, D. et al., 2013, EURO-CORDEX: New high-resolution climate change projections for European impact research, Regional Environmental change, doi:10.1007/s10113-013-0499-2","ecmwf-eraint-knmi-racmo22e-v1":"<br>http://www.knmi.nl/research/regional_climate","ecmwf-eraint-knmi-racmo22t-v1":"<br>http://www.knmi.nl/research/regional_climate","ecmwf-eraint-mohc-hadgem3-ra-v1":"MOHC Unified Model for regional analysis<br>'W. Moufouma-Okia and R. Jones (2015) Resolution dependence in simulating the African hydroclimate with the HadGEM3-RA regional climate model Climate Dynamics, Volume 44, Issue 3-4, pp 609-632, February 2015'","ecmwf-eraint-mohc-hadrm3p-v1":"Hadley Centre Coupled Model Version 3 - HadCM3 (Gordon et. al 2000) atmosphere: HadAM3<br>'Gordon, C., C. Cooper, C.A. Senior, H.T. Banks, J.M. Gregory, T.C. Johns, J.F.B Mitchell and R.A. Wood (2000) The simulation of SST, sea ice extents and ocean heat transports in a version of the Hadley Centre coupled model without flux adjustments. Climate Dynamics, 16, pp147-168. Jones RG, Noguer M, Hasell DC, Hudson DA, Wilson SS, Jenkins GJ and Mitchell JFB, 200. Generating High Resolution Climate Change Scenarios usign PRECIS. Met Office Hadley Centre, Exeter, UK, 40pp, April 2004. http;//www.metoffice.gov.uk/media/pdf/6/5/PRECIS_Handbook.pdf'","ecmwf-eraint-mpi-csc-remo2009-v1":"MPI-CSC-REMO2009<br>http://www.remo-rcm.de/","ecmwf-eraint-rmib-ugent-alaro-0-v1":"<br>http://dx.doi.org/10.5194/gmd-9-1143-2016","ecmwf-eraint-smhi-rca4-v1":"<br>http://www.smhi.se/en/Research/Research-departments/climate-research-rossby-centre",en:"<br>","fgoals-f3-l":`FGOALS-f3-L (2017): 
<br>Model described by He Bian et al.,Bao Qing et al.`,"fgoals-g2":"FGOALS-g2 2011 atmosphere: GAMIL (gamil2, 128x60L26); ocean: LICOM (licom2, 360x196L30); ice: CICE (cice4_lasg, 360x196L4); land: CLM (clm3, 128x60)<br>http://www.lasg.ac.cn/FGOALS/CMIP5","fgoals-gl":"FGOALS-gl; atmosphere: GAMIL (gamil1, 72x40L26); ocean: LICOM (licom1, 360x170L30); ice: CSIM (csim4, 360x196); land: CLM (clm2, 72x40)<br>ZHOU Tianjun, WU Bo, WEN Xinyu, LI Lijuan, WANG Bin, 2008, A Fast Version of LASG/IAP Climate System Model and Its 1000-year Control Integration, Advances in Atmospheric Sciences, 25(4), 655-672","fgoals-s2":"FGOALS-s2 SAMIL 2-4-7<br>http://www.lasg.ac.cn/FGOALS/CMIP5","fio-esm":"FIO-ESM<br>","fub-dwd":"GlobVapour - 2011, satellite observations, Total Column Water Vapour Monthly Mean from SSMI+MERIS, version v1-00<br>GlobVapour web site (www.globvapour.info)",gcfs2:"ECHAM6<br>","gcfs2--episodes2018--hyr-12km":"DWD-EPISODES2018<br>http://www.dwd.de/episodes","geos-5":"MERRA Monthly 0.25x0.25 degree merged<br>/","gfdl-am4":"GFDL-AM4<br>see further_info_url attribute","gfdl-cm2p1":"GFDL-CM2p1 2010 ocean: MOM4 (MOM4p1_x1_Z50_cCM2M,Tripolar360x200L50); atmosphere: AM2 (AM2p14,M45L24); sea ice: SIS (SISp2,Tripolar360x200L50); land: LM2 (LM2,M45)<br>The GFDL Data Portal (http://nomads.gfdl.noaa.gov/) provides access to NOAA/GFDL's publicly available model input and output data sets. From this web site one can view and download data sets and documentation, including those related to the GFDL coupled models experiments run for the IPCC's 5th Assessment Report and the US CCSP.","gfdl-cm3":"GFDL-CM3 2010 atmosphere: AM3 (AM3p9,C48L48); sea ice: SIS (SISp2,Tripolar360x200); land: LM3 (LM3p7_cCM3,C48); ocean: MOM4 (MOM4p1_x1_Z50_cCM3,Tripolar360x200L50)<br>The GFDL Data Portal (http://nomads.gfdl.noaa.gov/) provides access to NOAA/GFDL's publicly available model input and output data sets. From this web site one can view and download data sets and documentation, including those related to the GFDL coupled models experiments run for the IPCC's 5th Assessment Report and the US CCSP.","gfdl-cm4":"GFDL-CM4<br>see further_info_url attribute","gfdl-dp":"<br>","gfdl-esm2g":"GFDL-ESM2G 2010 ocean: TOPAZ (TOPAZ1p2,Tripolar360x210L63); atmosphere: AM2 (AM2p14,M45L24); sea ice: SIS (SISp2,Tripolar360x210L63); land: LM3 (LM3p7_cESM,M45)<br>The GFDL Data Portal (http://nomads.gfdl.noaa.gov/) provides access to NOAA/GFDL's publicly available model input and output data sets. From this web site one can view and download data sets and documentation, including those related to the GFDL coupled models experiments run for the IPCC's 5th Assessment Report and the US CCSP.","gfdl-esm2m":"GFDL-ESM2M 2010 ocean: MOM4 (MOM4p1_x1_Z50_cCM2M,Tripolar360x200L50); atmosphere: AM2 (AM2p14,M45L24); sea ice: SIS (SISp2,Tripolar360x200L50); land: LM3 (LM3p7_cESM,M45)<br>The GFDL Data Portal (http://nomads.gfdl.noaa.gov/) provides access to NOAA/GFDL's publicly available model input and output data sets. From this web site one can view and download data sets and documentation, including those related to the GFDL coupled models experiments run for the IPCC's 5th Assessment Report and the US CCSP.","gfdl-hiram-c180":"GFDL-HIRAM-C180 2010 atmosphere: HIRAM (HIRAMp1,C180L32); land: LM3 (LM3p7_cHIRAM,C180)<br>The GFDL Data Portal (http://nomads.gfdl.noaa.gov/) provides access to NOAA/GFDL's publicly available model input and output data sets. From this web site one can view and download data sets and documentation, including those related to the GFDL coupled models experiments run for the IPCC's 5th Assessment Report and the US CCSP.","gfdl-hiram-c360":"GFDL-HIRAM-C360 2010 atmosphere: HIRAM (HIRAMp1,C360L32); land: LM3 (LM3p7_cHIRAM,C360)<br>The GFDL Data Portal (http://nomads.gfdl.noaa.gov/) provides access to NOAA/GFDL's publicly available model input and output data sets. From this web site one can view and download data sets and documentation, including those related to the GFDL coupled models experiments run for the IPCC's 5th Assessment Report and the US CCSP.","gfdl-hs":"<br>",gfs:"<br>Kruschke, T. et al., Evaluating decadal predictions of northern hemispheric cyclone frequencies, Tellus A 66, 22830, doi: 10.3402/tellusa.v66.22830; Murray, R.J. & Simmonds, I., A numerical scheme for tracking cyclone centres from digital data. Part I: Development and operation of the scheme, Aust. Meteor. Mag., 1991, 39, 155166; Pinto, J.G. et al., Sensitivities of a cyclone detection and tracking algorithm: individual tracks and climatology Meteor. Z., 2005, 14, 823-838",gfz:"GRACE (since 2003)<br>","giss-e2-1-g":`GISS-E2.1G (2016): 
<br>https://data.giss.nasa.gov/modelE/cmip6`,"giss-e2-h":"GISS-E2-H-Eh135f9a Atmosphere: GISS-E2; Ocean: H<br>www.giss.nasa.gov/research/modeling","giss-e2-h-cc":"GISS-E2-H-CC-E4arobio_h8RCP85 Atmosphere: GISS-E2; Ocean: R<br>http://data.giss.nasa.gov/modelE/ar5","giss-e2-r":"GISS-E2-R-E137Tcadf9fF40oQ32 Atmosphere: GISS-E2; Ocean: R<br>http://data.giss.nasa.gov/modelE/ar5","giss-e2-r-cc":"GISS-E2-R-CC-E4arobio_g8RCP85 Atmosphere: GISS-E2; Ocean: R<br>http://data.giss.nasa.gov/modelE/ar5",hadcm3:"HadCM3 - Hadley Centre Coupled Model Version 3 (2000) atmosphere: HadAM3 (N48L19); ocean: HadOM (lat: 1.25 lon: 1.25 L20); land-surface/vegetation: MOSES1;<br>Collins, M., S.F.B Tett, and C. Cooper (2009) the Internal Climate Variability of HadCM3, a Version of the Hadley Centre Coupled Model without Flux Adjustments. Climate Dynamics, 17 (1): 61-81.; Smith, D.M., S. Cusack, A.W. Colman, C.K. Folland, G.R. Harris and J.M. Murphy (2009) Improved surface temperature prediction for the coming decade from a global climate model. Science, 317, 796-799; Smith, D.M., R. Eade, N.J. Dunstone, D. Fereday, J.M. Murphy, H. Pohlman, and A.A. Scaife (2010) Skilful multi-year predictions of Atlantic hurricane frequency. Nature Geoscience, DOI: 10.1038/NGE01004;",hadcrut4:"CRUTEM.4.2.0.0, HadSST.3.1.0.0<br>","hadcrut4-median":"CRUTEM.4.6.0.0, HadSST.3.1.1.0<br>","hadgem2-a":"HadGEM2-A (2009) atmosphere: HadGAM2 (N96L38); land-surface/vegetation: MOSES2<br>Bellouin N. et al, (2007) Improved representation of aerosols for HadGEM2. Meteorological Office Hadley Centre, Technical Note 73, March 2007; Collins W.J. et al, (2008) Evaluation of the HadGEM2 model. Meteorological Office Hadley Centre, Technical Note 74,; Johns T.C. et al, (2006) The new Hadley Centre climate model HadGEM1: Evaluation of coupled simulations. Journal of Climate, American Meteorological Society, Vol. 19, No. 7, pages 1327-1353.; Martin G.M. et al, (2006) The physical properties of the atmosphere in the new Hadley Centre Global Environmental Model, HadGEM1 - Part 1: Model description and global climatology. Journal of Climate, American Meteorological Society, Vol. 19, No.7, pages 1274-1301.; Ringer M.A. et al, (2006) The physical properties of the atmosphere in the new Hadley Centre Global Environmental Model, HadGEM1 - Part 2: Aspects of variability and regional climate. Journal of Climate, American Meteorological Society, Vol. 19, No. 7, pages 1302-1326.","hadgem2-ao":"HadGEM2-AO r6.6.3 (2010): atmosphere: HadGAM (HadGAM2, N96L38); ocean: HadGOM (HadGOM2, 1x1L40, increased res at Equator); sea ice: part of HadGOM2; land: MOSES-2<br>Model described by Johns et al. (J. Clim., 2006, 1327-1353). Also see http://www.metoffice.gov.uk/research/hadleycentre/pubs/HCTN/HCTN_54.pdf transient experiments described in Stott et al. (J. Clim., 2006, 2763-2782.)","hadgem2-cc":"HadGEM2-CC (2011) atmosphere: HadGAM2(N96L60); ocean: HadGOM2 (lat: 1.0-0.3 lon: 1.0 L40); land-surface/vegetation: MOSES2 and TRIFFID; ocean biogeochemistry: diat-HadOCC<br>Jones, C.D. et al. (2011) The HadGEM2-ES implementation of CMIP5 centennial simulations. Geosci. Model Dev., 4, 543-570, http://www.geosci-model-dev.net/4/543/2011/gmd-4-543-2011.html; Martin G.M. et al. (2011) The HadGEM2 family of Met Office Unified Model climate configurations, Geosci. Model Dev., 4, 723-757, http://www.geosci-model-dev.net/4/723/2011/gmd-4-723-2011.html; Collins, W.J. et al. (2011) Development and evaluation of an Earth-system model - HadGEM2, Geosci. Model Dev. Discuss., 4, 997-1062, http://www.geosci-model-dev-discuss.net/4/997/2011/gmdd-4-997-2011.html","hadgem2-es":"HadGEM2-ES (2009) atmosphere: HadGAM2 (N96L38); ocean: HadGOM2 (lat: 1.0-0.3 lon: 1.0 L40); land-surface/vegetation: MOSES2 and TRIFFID; tropospheric chemistry: UKCA; ocean biogeochemistry: diat-HadOCC<br>Bellouin N. et al, (2007) Improved representation of aerosols for HadGEM2. Meteorological Office Hadley Centre, Technical Note 73, March 2007; Collins W.J. et al, (2008) Evaluation of the HadGEM2 model. Meteorological Office Hadley Centre, Technical Note 74,; Johns T.C. et al, (2006) The new Hadley Centre climate model HadGEM1: Evaluation of coupled simulations. Journal of Climate, American Meteorological Society, Vol. 19, No. 7, pages 1327-1353.; Martin G.M. et al, (2006) The physical properties of the atmosphere in the new Hadley Centre Global Environmental Model, HadGEM1 - Part 1: Model description and global climatology. Journal of Climate, American Meteorological Society, Vol. 19, No.7, pages 1274-1301.; Ringer M.A. et al, (2006) The physical properties of the atmosphere in the new Hadley Centre Global Environmental Model, HadGEM1 - Part 2: Aspects of variability and regional climate. Journal of Climate, American Meteorological Society, Vol. 19, No. 7, pages 1302-1326.","harmonie-aladin":"<br>",hyras:"surface observation<br>Datenlieferung2018_AP101b_HYRAS-TAS.pdf","ichec-ec-earth-clmcom-cclm4-8-17-clm3-5-v1":"Climate Limited-area Modelling Community (CLM-Community)<br>http://cordex.clm-community.eu","ichec-ec-earth-clmcom-cclm4-8-17-v1":"CLMcom-CCLM4-8-17<br>http://www.clm-community.eu/","ichec-ec-earth-dmi-hirham5-v1":"HIRHAM<br>","ichec-ec-earth-dmi-hirham5-v2":"<br>","ichec-ec-earth-gerics-remo2015-v1":"<br>","ichec-ec-earth-knmi-racmo22e-v1":"<br>http://www.knmi.nl/research/regional_climate","ichec-ec-earth-knmi-racmo22t-v1":"<br>http://www.knmi.nl/research/regional_climate","ichec-ec-earth-mpi-csc-remo2009-v1":"MPI-CSC-REMO2009<br>http://www.remo-rcm.de/","ichec-ec-earth-smhi-rca4-v1":"<br>http://www.smhi.se/en/Research/Research-departments/climate-research-rossby-centre",ifs:"Integrated Forecasting System (IFS)<br>Kruschke, T. et al., Evaluating decadal predictions of northern hemispheric cyclone frequencies, Tellus A 66, 22830, doi: 10.3402/tellusa.v66.22830; Murray, R.J. & Simmonds, I., A numerical scheme for tracking cyclone centres from digital data. Part I: Development and operation of the scheme, Aust. Meteor. Mag., 1991, 39, 155166; Pinto, J.G. et al., Sensitivities of a cyclone detection and tracking algorithm: individual tracks and climatology Meteor. Z., 2005, 14, 823-838","ifs-c43r3":"<br>","ifs-c43r3@36km":"<br>","ifs-c43r3@ambav-v10t01":"<br>","ifs-c45r1@ambav-v10t01":"<br>",inmcm4:"inmcm4 (2009)<br>Volodin, Diansky, Gusev 2010. Climate model INMCM4.0. Izvestia RAS. Atmospheric and oceanic physics, V.46, N4, in print.",ipsl:"CNRS/CNES L3 dataset producted by IPSL/ICARE from NASA CALIOP Level 1 data<br>H. Chepfer, S.Bony, D. M. Winker, G. Cesana, JL. Dufresne, P. Minnis, C.J. Stubenrauch, S. Zeng, 2009 'The GCM OrientedCALIPSO Cloud Product (CALIPSO-GOCCP)', J. Geophys. Res., 105, D00H16, doi:10.1029/2009JD012251, [http://www.agu.org/journals/jd/jd1005/2009JD012251/|http://www.agu.org/journals/jd/jd1005/2009JD012251/]","ipsl-cm5a-lr":"IPSL-CM5A-LR (2010) : atmos : LMDZ4 (LMDZ4_v5, 96x95x39); ocean : ORCA2 (NEMOV2_3, 2x2L31); seaIce : LIM2 (NEMOV2_3); ocnBgchem : PISCES (NEMOV2_3); land : ORCHIDEE (orchidee_1_9_4_AR5)<br>Model documentation and further reference available here : http://icmc.ipsl.fr","ipsl-cm5a-mr":"IPSL-CM5A-MR (2010) : atmos : LMDZ4 (LMDZ4_v5, 144x143x39); ocean : ORCA2 (NEMOV2_3, 2x2L31); seaIce : LIM2 (NEMOV2_3); ocnBgchem : PISCES (NEMOV2_3); land : ORCHIDEE (orchidee_1_9_4_AR5)<br>Model documentation and further reference available here : http://icmc.ipsl.fr","ipsl-cm5b-lr":"IPSL-CM5B-LR (2011) : atmos : LMDZ5 (LMDZ5_NPv3.1, 96x95x39); ocean : ORCA2 (NEMOV2_3, 2x2L31); seaIce : LIM2 (NEMOV2_3); ocnBgchem : PISCES (NEMOV2_3); land : ORCHIDEE (orchidee_1_9_4_AR5)<br>Model documentation and further reference available here : http://icmc.ipsl.fr","ipsl-cm6a-atm-hr":"IPSL-CM6A-ATM-HR (2018): atmos: LMDZ (NPv6, N256; 512 x 360 longitude/latitude; 79 levels; top level 40000 m) land: ORCHIDEE (v2.0, Water/Carbon/Energy mode)<br>","ipsl-cm6a-lr":"IPSL-CM6A-LR (2017): atmos: LMDZ (NPv6, N96; 144 x 143 longitude/latitude; 79 levels; top level 40000 m) land: ORCHIDEE (v2.0, Water/Carbon/Energy mode) ocean: NEMO-OPA (eORCA1.3, tripolar primarily 1deg; 362 x 332 longitude/latitude; 75 levels; top grid cell 0-2 m) ocnBgchem: NEMO-PISCES seaIce: NEMO-LIM3<br>","ipsl-dp":"<br>","ipsl-hs":"IPSL-CM5A-LR (2010) : atmos : LMDZ4 (LMDZ4_v5, 96x95x39); ocean : ORCA2 (NEMOV2_3, 2x2L31); seaIce : LIM2 (NEMOV2_3); ocnBgchem : PISCES (NEMOV2_3); land : ORCHIDEE (orchidee_1_9_4_AR5)<br>Model documentation and further reference available here : http://icmc.ipsl.fr","ipsl-ipsl-cm5a-lr-gerics-remo2009-v1":"GERICS-REMO2009<br>http://www.remo-rcm.de/","ipsl-ipsl-cm5a-mr-ipsl-ineris-wrf331f-v1":"WRF 3.3.1 F<br>Vautard, R. et al., 2012 : The simulation of European heat waves from an ensemble of 1 regional climate models within the EURO-CORDEX project. Climate Dynamics, doi:10.1007/s00382-01-1714-z Jacob, D. et al., 2013, EURO-CORDEX: New high-resolution climate change projections for European impact research, Regional Environmental change, doi:10.1007/s10113-013-0499-2","ipsl-ipsl-cm5a-mr-smhi-rca4-v1":"<br>http://www.smhi.se/en/Research/Research-departments/climate-research-rossby-centre",jcdas:"Japanese 25-year ReAnalysis, anl_p, Monthly mean<br>http://www.jma.go.jp/jma/indexe.html",loa_ipsl:"CNRS/CNES L3 dataset producted by LOA/IPSL/ICARE from CNES Level 1 data<br>","miroc-esm":"MIROC-ESM 2010 atmosphere: MIROC-AGCM (MIROC-AGCM 2010, T42L80); ocean: COCO (COCO3.4, 256x192 L44); sea ice: COCO (COCO3.4); land: MATSIRO (MATSIRO, L6); aerosols: SPRINTARS (SPRINTARS 5.00, T42L80):ocean-biogeochemistry: NPZD ; land-biogeochemistry: SEIB-DGVM (SEIB-DGVM, T42)<br>Watanabe et al., : MIROC-ESM: model description and basic results of CMIP5-20c3m experiments, Geosci. Model Dev. Discuss., 4, 1063-1128, doi:10.5194/gmdd-4-1063-2011, 2011.","miroc-esm-chem":"MIROC-ESM-CHEM 2010 atmosphere: MIROC-AGCM (MIROC-AGCM 2010, T42L80); ocean: COCO (COCO3.4, 256x192 L44); sea ice: COCO (COCO3.4); land: MATSIRO (MATSIRO, L6); aerosols: SPRINTARS (SPRINTARS 5.00, T42L80):atmospheric-chemistry: CHASER (CHASER 4.1, T42L80): ocean-biogeochemistry: NPZD ; land-biogeochemistry: SEIB-DGVM (SEIB-DGVM, T42)<br>Watanabe et al., : MIROC-ESM: model description and basic results of CMIP5-20c3m experiments, Geosci. Model Dev. Discuss., 4, 1063-1128, doi:10.5194/gmdd-4-1063-2011, 2011.","miroc-miroc5-gerics-remo2009-v1":"GERICS-REMO2009<br>http://www.remo-rcm.de/","miroc-miroc5-gerics-remo2015-v1":"<br>","miroc-miroc5-smhi-rca4-v1":"<br>http://www.smhi.se/en/Research/Research-departments/climate-research-rossby-centre",miroc4h:"MIROC4h 2009 atmosphere: AGCM (AGCM5.8, T213L56); ocean: COCO (COCO3.4, rotated pole 1280x912 L48); sea ice: COCO (COCO3.4); land: MATSIRO (MATSIRO, 2x3L5)<br>Sakamoto et al., 2011: MIROC4h -- a new high-resolution atmosphere-ocean coupled general circulation model. (in preparation); Tatebe et al., 2011: (in preparation)",miroc5:"MIROC5 2010 atmosphere: MIROC-AGCM6 (T85L40); ocean: COCO (COCO4.5, 256x224 L50); sea ice: COCO (COCO4.5); land: MATSIRO (MATSIRO, L6); aerosols: SPRINTARS (SPRINTARS 5.00, T85L40)<br>Watanabe et al., 2010: Improved climate simulation by MIROC5: Mean states, variability, and climate sensitivity. J. Climate, 23, 6312-6335",miroc6:`MIROC6 (2017): 
<br>`,mitgcm:"<br>",model:"MPI-ESM-LR 2011; URL: http://svn.zmaw.de/svn/cosmos/branches/releases/mpi-esm-cmip5/src/mod; atmosphere: ECHAM6 (REV: 5969), T63L47; land: JSBACH (REV: 5969); ocean: MPIOM (REV: 5969), GR15L40; sea ice: 5969; marine bgc: HAMOCC (REV: 5969);<br>ECHAM6: n/a; JSBACH: Raddatz et al., 2007. Will the tropical land biosphere dominate the climate-carbon cycle feedback during the twenty first century? Climate Dynamics, 29, 565-574, doi 10.1007/s00382-007-0247-8; MPIOM: Marsland et al., 2003. The Max-Planck-Institute global ocean/sea ice model with orthogonal curvilinear coordinates. Ocean Modelling, 5, 91-127; HAMOCC: Technical Documentation, http://www.mpimet.mpg.de/fileadmin/models/MPIOM/HAMOCC5.1_TECHNICAL_REPORT.pdf;","modini-mpi-esm-lr-ws":"ECHAM6.1<br>",mohc:"CRUTEM.4.5.0.0, HadSST.3.1.1.0<br>Donat, M. G., et al. (2013), Updated analyses of temperature and precipitation extreme indices since the beginning of the twentieth century: The HadEX2 dataset, J. Geophys. Res. Atmos., 118, doi:10.1002/jgrd.50150","mohc-cru":"HadSST.3.1.1.0<br>Kennedy J.J., Rayner, N.A., Smith, R.O., Saunby, M. and Parker, D.E. (2011). Reassessing biases and other uncertainties in sea-surface temperature observations since 1850 part 2: biases and homogenisation. in press JGR Atmospheres","mohc-hadgem2-es-clmcom-cclm4-8-17-v1":"CLMcom-CCLM4-8-17<br>http://www.clm-community.eu/","mohc-hadgem2-es-gerics-remo2009-v1":"GERICS-REMO2009<br>http://www.remo-rcm.de/","mohc-hadgem2-es-gerics-remo2015-v1":"<br>","mohc-hadgem2-es-ictp-regcm4-3-v1":"RegCM Model output file<br>http://gforge.ictp.it/gf/project/regcm","mohc-hadgem2-es-knmi-racmo22e-v2":"<br>http://www.knmi.nl/research/regional_climate","mohc-hadgem2-es-knmi-racmo22t-v2":"<br>http://www.knmi.nl/research/regional_climate","mohc-hadgem2-es-smhi-rca4-v1":"<br>http://www.smhi.se/en/Research/Research-departments/climate-research-rossby-centre","mpi-esm-hr":"MPI-ESM-HR 2011; URL: http://svn.zmaw.de/svn/cosmos/branches/releases/mpi-esm-cmip5/src/mod; atmosphere: ECHAM6 (REV: 8824), T63L47; land: JSBACH (REV: 8824); ocean: MPIOM (REV: 8824), GR15L40; sea ice: 8824; marine bgc: HAMOCC (REV: 8824);<br>ECHAM6: n/a; JSBACH: Raddatz et al., 2007. Will the tropical land biosphere dominate the climate-carbon cycle feedback during the twenty first century? Climate Dynamics, 29, 565-574, doi 10.1007/s00382-007-0247-8; MPIOM: Marsland et al., 2003. The Max-Planck-Institute global ocean/sea ice model with orthogonal curvilinear coordinates. Ocean Modelling, 5, 91-127; HAMOCC: Technical Documentation, http://www.mpimet.mpg.de/fileadmin/models/MPIOM/HAMOCC5.1_TECHNICAL_REPORT.pdf;","mpi-esm-lr":"<br>","mpi-esm-mr":"MPI-ESM-MR 2011; URL: http://svn.zmaw.de/svn/cosmos/branches/releases/mpi-esm-cmip5/src/mod; atmosphere: ECHAM6 (REV: 5051), T63L47; land: JSBACH (REV: 5051); ocean: MPIOM (REV: 5051), GR15L40; sea ice: 5051; marine bgc: HAMOCC (REV: 5051);<br>ECHAM6: n/a; JSBACH: Raddatz et al., 2007. Will the tropical land biosphere dominate the climate-carbon cycle feedback during the twenty first century? Climate Dynamics, 29, 565-574, doi 10.1007/s00382-007-0247-8; MPIOM: Marsland et al., 2003. The Max-Planck-Institute global ocean/sea ice model with orthogonal curvilinear coordinates. Ocean Modelling, 5, 91-127; HAMOCC: Technical Documentation, http://www.mpimet.mpg.de/fileadmin/models/MPIOM/HAMOCC5.1_TECHNICAL_REPORT.pdf;","mpi-esm-p":"MPI-ESM-P 2011; URL: http://svn.zmaw.de/svn/cosmos/branches/releases/mpi-esm-cmip5/src/mod; atmosphere: ECHAM6 (REV: 5234), T63L47; land: JSBACH (REV: 5234);<br>ECHAM6: n/a; JSBACH: Raddatz et al., 2007. Will the tropical land biosphere dominate the climate-carbon cycle feedback during the twenty first century? Climate Dynamics, 29, 565-574, doi 10.1007/s00382-007-0247-8;","mpi-m":"HOAPS-S Hamburg Ocean-Atmosphere Parameters and fluxes from Satellite, March 2011, Release 3.2<br>http://www.hoaps.org/","mpi-m-mpi-esm-lr-awi-hirham5-v2":"<br>","mpi-m-mpi-esm-lr-clmcom-cclm4-8-17-clm3-5-v1":"Climate Limited-area Modelling Community (CLM-Community)<br>http://cordex.clm-community.eu","mpi-m-mpi-esm-lr-clmcom-cclm4-8-17-v1":"CLMcom-CCLM4-8-17<br>http://www.clm-community.eu/","mpi-m-mpi-esm-lr-mpi-csc-remo2009-v1":"MPI-CSC-REMO2009<br>http://www.remo-rcm.de/","mpi-m-mpi-esm-lr-smhi-rca4-v1":"<br>http://www.smhi.se/en/Research/Research-departments/climate-research-rossby-centre","mpi-m-mpi-esm-lr-smhi-rca4-v1a":"<br>http://www.smhi.se/en/Research/Research-departments/climate-research-rossby-centre","mpi-twc-esm-lr":"MPI-TWC-ESM-LR 2015; URL: https://svn.zmaw.de/svn/cosmos/branches/mpiesm-1.0.00-oa3mct/src/mod; atmosphere: ECHAM6 (echam-6.1.00), T63L47; land: JSBACH (jsbach-2.01); ocean: MPIOM (mpiom-1.5.1p1 incl. marine biogeochemistry HAMOCC), GR15L40;<br>ECHAM6: n/a; JSBACH: Raddatz et al., 2007. Will the tropical land biosphere dominate the climate-carbon cycle feedback during the twenty first century? Climate Dynamics, 29, 565-574, doi 10.1007/s00382-007-0247-8; MPIOM: Marsland et al., 2003. The Max-Planck-Institute global ocean/sea ice model with orthogonal curvilinear coordinates. Ocean Modelling, 5, 91-127; HAMOCC: Technical Documentation, http://www.mpimet.mpg.de/fileadmin/models/MPIOM/HAMOCC5.1_TECHNICAL_REPORT.pdf;","mpi1.2-esm-lr":"<br>HIGH","mpiesm-lr":"<br>","mri-agcm3-2h":"MRI-AGCM3-2H 2009 (gsmuv-091102, TL319L64)<br>Model described by Mizuta et al. (Journal of the Meteorological Society of Japan, 2012)","mri-agcm3-2s":"MRI-AGCM3-2S 2009 (gsmuv-091102, TL959L64)<br>Model described by Mizuta et al. (Journal of the Meteorological Society of Japan, 2011, submitted)","mri-cgcm3":"MRI-CGCM3 2011 atmosphere: GSMUV (gsmuv-110120o_cosp110519, TL159L48); land: HAL (HAL_cmip5_v0.31_04); aerosol: MASINGAR-mk2 (masingar_mk2-20110111, TL95L48)<br>Model described by Yukimoto et al. (Technical Report of the Meteorological Research Institute, 2011, 64, 83pp.)","mri-esm1":"MRI-ESM1 2011 atmosphere: GSMUV (gsmuv-110120oc, TL159L48); ocean: MRI.COM3 (MRICOM-3_0-20101222, 1x0.5L51); sea ice: MRI.COM3; land: HAL (HAL_cmip5_v0.31_04); aerosol: MASINGAR-mk2 (masingar_mk2-20110111_1121, TL95L48); ozone: MRI-CCM2 (ccm2-101129, T42L48)<br>Model described by Yukimoto et al. (Technical Report of the Meteorological Research Institute, 2011, 64, 83pp.)","mri-esm2-0":`MRI-ESM2.0 (2017): 
<br>`,"nasa-gsfc":"TRMM Tropical Rainfall Measuring Mission 3B43 Monthly 0.25x0.25 degree merged<br>http://science.nasa.gov/missions/trmm/","nasa-jpl":"TES (Tropospheric Emission Spectrometer), averaged, interpolated and converted from TES L2 monthly HDF-EOS files, version 4 (R11)<br>'Level 3 Algorithms, Requirements, & Products' available at http://eosweb.larc.nasa.gov/PRODOCS/tes/table_tes.html","nasa-larc":"CERES retrievals<br>http://ceres.larc.nasa.gov/cmip5_data.php","ncc-noresm1-m-dmi-hirham5-v1":"<br>","ncc-noresm1-m-smhi-rca4-v1":"<br>http://www.smhi.se/en/Research/Research-departments/climate-research-rossby-centre",nemo:"<br>http://www.ecmwf.int/products/forecasts/d/charts/oras4/","nicam-09":"NICAM.09 2009 atmosphere: NICAM (gl09=14km)<br>Model described by Tomita and Satoh (Fluid. Dyn. Res., 2004, 357-400) and Satoh et al. (J. Comput. Phys., 2008, 3486-3514). For simulation, see Yoshizaki et al (SOLA, accepted) and Yoshizaki et al. (SOLA, 2012, 33-36).",noaa:"<br>","noaa-gfdl-gfdl-esm2g-gerics-remo2009-v1":"GERICS-REMO2009<br>http://www.remo-rcm.de/","noaa-gfdl-gfdl-esm2m-smhi-rca4-v1":"<br>http://www.smhi.se/en/Research/Research-departments/climate-research-rossby-centre","noaa-ncdc":"<br>",noaancepcpc:"<br>",nodc:"<br>Levitus, S., J. I. Antonov, T. P. Boyer, O. K. Baranova, H. E. Garcia, R. A. Locarnini, A.V. Mishonov, J. R. Reagan, D. Seidov, E. S. Yarosh, M. M. Zweng, 2012: World Ocean heat content and thermosteric sea level change (0-2000 m) 1955-2010. Geophys. Res. Lett. , 39, L10603, doi:10.1029/2012GL051106","noresm1-m":"NorESM1-M 2011 atmosphere: CAM-Oslo (CAM4-Oslo-noresm-ver1_cmip5-r112, f19L26); ocean: MICOM (MICOM-noresm-ver1_cmip5-r112, gx1v6L53); sea ice: CICE (CICE4-noresm-ver1_cmip5-r112); land: CLM (CLM4-noresm-ver1_cmip5-r112)<br>","noresm1-me":"NorESM1-ME 2011 atmosphere: CAM-Oslo (CAM4-Oslo-noresm-ver1_cmip5-r139, f19L26); ocean: MICOM (MICOM-noresm-ver1_cmip5-r139, gx1v6L53); ocean biogeochemistry: HAMOCC (HAMOCC-noresm-ver1_cmip5-r139, gx1v6L53); sea ice: CICE (CICE4-noresm-ver1_cmip5-r139); land: CLM (CLM4-noresm-ver1_cmip5-r139)<br>",pcmdi:"<br>http://climserv.ipsl.polytechnique.fr/cfmip-obs/",pr:"<br>",reanalysis:"<br>","remo-2009":"REMO-2009<br>","remo-h":"<br>","remo-o1":"<br>","remo-o2":"<br>",remo2011:"REMO2011<br>",remss:"Sea Surface Temperature from AMSR-E onboard AQUA.<br>",ssaltoduacs:"<br>www.aviso.altimetry.fr",sst:"HadISST<br>Rayner, N. A., Parker, D. E., Horton, E. B., Folland, C. K., Alexander, L. V., Rowell, D. P., Kent, E. C., Kaplan, A. Global analyses of sea surface temperature, sea ice, and night marine air temperature since the late nineteenth century J. Geophys. Res.Vol. 108, No. D14, 4407 10.1029/2002JD002670",t2m:"CRUTEM.4.3.0.0, HadSST.3.1.1.0<br>",test:"<br>",ub:"satellite SMOS<br>Huntemann, M., Heygster, G., Kaleschke, L., Krumpen, T., M\xC3\xA4kynen, M., & Drusch, M. (2014). Empirical sea ice thickness retrieval during the freeze-up period from SMOS high incident angle observations. The Cryosphere, 8(2), 439\xE2\x80\x93451. doi:10.5194/tc-8-439-2014",uea:"Run ID = 1709081022. Data generated from:tmp.1708041519.dtb<br>Information on the data is available at http://badc.nerc.ac.uk/data/cru/","ukesm1-0-ll":`UKESM1.0-LL (2018): 
<br>`,uoc:"<br>",uor:"<br>","wrf-3-5-1":"WRF-3-5-1<br>"}),product:Object.freeze({aerchemmip:"Aerosols and Chemistry Model Intercomparison Project",c4mip:"Coupled Climate Carbon Cycle Model Intercomparison Project",cdrmip:"Carbon Dioxide Removal Model Intercomparison Project",cfmip:"Cloud Feedback Model Intercomparison Project",cmip:"CMIP DECK: 1pctCO2, abrupt4xCO2, amip, esm-piControl, esm-historical, historical, and piControl experiments",cordex:"Coordinated Regional Climate Downscaling Experiment",damip:"Detection and Attribution Model Intercomparison Project",dcpp:"Decadal Climate Prediction Project",dynvarmip:"Dynamics and Variability Model Intercomparison Project",fafmip:"Flux-Anomaly-Forced Model Intercomparison Project",geomip:"Geoengineering Model Intercomparison Project",gmmip:"Global Monsoons Model Intercomparison Project",highresmip:"High-Resolution Model Intercomparison Project",ismip6:"Ice Sheet Model Intercomparison Project for CMIP6",ls3mip:"Land Surface, Snow and Soil Moisture",lumip:"Land-Use Model Intercomparison Project",omip:"Ocean Model Intercomparison Project",pamip:"Polar Amplification Model Intercomparison Project",pmip:"Palaeoclimate Modelling Intercomparison Project",rfmip:"Radiative Forcing Model Intercomparison Project",scenariomip:"Scenario Model Intercomparison Project",simip:"Sea Ice Model Intercomparison Project",viacsab:"Vulnerability, Impacts, Adaptation and Climate Services Advisory Board",volmip:"Volcanic Forcings Model Intercomparison Project"}),project:Object.freeze({cmip5:"Coupled Model Intercomparison Project 5",cmip6:"Coupled Model Intercomparison Project 6",cordex:"Coordinated Regional Climate Downscaling experiment",dcpp:"Decadal Climate Prediction Project: https://gmd.copernicus.org/articles/9/3751/2016/",dyamond:"DYnamics of the Atmospheric general circulation Modeled On Non-hydrostatic Domains (DYAMOND)",dyamond_winter:"DYnamics of the Atmospheric general circulation Modeled On Non-hydrostatic Domains (DYAMOND) Winter Simulations","miklip-forecast":"Mittelfristige Klimaprognosen (Medium-term Climate Predictions)","miklip-ini":"Mittelfristige Klimaprognosen (Medium-term Climate Predictions)","mpi-ge":"Max Planck Institute for Meteorology (MPI-M) Grand Ensemble (MPI-GE)",nextgems:"Next Generation Earth Modelling Systems",nukleus:"Nutzbare lokale Klimainformationen f\xC3\xBCr Deutschland (a cross-sectional activity of RegIKlim)"}),realm:Object.freeze({aerosol:"Aerosol",atmos:"Atmosphere",atmoschem:"Atmospheric Chemistry",land:"Land Surface",landice:"Land Ice",ocean:"Ocean",ocnbgchem:"Ocean Biogeochemistry",seaice:"Sea Ice"}),time_aggregation:Object.freeze({acc:"sum of values within the time period",inst:"specified time point within the time period",max:"maximum value within the time period",mean:"mean value within the time period",min:"minimum value within the time period"}),time_frequency:Object.freeze({"15min":"15 minutes mean samples","15minpt":"sampled every 15 minutes, at specified time poin within the time period","1day":"daily mean samples","1daypt":"sampled daily, at specified time point within the time period","1hr":"hourly mean samples","1hrcm":"monthly-mean diurnal cycle resolving each day into 1-hour means","1hrpt":"sampled hourly, at specified time point within the time period","1min":"1 minute mean samples","1minpt":"sampled every 1 minute, at specified time point within the time period","2min":"2 minutes mean samples","2minpt":"sampled every 2 minutes, at specified time point within the time period","3hr":"3 hourly mean samples","3hrpt":"sampled 3 hourly, at specified time point within the time period","5min":"5 minutes mean samples","5minpt":"sampled every 5 minutes, at specified time point within the time period","6hr":"6 hourly mean samples","6hrpt":"sampled 6 hourly, at specified time point within the time period",dec:"decadal mean samples",fx:"fixed (time invariant) field",mon:"monthly mean samples",monc:"monthly climatology computed from monthly mean samples",monpt:"sampled monthly, at specified time point within the time period",sem:"weekly mean samples",subhr:"sub-hourly mean samples",subhrpt:"sampled sub-hourly, at specified time point within the time period",yr:"annual mean samples",yrc:"annual climatology computed from annual mean samples",yrpt:"sampled yearly, at specified time point within the time period"}),variable:Object.freeze({"Bedeckungsgrad mittlerer Wolken (400-800 hPa)":"Medium Cloud Fraction","Bedeckungsgrad niedriger Wolken":"Low Cloud Fraction","Bedeckungsgrad niedriger Wolken (> 800 hPa)":"Low Cloud Fraction","Bias-korrigierte oberfl\xC3\xA4chennahe Lufttemperatur":"Bias-Corrected Near-Surface Air Temperature","Bias-korrigierter Niederschlag":"Bias-Corrected Precipitation","Blattfl\xC3\xA4chenindex":"Leaf Area Index",Bodendruck:"Surface Air Pressure",Bodentemperatur:"Temperature of Soil","Bodenwassergehalt Oberfl\xC3\xA4che":"Moisture in Upper Portion of Soil Column","Condensed Water Path":"Condensed Water Path","Geopotentielle H\xC3\xB6he der Druckfl\xC3\xA4che 1000hPa":"Geopotential Height at 1000 hPa","Geopotentielle H\xC3\xB6he der Druckfl\xC3\xA4che 200hPa":"Geopotential Height at 250 hPa","Geopotentielle H\xC3\xB6he der Druckfl\xC3\xA4che 250hPa":"Geopotential Height at 250 hPa","Geopotentielle H\xC3\xB6he der Druckfl\xC3\xA4che 300hPa":"Geopotential Height at 300 hPa","Geopotentielle H\xC3\xB6he der Druckfl\xC3\xA4che 500hPa":"Geopotential Height at 500 hPa","Geopotentielle H\xC3\xB6he der Druckfl\xC3\xA4che 700hPa":"Geopotential Height at 700 hPa","Geopotentielle H\xC3\xB6he der Druckfl\xC3\xA4che 850hPa":"Geopotential Height at 850 hPa","Geopotentielle H\xC3\xB6he der Druckfl\xC3\xA4che 925hPa":"Geopotential Height at 925 hPa","Geopotentielle H\xC3\xB6he der Druckfl\xC3\xA4che 950hPa":"Geopotential Height at 950 hPa",Gesamtabfluss:"Total Runoff",Gesamtbedeckungsgrad:"Total Cloud Cover Percentage","Gitterzellenfl\xC3\xA4che f\xC3\xBCr atmosph\xC3\xA4rische Gittervariablen":"Grid-Cell Area for Atmospheric Grid Variables","Grenzschichth\xC3\xB6he":"Height of Boundary Layer","H\xC3\xB6he \xC3\xBCber Meeresspiegel":"Surface Altitude","Ice Water Path":"Ice Water Path","Konvektionshemmung (CIN)":"Convective Inhibition of Mean Surface Layer Parcel","Konvektiver Niederschlag":"Convective Precipitation","Kurzwellige abw\xC3\xA4rts gerichtete Strahlungsflussdichte am Boden":"Surface Downwelling Shortwave Radiation","Kurzwellige abw\xC3\xA4rts gerichtete Strahlungsflussdichte am Oberrand (TOA)":"TOA Incident Shortwave Radiation","Kurzwellige aufw\xC3\xA4rts gerichtete Strahlungsflussdichte am Boden":"Surface Upwelling Shortwave Radiation","Kurzwellige aufw\xC3\xA4rts gerichtete Strahlungsflussdichte am Oberrand (TOA)":"TOA Outgoing Shortwave Radiation","Landanteil der Gitterbox":"Percentage of the Grid Cell Occupied by Land (Including Lakes)","Langwellige abw\xC3\xA4rts gerichtete Strahlungsflussdichte am Boden":"Surface Downwelling Longwave Radiation","Langwellige aufw\xC3\xA4rts gerichtete Strahlungsflussdichte am Boden":"Surface Upwelling Longwave Radiation","Langwellige aufw\xC3\xA4rts gerichtete Strahlungsflussdichte am Oberrand (TOA)":"TOA Outgoing Longwave Radiation","Luftdruck auf Meeresniveau":"Sea Level Pressure","Lufttemperatur auf Druckfl\xC3\xA4che 1000hPa":"Air Temperature at 1000 hPa","Lufttemperatur auf Druckfl\xC3\xA4che 200hPa":"Air Temperature at 200 hPa","Lufttemperatur auf Druckfl\xC3\xA4che 250hPa":"Air Temperature at 250 hPa","Lufttemperatur auf Druckfl\xC3\xA4che 300hPa":"Air Temperature at 300 hPa","Lufttemperatur auf Druckfl\xC3\xA4che 500hPa":"Air Temperature at 500 hPa","Lufttemperatur auf Druckfl\xC3\xA4che 700hPa":"Air Temperature at 700 hPa","Lufttemperatur auf Druckfl\xC3\xA4che 850hPa":"Air Temperature at 850 hPa","Lufttemperatur auf Druckfl\xC3\xA4che 925hPa":"Air Temperature at 925 hPa","Lufttemperatur auf Druckfl\xC3\xA4che 950hPa":"Air Temperature at 950 hPa","Lufttemperatur in 2m":"Near-Surface Air Temperature","Maximum der Lufttemperatur in 2 m":"Maximum Near-Surface Air Temperature",Meereisdicke:"Sea Ice Thickness","Meridionale Windkomponente in 10 m (S\xC3\xBCdwind)":"Northward Near-Surface Wind","Minimum der Lufttemperatur in 2 m":"Minimum Near-Surface Air Temperature",Niederschlag:"Precipitation","Oberfl\xC3\xA4chenabfluss":"Surface Runoff","Oberfl\xC3\xA4chennahe Windgeschwindigkeit":"Near-Surface Wind Speed","Oberfl\xC3\xA4chentemperatur":"Surface Temperature","Potenzielle Evapotranspiration":"Potential Evapotranspiration","Relative Luftfeuchte auf Druckfl\xC3\xA4che 1000hPa":"Relative Humidity at 1000 hPa","Relative Luftfeuchte auf Druckfl\xC3\xA4che 200hPa":"Relative Humidity at 200 hPa","Relative Luftfeuchte auf Druckfl\xC3\xA4che 250hPa":"Relative Humidity at 250 hPa","Relative Luftfeuchte auf Druckfl\xC3\xA4che 300hPa":"Relative Humidity at 300 hPa","Relative Luftfeuchte auf Druckfl\xC3\xA4che 500hPa":"Relative Humidity at 500 hPa","Relative Luftfeuchte auf Druckfl\xC3\xA4che 700hPa":"Relative Humidity at 700 hPa","Relative Luftfeuchte auf Druckfl\xC3\xA4che 850hPa":"Relative Humidity at 850 hPa","Relative Luftfeuchte auf Druckfl\xC3\xA4che 925hPa":"Relative Humidity at 925 hPa","Relative Luftfeuchte auf Druckfl\xC3\xA4che 950hPa":"Relative Humidity at 950 hPa","Relative Luftfeuchte in 2 m":"Near-Surface Relative Humidity",Schneebedeckung:"Snow Area Percentage","Schneeh\xC3\xB6he":"Snow Depth","Schneemenge (Wasser\xC3\xA4quivalent)":"Surface Snow Amount",Schneeniederschlag:"Snowfall Flux",Schneeschmelze:"Surface Snow Melt",Sonnenscheindauer:"Duration of Sunshine","Spezifische Luftfeuchte auf Druckfl\xC3\xA4che 1000hPa":"Specific Humidity at 1000 hPa","Spezifische Luftfeuchte auf Druckfl\xC3\xA4che 200hPa":"Specific Humidity at 200 hPa","Spezifische Luftfeuchte auf Druckfl\xC3\xA4che 250hPa":"Specific Humidity at 250 hPa","Spezifische Luftfeuchte auf Druckfl\xC3\xA4che 300hPa":"Specific Humidity at 300 hPa","Spezifische Luftfeuchte auf Druckfl\xC3\xA4che 500hPa":"Specific Humidity at 500 hPa","Spezifische Luftfeuchte auf Druckfl\xC3\xA4che 700hPa":"Specific Humidity at 700 hPa","Spezifische Luftfeuchte auf Druckfl\xC3\xA4che 850hPa":"Specific Humidity at 850 hPa","Spezifische Luftfeuchte auf Druckfl\xC3\xA4che 925hPa":"Specific Humidity at 925 hPa","Spezifische Luftfeuchte auf Druckfl\xC3\xA4che 950hPa":"Specific Humidity at 950 hPa","Spezifische Luftfeuchte in 2 m":"Near-Surface Specific Humidity","S\xC3\xBCdwindkomponente auf Druckfl\xC3\xA4che 1000hPa":"Northward Wind at 1000 hPa","S\xC3\xBCdwindkomponente auf Druckfl\xC3\xA4che 200hPa":"Northward Wind at 200 hPa","S\xC3\xBCdwindkomponente auf Druckfl\xC3\xA4che 250hPa":"Northward Wind at 250 hPa","S\xC3\xBCdwindkomponente auf Druckfl\xC3\xA4che 300hPa":"Northward Wind at 300 hPa","S\xC3\xBCdwindkomponente auf Druckfl\xC3\xA4che 500hPa":"Northward Wind at 500 hPa","S\xC3\xBCdwindkomponente auf Druckfl\xC3\xA4che 700hPa":"Northward Wind at 700 hPa","S\xC3\xBCdwindkomponente auf Druckfl\xC3\xA4che 850hPa":"Northward Wind at 850 hPa","S\xC3\xBCdwindkomponente auf Druckfl\xC3\xA4che 925hPa":"Northward Wind at 925 hPa","S\xC3\xBCdwindkomponente auf Druckfl\xC3\xA4che 950hPa":"Northward Wind at 950 hPa","Taupunkttemperatur in 2m":"2m Dewpoint Temperature","Turbulenter f\xC3\xBChbarer W\xC3\xA4rmefluss am Boden":"Surface Upward Sensible Heat Flux","Turbulenter latenter W\xC3\xA4rmefluss am Boden":"Surface Upward Latent Heat Flux","Verdunstung gesamt":"Evaporation Including Sublimation and Transpiration","Verf\xC3\xBCgbare konvektive potentielle Energie (CAPE)":"Convective Available Potential Energy","Wassergehalt der Bodenschichten (inkl. Gefrorenem Anteil)":"Total Soil Moisture Content","Water Vapor Path":"Water Vapor Path","Westwindkomponente auf Druckfl\xC3\xA4che 1000hPa":"Eastward Wind at 1000 hPa","Westwindkomponente auf Druckfl\xC3\xA4che 200hPa":"Eastward Wind at 200 hPa","Westwindkomponente auf Druckfl\xC3\xA4che 250hPa":"Eastward Wind at 250 hPa","Westwindkomponente auf Druckfl\xC3\xA4che 300hPa":"Eastward Wind at 300 hPa","Westwindkomponente auf Druckfl\xC3\xA4che 500hPa":"Eastward Wind at 500 hPa","Westwindkomponente auf Druckfl\xC3\xA4che 700hPa":"Eastward Wind at 700 hPa","Westwindkomponente auf Druckfl\xC3\xA4che 850hPa":"Eastward Wind at 850 hPa","Westwindkomponente auf Druckfl\xC3\xA4che 925hPa":"Eastward Wind at 925 hPa","Westwindkomponente auf Druckfl\xC3\xA4che 950hPa":"Eastward Wind at 950 hPa","Zonale Windkomponente in 10 m (Westwind)":"Eastward Near-Surface Wind",abs550aer:"Ambient Aerosol Absorption Optical Thickness at 550 nm",acabf:"Surface Mass Balance Flux",acabfis:"Ice Sheet Surface Mass Balance flux",aerasymbnd:"Aerosol Level Asymmetry Parameter for Each Band",aeroptbnd:"Aerosol Level Absorption Optical Thickness for Each Band",aerssabnd:"Aerosol Level Single Scattering Albedo for Each Band",ageice:"Age of Sea Ice",agesno:"Mean Age of Snow",agessc:"Sea Water Age Since Surface Contact",airmass:"Vertically Integrated Mass Content of Air in Layer",al:"Albedo",albc:"Canopy Albedo",albdiffbnd:"Diffuse Surface Albedo for Each Band",albdirbnd:"Direct Surface Albedo for Each Band",albedo:"Surface albedo",albisccp:"ISCCP Mean Cloud Albedo",albsn:"Snow Albedo",amo:"Sea Surface Temperature","amo-g":"sea surface temperature",anor:"Angle of Sub-gridscale Orography",aoanh:"Northern Hemisphere Tracer Lifetime",aod550volso4:"Aerosol Optical Depth at 550nm Due to Stratospheric Volcanic Aerosols",arag:"Aragonite Concentration",aragos:"Surface Aragonite Concentration",areacella:"Grid-Cell Area for Atmospheric Grid Variables",areacellg:"Grid-Cell Area for Ice Sheet Variables",areacello:"Grid-Cell Area for Ocean Variables",areacellr:"Grid-Cell Area for River Model Variables",ares:"Aerodynamic Resistance",asn:"Snow Albedo",atlantic_moc:"Atlantic ocean meridional overturning streamfunction",avspsbl:"Evaporation",bacc:"Bacterial Carbon Concentration",baccos:"Surface Bacterial Carbon Concentration",baresoilfrac:"Bare Soil Percentage Area Coverage",basin:"Region Selection Index",bbihghatm90frac:"blocking fraction based on daily binary blocking index (bbi) HGHA-TM90-2D",bbitm90frac:"blocking fraction based on daily binary blocking index (bbi) TM90-2D",bddtalk:"Rate of Change of Alkalinity Due to Biological Activity",bddtdic:"Rate of Change of Dissolved Inorganic Carbon Due to Biological Activity",bddtdife:"Rate of Change of Dissolved Inorganic Iron Due to Biological Activity",bddtdin:"Rate of Change of Nitrogen Nutrient Due to Biological Activity",bddtdip:"Rate of Change of Dissolved Phosphorus Due to Biological Activity",bddtdisi:"Rate of Change of Dissolved Inorganic Silicon Due to Biological Activity",bfe:"Mole Concentration of Particulate Organic Matter Expressed as Iron in Sea Water",bfeos:"Surface Mole Concentration of Particulate Organic Matter Expressed as Iron in Sea Water",bigthetao:"Sea Water Conservative Temperature",bigthetaoga:"Global Average Sea Water Conservative Temperature",bld:"Boundary Layer Dissipation",bldep:"Boundary Layer Depth",bmelt:"Rate of Melt at Sea Ice Base",bry:"Total Inorganic Bromine Volume Mixing Ratio",bs550aer:"Aerosol Backscatter Coefficient",bsi:"Mole Concentration of Particulate Organic Matter Expressed as Silicon in Sea Water",bsios:"Surface Mole Concentration of Particulate Organic Matter Expressed as Silicon in Sea Water",budg:"Evaporation-Precipitation",budgsatm:"Evaporation-Precipitation Satellite Bit Mask",burntarea:"Burnt Area Fraction",burntfractionall:"Percentage of Entire Grid cell that is Covered by Burnt Vegetation (All Classes)",c13land:"Mass of 13C in All Terrestrial Carbon Pools",c13litter:"Mass of 13C in Litter Pool",c13soil:"Mass of 13C in Soil Pool",c13veg:"Mass of 13C in Vegetation",c14land:"Mass of 14C in All Terrestrial Carbon Pools",c14litter:"Mass of 14C in Litter Pool",c14soil:"Mass of 14C in Soil Pool",c14veg:"Mass of 14C in Vegetation",c2h2:"C2H2 Volume Mixing Ratio",c2h6:"C2H6 Volume Mixing Ratio",c3h6:"C3H6 Volume Mixing Ratio",c3h8:"C3H8 Volume Mixing Ratio",c3pftfrac:"Percentage Cover by C3 Plant Functional Type",c4pftfrac:"Percentage Cover by C4 Plant Functional Type",calc:"Calcite Concentration",calcos:"Surface Calcite Concentration",cape:"Convective Available Potential Energy",ccb:"Air Pressure at Convective Cloud Base",ccldncl:"Cloud Droplet Number Concentration of Convective Cloud Tops",ccn:"Cloud Condensation Nuclei Concentration at Liquid Cloud Top",cct:"Air Pressure at Convective Cloud Top",ccwd:"Carbon Mass in Coarse Woody Debris",cdd:"Maximum length of dry spell (RR < 1 mm)",cddetccdi:"Maximum Number of Consecutive Days with Less Than 1mm of Precipitation",cdnc:"Cloud Liquid Droplet Number Concentration",cfad2lidarsr532:"CALIPSO Scattering Ratio",cfaddbze94:"CloudSat Radar Reflectivity CFAD",cfadlidarsr532:"CALIPSO Scattering Ratio CFAD",cfc11:"Mole Concentration of CFC11 in Sea Water",cfc113global:"Global Mean Mole Fraction of CFC113",cfc11global:"Global Mean Mole Fraction of CFC11",cfc12:"Mole Concentration of CFC12 in Sea Water",cfc12global:"Global Mean Mole Fraction of CFC12",cfl:"Surface Longwave Cloud Radiative Effect",cflstddev:"Surface Longwave Cloud Radiative Effect Standard Deviation",ch3coch3:"CH3COCH3 Volume Mixing Ratio",ch4:"Mole Fraction of CH4",ch4clim:"Mole Fraction of CH4",ch4global:"Global Mean Mole Fraction of CH4",ch4globalclim:"Global Mean Mole Fraction of CH4",cheaqpso4:"Aqueous-Phase Production Rate of SO4",chegpso4:"Gas-Phase Production Rate of SO4",chepasoa:"Total Net Production of Anthropogenic Secondary Organic Aerosol",chepsoa:"Chemical Production of Dry Aerosol Secondary Organic Matter",chl:"Mass Concentration of Total Phytoplankton Expressed as Chlorophyll in Sea Water",chl3d:"Total Chlorophyll Mass Concentration",chlcalc:"Mass Concentration of Calcareous Phytoplankton Expressed as Chlorophyll in Sea Water",chlcalcos:"Surface Mass Concentration of Calcareous Phytoplankton Expressed as Chlorophyll in Sea Water",chldiat:"Mass Concentration of Diatoms Expressed as Chlorophyll in Sea Water",chldiatos:"Surface Mass Concentration of Diatoms Expressed as Chlorophyll in Sea Water",chldiaz:"Mass Concentration of Diazotrophs Expressed as Chlorophyll in Sea Water",chldiazos:"Surface Mass Concentration of Diazotrophs Expressed as Chlorophyll in Sea Water",chlmisc:"Mass Concentration of Other Phytoplankton Expressed as Chlorophyll in Sea Water",chlmiscos:"Surface Mass Concentration of Other Phytoplankton Expressed as Chlorophyll in Sea Water",chlos:"Surface Mass Concentration of Total Phytoplankton Expressed as Chlorophyll in Sea Water",chlpico:"Mass Concentration of Picophytoplankton Expressed as Chlorophyll in Sea Water",chlpicoos:"Surface Mass Concentration of Picophytoplankton Expressed as Chlorophyll in Sea Water",ci:"Fraction of Time Convection Occurs in Cell",cin:"Convective Inhibition of Mean Surface Layer Parcel",cl:"Percentage Cloud Cover",cland:"Total Carbon in All Terrestrial Carbon Pools",clayfrac:"Clay Fraction",clc:"Convective Cloud Area Percentage",clcalipso:"CALIPSO Percentage Cloud Cover",clcalipso2:"CALIPSO Cloud Cover Percentage Undetected by CloudSat (as Percentage of Area Covered)",clcalipsoice:"CALIPSO Ice Cloud Percentage",clcalipsoliq:"CALIPSO Liquid Cloud Percentage",clccalipso:"CALIPSO Clear Cloud Fraction",cldicemxrat:"Cloud Ice Mixing Ratio",cldicemxrat27:"Cloud Ice Mixing Ratio",cldnci:"Ice Crystal Number Concentration of Cloud Tops",cldncl:"Cloud Droplet Number Concentration of Cloud Tops",cldnvi:"Column Integrated Cloud Droplet Number",cldwatmxrat:"Cloud Water Mixing Ratio",cldwatmxrat27:"Cloud Water Mixing Ratio",cleaf:"Carbon Mass in Leaves",clh:"Low Cloud Fraction",clhcalipso:"CALIPSO High Level Cloud Area Percentage",cli:"Mass Fraction of Cloud Ice",clic:"Mass Fraction of Convective Cloud Ice",climodis:"MODIS Ice Cloud Area Percentage",clis:"Mass Fraction of Stratiform Cloud Ice",clisccp:"ISCCP Cloud Area Percentage",clitter:"Carbon Mass in Litter Pool",clitterabove:"Carbon Mass in Above-Ground Litter",clitterbelow:"Carbon Mass in Below-Ground Litter",clittercwd:"Carbon Mass in Coarse Woody Debris",clittergrass:"Carbon Mass in Litter on Grass Tiles",clitterlut:"Carbon in Above and Below-Ground Litter Pools on Land-Use Tiles",clittershrub:"Carbon Mass in Litter on Shrub Tiles",clittersubsurf:"Carbon Mass in Below-Ground Litter",clittersurf:"Carbon Mass in Above-Ground Litter",clittertree:"Carbon Mass in Litter on Tree Tiles",clivi:"Ice Water Path",clivic:"Convective Ice Water Path",clivierr:"Ice Water Path Retrieval Error",clivistddev:"Ice Water Path Standard Deviation",cll:"Low Cloud Fraction",cllcalipso:"CALIPSO Low Level Cloud Cover Percentage",clm:"Medium Cloud Fraction",clmcalipso:"CALIPSO Mid Level Cloud Cover Percentage",clmisr:"Percentage Cloud Cover as Calculated by the MISR Simulator (Including Error Flag)",clrcalipso:"CALIPSO 3D Clear fraction",cls:"Percentage Cover of Stratiform Cloud",clt:"Total Cloud Cover Percentage",cltc:"Convective Cloud Cover Percentage",cltcalipso:"CALIPSO Total Cloud Fraction",cltisccp:"ISCCP Total Cloud Cover Percentage",cltmodis:"MODIS Total Cloud Cover Percentage",cltnobs:"Total Cloud Fraction Number of Observations",cltstddev:"Total Cloud Fraction Standard Deviation",clw:"Mass Fraction of Cloud Liquid Water",clwc:"Mass Fraction of Convective Cloud Liquid Water",clwmodis:"MODIS Liquid Cloud Percentage",clws:"Mass Fraction of Stratiform Cloud Liquid Water",clwvi:"Condensed Water Path",clwvic:"Convective Condensed Water Path",clwvierr:"Condensed Water Path Retrieval Error",clwvistddev:"Condensed Water Path Standard Deviation",cly:"Total Inorganic Chlorine Volume Mixing Ratio",cmisc:"Carbon Mass in Other Living Compartments on Land",cnc:"Canopy Covered Area Percentage",co:"CO Volume Mixing Ratio",co2:"Mole Fraction of CO2",co23d:"3D-Field of Transported CO2",co2clim:"Mole Fraction of CO2",co2mass:"Total Atmospheric Mass of CO2",co2massclim:"Total Atmospheric Mass of CO2",co2s:"Atmosphere CO2",co3:"Carbonate Ion Concentration",co33d:"Mole Concentration of Carbonate expressed as Carbon in Sea Water",co3abio:"Abiotic Carbonate Ion Concentration",co3abioos:"Surface Abiotic Carbonate Ion Concentration",co3nat:"Natural Carbonate Ion Concentration",co3natos:"Surface Natural Carbonate Ion Concentration",co3os:"Surface Carbonate Ion Concentration",co3satarag:"Mole Concentration of Carbonate Ion in Equilibrium with Pure Aragonite in Sea Water",co3sataragos:"Surface Mole Concentration of Carbonate Ion in Equilibrium with Pure Aragonite in Sea Water",co3satcalc:"Mole Concentration of Carbonate Ion in Equilibrium with Pure Calcite in Sea Water",co3satcalc3d:"Mole Concentration of Calcite expressed as Carbon in Sea Water at Saturation",co3satcalcos:"Surface Mole Concentration of Carbonate Ion in Equilibrium with Pure Calcite in Sea Water",cod:"Cloud Optical Depth",columnmassflux:"Column Integrated Mass Flux",concaerh2o:"Concentration of Aerosol Water",concbb:"Concentration of Biomass Burning Aerosol",concbc:"Concentration of Black Carbon Aerosol",conccmcn:"Number Concentration Coarse Mode Aerosol",conccn:"Aerosol Number Concentration",concdms:"Mole Fraction of DMS",concdust:"Concentration of Dust",concnh4:"Concentration of NH4",concnmcn:"Number Concentration of Nucleation Mode Aerosol",concno3:"Concentration of NO3 Aerosol",concoa:"Concentration of Dry Aerosol Organic Matter",concpoa:"Concentration of Dry Aerosol Primary Organic Matter",concso2:"Mole Fraction of SO2",concso4:"Concentration of SO4",concsoa:"Concentration of Dry Aerosol Secondary Organic Matter",concss:"Concentration of Seasalt",cother:"Carbon Mass in Vegetation Components Other than Leaves, Stems and Roots",cph:"Area Fraction of Liquid Cloud_Water Particles at Cloud Top",cproduct:"Carbon Mass in Products of Land-Use Change",cproductlut:"wood and agricultural product pool carbon associated with land use tiles; examples of products include paper, cardboard, timber for construction, and crop harvest for food or fuel.",crcw:"Specific Rain Water Content",croot:"Carbon Mass in Roots",cropfrac:"Percentage Crop Cover",cropfracc3:"Percentage Cover by C3 Crops",cropfracc4:"Percentage Cover by C4 Crops",csoil:"Carbon Mass in Model Soil Pool",csoilabove1m:"Carbon Mass in Soil Pool Above 1m Depth",csoilfast:"Carbon Mass in Fast Soil Pool",csoilgrass:"Carbon Mass in Soil on Grass Tiles",csoillevels:"Carbon Mass in Each Model Soil Level (Summed over All Soil Carbon Pools in That Level)",csoillut:"Carbon in Soil Pool on Land-Use Tiles",csoilmedium:"Carbon Mass in Medium Soil Pool",csoilpools:"Carbon Mass in Each Model Soil Pool (Summed over Vertical Levels)",csoilshrub:"Carbon Mass in Soil on Shrub Tiles",csoilslow:"Carbon Mass in Slow Soil Pool",csoiltree:"Carbon Mass in Soil on Tree Tiles",cstem:"Carbon Mass in Stem",cswc:"Specific Snow Water Content",cth:"Height at Cloud Top",cthstddev:"Height at Cloud Top Standard Deviation",ctotfirelut:"Total Carbon Loss from Natural and Managed Fire on Land-Use Tile, Including Deforestation Fires",ctp:"Air Pressure at Cloud Top",ctpstddev:"Air Pressure at Cloud Top Standard Deviation",ctt:"Air Temperature at Cloud Top",cttisccp:"ISCCP Mean Cloud Top Temperature (Cloud-fraction weighted & daytime only)",cttstddev:"Air Temperature at Cloud Top Standard Deviation",cveg:"Carbon Mass in Vegetation",cveggrass:"Carbon Mass in Vegetation on Grass Tiles",cveglut:"carbon in vegetation on land use tiles",cvegshrub:"Carbon Mass in Vegetation on Shrub Tiles",cvegtree:"Carbon Mass in Vegetation on Tree Tiles",cvh:"High Vegetation Cover",cvl:"Low Vegetation Cover",cw:"Total Canopy Water Storage",cwd:"Maximum length of wet spell (RR > 1 mm)",cwood:"Carbon Mass in Wood",cya:"Cyanobacteria concentration",cyctd:"track density of cyclones per DJF within 1000km radius",cyctf:"cyclone_track_flux",d:"Divergence",dapaclip:"Global DAily Precipitation Analysis for the validation of medium-range CLImate Predictions (MiKlip first phase)",darag:"Aragonite Dissolution",daycounttasmax28c:"Days per year with max. Near-Surface Air Temperature greater than 28C",days_with_tasmnmx_interval:"Days per year with min. and max. Near-Surface Air Temperature between 22C and 31C",dcalc:"Calcite Dissolution",dcw:"Change in Interception Storage",demc:"Convective Cloud Emissivity",dems:"Stratiform Cloud Emissivity",depdust:"Total Deposition Rate of Dust",deptho:"Sea Floor Depth Below Geoid",detoc:"Detrital Organic Carbon Concentration",detocos:"Surface Detrital Organic Carbon Concentration",dfe:"Dissolved Iron Concentration",dfe3d:"Dissolved Iron Concentration",dfeos:"Surface Dissolved Iron Concentration",dgw:"Change in Groundwater",dhum:"Deviation of Near-Surface Specific Humidity from Saturation",dhumnumd:"Deviation of Near-Surface Specific Humidity from Saturation Number of Days with Observations",dhumsatm:"Deviation of Near-Surface Specific Humidity from Saturation Satellite Bit Mask",dhumstddev:"Deviation of Near-Surface Specific Humidity from Saturation Standard Deviation",diabdrag:"Tendency of Eastward Wind from Numerical Artefacts",difmxybo:"Ocean Momentum XY Biharmonic Diffusivity",difmxybo2d:"Ocean Momentum XY Biharmonic Diffusivity",difmxylo:"Ocean Momentum XY Laplacian Diffusivity",difmxylo2d:"Ocean Momentum XY Laplacian Diffusivity",diftrbbo:"Ocean Tracer Bolus Biharmonic Diffusivity",diftrbbo2d:"Ocean Tracer Bolus Biharmonic Diffusivity",diftrblo:"Ocean Tracer Diffusivity Due to Parameterized Mesoscale Advection",diftrblo2d:"Ocean Tracer Diffusivity Due to Parameterized Mesoscale Advection",diftrebo:"Ocean Tracer Epineutral Biharmonic Diffusivity",diftrebo2d:"Ocean Tracer Epineutral Biharmonic Diffusivity",diftrelo:"Ocean Tracer Epineutral Laplacian Diffusivity",diftrelo2d:"Ocean Tracer Epineutral Laplacian Diffusivity",diftrxybo:"Ocean Tracer XY Biharmonic Diffusivity",diftrxybo2d:"Ocean Tracer XY Biharmonic Diffusivity",diftrxylo:"Ocean Tracer XY Laplacian Diffusivity",diftrxylo2d:"Ocean Tracer XY Laplacian Diffusivity",difvho:"Ocean Vertical Heat Diffusivity",difvmbo:"Ocean Vertical Momentum Diffusivity due to Background",difvmfdo:"Ocean Vertical Momentum Diffusivity Due to Form Drag",difvmo:"Ocean Vertical Momentum Diffusivity",difvmto:"Ocean Vertical Momentum Diffusivity due to Tides",difvso:"Ocean Vertical Salt Diffusivity",difvtrbo:"Ocean Vertical Tracer Diffusivity due to Background",difvtrto:"Ocean Vertical Tracer Diffusivity Due to Tides",dispkevfo:"Ocean Kinetic Energy Dissipation per Unit Area Due to Vertical Friction",dispkexyfo:"Ocean Kinetic Energy Dissipation per Unit Area Due to XY Friction",dispkexyfo2d:"Ocean Kinetic Energy Dissipation per Unit Area Due to XY Friction",dissi13c:"Dissolved Inorganic Carbon-13 Concentration",dissi13cos:"Surface Dissolved Inorganic Carbon-13 Concentration",dissi14c:"Dissolved Inorganic Carbon-14 Concentration",dissi14cabio:"Abiotic Dissolved Inorganic Carbon-14 Concentration",dissi14cabioos:"Surface Abiotic Dissolved Inorganic Carbon-14 Concentration",dissic:"Dissolved Inorganic Carbon Concentration",dissic3d:"Dissolved Inorganic Carbon Concentration",dissicabio:"Abiotic Dissolved Inorganic Carbon Concentration",dissicabioos:"Surface Abiotic Dissolved Inorganic Carbon Concentration",dissicnat:"Natural Dissolved Inorganic Carbon Concentration",dissicnatos:"Surface Natural Dissolved Inorganic Carbon Concentration",dissicos:"Surface Dissolved Inorganic Carbon Concentration",dissoc:"Dissolved Organic Carbon Concentration",dissoc3d:"Dissolved Organic Carbon Concentration",dissocos:"Surface Dissolved Organic Carbon Concentration",divice:"Strain Rate Divergence of Sea Ice",dls:"",dmc:"Deep Convective Mass Flux",dmlt:"Depth to Soil Thaw",dms:"Dimethyl Sulphide (DMS) Mole Fraction",dmso:"Mole Concentration of Dimethyl Sulphide in Sea Water",dmsos:"Surface Mole Concentration of Dimethyl Sulphide in Sea Water",dpco2:"Delta CO2 Partial Pressure",dpco2_nontherm:"Surface Aqueous Partial Pressure of CO2",dpco2_nontherm_res:"Surface Aqueous Partial Pressure of CO2",dpco2abio:"Abiotic Delta Pco Partial Pressure",dpco2nat:"Natural Delta CO2 Partial Pressure ",dpo2:"Delta O2 Partial Pressure",dpocdtdiaz:"Tendency of Mole Concentration of Organic Carbon in Sea Water due to Net Primary Production by Diazotrophs",dpocdtpico:"Tendency of Mole Concentration of Organic Carbon in Sea Water due to Net Primary Production by Picophytoplankton",drivw:"Change in River Storage",drybc:"Dry Deposition Rate of Black Carbon Aerosol Mass",drydms:"Dry Deposition Rate of DMS",drydust:"Dry Deposition Rate of Dust",drynh3:"Dry Deposition Rate of NH3",drynh4:"Dry Deposition Rate of NH4",drynoy:"Dry Deposition Rate of NOy",dryo3:"Dry Deposition Rate of O3",dryoa:"Dry Deposition Rate of Dry Aerosol Total Organic Matter",drypoa:"Dry Deposition Rate of Dry Aerosol Primary Organic Matter",dryso2:"Dry Deposition Rate of SO2",dryso4:"Dry Deposition Rate of SO4",drysoa:"Dry Deposition Rate of Dry Aerosol Secondary Organic Matter",dryss:"Dry Deposition Rate of Sea-Salt Aerosol",dryta:"DRY AIR Temperature",dslw:"Change in Soil Moisture",dsn:"Change in Snow Water Equivalent",dsw:"Change in Surface Water Storage",dtauc:"Convective Cloud Optical Depth",dtaucerr:"Convective Cloud Optical Depth Standard Error",dtaucstddev:"Convective Cloud Optical Depth Standard Deviation",dtaus:"Stratiform Cloud Optical Depth",dtes:"Change in Surface Heat Storage",dtesn:"Change in Snow and Ice Cold Content",dtretccdi:"Mean Diurnal Temperature Range",eady:"Eady Growth Rate",ec:"Interception Evaporation",ec550aer:"Ambient Aerosol Extinction Coefficient at 550 nm",edt:"Eddy Diffusivity Coefficient for Temperature",egr:"Eady Growth Rate",emiaco:"Total Emission Rate of Anthropogenic CO",emianox:"Total Emission Rate of Anthropogenic NOx",emiaoa:"Total Emission Rate of Anthropogenic Organic Aerosol",emibb:"Total Emission of Primary Aerosol from Biomass Burning",emibc:"Total Emission Rate of Black Carbon Aerosol Mass",emibvoc:"Total Emission Rate of Biogenic NMVOC",emico:"Total Emission Rate of CO",emidms:"Total Emission Rate of DMS",emidust:"Total Emission Rate of Dust",emiisop:"Total Emission Rate of Isoprene",emilnox:"Layer-Integrated Lightning Production of NOx",eminh3:"Total Emission Rate of NH3",eminox:"Total Emission Rate of NOx",emioa:"Primary Emission and Chemical Production of Dry Aerosol Organic Matter",emipoa:"Emission Rate of Dry Aerosol Primary Organic Matter",emiso2:"Total Emission Rate of SO2",emiso4:"Total Direct Emission Rate of SO4",emiss:"Total Emission Rate of Sea-Salt Aerosol",emivoc:"Total Emission Rate of NMVOC",eow:"Open Water Evaporation",eparag100:"Downward Flux of Aragonite",epc100:"Downward Flux of Particulate Organic Carbon",epcalc100:"Downward Flux of Calcite",epfe100:"Downward Flux of Particulate Iron",epfy:"Northward Component of the Eliassen-Palm Flux",epfz:"Upward Component of the Eliassen-Palm Flux",epn100:"Downward Flux of Particulate Nitrogen",epp100:"Downward Flux of Particulate Phosphorus",epsi100:"Downward Flux of Particulate Silicon",es:"Bare Soil Evaporation",esn:"Snow Evaporation",etadot:"Eta-coordinate Vertical Velocity",evap:"Water Evaporation Flux from Sea Ice",evisct:"Eddy Diffusivity Coefficients for Temperature",eviscu:"Eddy Viscosity Coefficients for Momentum",evs:"Water Evaporation Flux Where Ice Free Ocean over Sea",evspsbl:"Evaporation Including Sublimation and Transpiration",evspsblnumd:"Evaporation Number of Days with Observations",evspsblpot:"Potential Evapotranspiration",evspsblsatm:"Evaporation Satellite Bit Mask",evspsblsoi:"Water Evaporation from Soil",evspsblstddev:"Evaporation Standard Deviation",evspsblveg:"Evaporation from Canopy",evu:"Eddy Viscosity Coefficient for Momentum",exparag:"Downward Flux of Aragonite",expc:"Downward Flux of Particulate Organic Carbon",expcalc:"Downward Flux of Calcite",expcfe:"Sinking Particulate Iron Flux",expfe:"Sinking Particulate Iron Flux",expn:"Sinking Particulate Organic Nitrogen Flux",expp:"Sinking Particulate Organic Phosphorus Flux",expsi:"Sinking Particulate Silicon Flux",fahlut:"Anthropogenic Heat Flux Generated from non-Renewable Human Primary Energy Consumption",fal:"Forecast Albedo",fanthdisturb:"Carbon Mass Flux from Vegetation, Litter or Soil Pools into the Atmosphere Due to any Human Activity",fbddtalk:"Rate of Change of Biological Alkalinity Due to Biological Activity",fbddtdic:"Rate of Change of Dissolved Inorganic Carbon Due to Biological Activity",fbddtdife:"Rate of Change of Dissolved Inorganic Iron Due to Biological Activity",fbddtdin:"Rate of Change of Dissolved Inorganic Nitrogen Due to Biological Activity",fbddtdip:"Rate of Change of Dissolved Inorganic Phosphate due to Biological Activity",fbddtdisi:"Rate of Change of Dissolved Inorganic Silicate due to Biological Activity",fbnf:"Biological Nitrogen Fixation",fclandtoocean:"Lateral Transfer of Carbon out of Grid Cell That Eventually Goes into Ocean",fco2antt:"Carbon Mass Flux into Atmosphere Due to All Anthropogenic Emissions of CO2",fco2fos:"Carbon Mass Flux into Atmosphere Due to Fossil Fuel Emissions of CO2",fco2nat:"Surface Carbon Mass Flux into the Atmosphere Due to Natural Sources",fd:"Frost days index is the number of days where minimum of temperature is below 0 degree Celsius. The time period should be defined by the bounds of the time coordinate.",fddtalk:"Rate of Change of Total Alkalinity",fddtdic:"Rate of Change of Net Dissolved Inorganic Carbon",fddtdife:"Rate of Change of Net Dissolved Inorganic Iron",fddtdin:"Rate of Change of Net Dissolved Inorganic Nitrogen",fddtdip:"Rate of Change of Net Dissolved Inorganic Phosphorus",fddtdisi:"Rate of Change of Net Dissolved Inorganic Silicon",fdeforesttoatmos:"Deforested Biomass That Goes into Atmosphere as a Result of Anthropogenic Land-Use Change",fdeforesttoproduct:"Deforested Biomass That Goes into Product Pool as a Result of Anthropogenic Land-Use Change",fdetccdi:"Number of Frost Days",fediss:"Particulate Source of Dissolved Iron",fescav:"Non-Biogenic Iron Scavenging",ffire:"Carbon Mass Flux into Atmosphere Due to CO2 Emission from Fire Excluding Land-Use Change",ffireall:"Carbon Mass Flux into Atmosphere due to CO2 emissions from Fire resulting from all sources including natural, anthropogenic and land use change.",ffirenat:"Carbon Mass Flux into Atmosphere due to CO2 Emission from natural Fire",fg13co2:"Surface Downward Mass Flux of Carbon-13 as 13CO2",fg14co2:"Surface Downward Mass Flux of Carbon-14 as 14CO2",fg14co2abio:"Surface Downward Mass Flux of Carbon-14 as Abiotic 14CO2",fgcfc11:"Surface Downward CFC11 Flux",fgcfc12:"Surface Downward CFC12 Flux",fgco2:"Surface Downward Mass Flux of Carbon as CO2",fgco2abio:"Surface Downward Mass Flux of Carbon as Abiotic CO2",fgco2nat:"Surface Downward Mass Flux of Carbon as Natural CO2",fgdms:"Surface Upward Flux of DMS",fgo2:"Surface Downward Flux of O2",fgrazing:"Carbon Mass Flux into Atmosphere Due to Grazing on Land",fgsf6:"Surface Downward SF6 Flux",fharvest:"Carbon Mass Flux into Atmosphere Due to Crop Harvesting",fharvesttoatmos:"Harvested Biomass That Goes Straight into Atmosphere as Carbon Mass Flux",fharvesttoproduct:"Harvested Biomass That Goes into Product Pool",ficeberg:"Water Flux into Sea Water from Icebergs",ficeberg2d:"Water Flux into Sea Water from Icebergs",flandice:"Water Flux into Sea Water from Land Ice",flashrate:"Lightning Flash Rate",fldcapacity:"Field Capacity",flitterfire:"Carbon Mass Flux from Litter, CWD or any non-Living Pool into Atmosphere Due to CO2 Emission from All Fire",flittersoil:"Total Carbon Mass Flux from Litter to Soil",flsr:"Forecast Logarithm of Surface Roughness for Heat",fluc:"Net Carbon Mass Flux into Atmosphere Due to Land-Use Change",flulccatmlut:"Carbon Transferred Directly to Atmosphere Due to any Land-Use or Land-Cover Change Activities including deforestation or agricultural fire",flulccproductlut:"Carbon Harvested Due to Land-Use or Land-Cover Change Process That Enters Anthropogenic Product Pools on Tile",flulccresiduelut:"Carbon Transferred to Soil or Litter Pools Due to Land-Use or Land-Cover Change Processes on Tile",fn2o:"Total Land N2O Flux",fnanthdisturb:"Nitrogen Mass Flux out of Land Due to any Human Activity",fndep:"Dry and Wet Deposition of Reactive Nitrogen onto Land",fnfert:"Total Nitrogen Added for Cropland Fertilisation (Artificial and Manure)",fngas:"Total Nitrogen Lost to the Atmosphere (Sum of NHx, NOx, N2O, N2)",fngasfire:"Total Nitrogen Lost to the Atmosphere (Including NHx, NOx, N2O, N2) from Fire",fngasnonfire:"Total Nitrogen Lost to the Atmosphere (Including NHx, NOx, N2O, N2) from All Processes Except Fire",fnlandtoocean:"Lateral Transfer of Nitrogen out of Grid Cell That Eventually Goes into Ocean",fnleach:"Total Nitrogen Loss to Leaching or Runoff (Sum of Ammonium, Nitrite and Nitrate)",fnlittersoil:"Total Nitrogen Mass Flux from Litter to Soil",fnloss:"Total Nitrogen Lost (Including NHx, NOx, N2O, N2 and Leaching)",fnnetmin:"Net Nitrogen Release from Soil and Litter as the Outcome of Nitrogen Immobilisation and Gross Mineralisation",fnox:"Total Land NOx Flux",fnproduct:"Deforested or Harvested Biomass as a Result of Anthropogenic Land-Use or Change",fnup:"Total Plant Nitrogen Uptake (Sum of Ammonium and Nitrate) Irrespective of the Source of Nitrogen",fnveglitter:"Total Nitrogen Mass Flux from Vegetation to Litter",fnvegsoil:"Total Nitrogen Mass Flux from Vegetation Directly to Soil",fproductdecomp:"Decomposition out of Product Pools to CO2 in Atmosphere as Carbon Mass Flux",fproductdecomplut:"Net Carbon Mass Flux from Wood and Agricultural Product Pools on Land Use Tile into Atmosphere",fracinlut:"Annual Gross Percentage That Was Transferred into This Tile from Other Land-Use Tiles",fraclut:"Percentage of Grid Cell for Each Land-Use Tile",fracoutlut:"Annual Gross Percentage of Land-Use Tile That Was Transferred into Other Land-Use Tiles",frc:"Downward Carbon Flux at Ocean Bottom",frfe:"Iron Loss to Sediments",fric:"Downward Inorganic Carbon Flux at Ocean Bottom",friver:"Water Flux into Sea Water from Rivers",frn:"Nitrogen Loss to Sediments and Through Denitrification",froc:"Downward Organic Carbon Flux at Ocean Bottom",fsc:"Flux of Carbon Into Ocean Surface by Runoff and Sediment Dissolution",fsfe:"Surface Downward Net Flux of Iron",fsitherm:"Water Flux into Sea Water Due to Sea Ice Thermodynamics",fsn:"Surface Downward Net Flux of Nitrogen",fsr:"Forecast Surface Roughness",fvegfire:"Carbon Mass Flux from Vegetation into Atmosphere Due to CO2 Emission from All Fire",fveglitter:"Total Carbon Mass Flux from Vegetation to Litter",fveglittermortality:"Total Carbon Mass Flux from Vegetation to Litter as a Result of Mortality",fveglittersenescence:"Total Carbon Mass Flux from Vegetation to Litter as a Result of Leaf, Branch, and Root Senescence",fvegsoil:"Total Carbon Mass Flux from Vegetation Directly to Soil",fvegsoilmortality:"Total Carbon Mass Flux from Vegetation to Soil as a Result of Mortality",fvegsoilsenescence:"Total Carbon Mass Flux from Vegetation to Soil as a Result of Leaf, Branch, and Root Senescence",gdd:"Growing Degree Days",global_moc:"Global ocean meridional overturning streamfunction","gpcc-di":"GPCC Drought Index Product",gpccdi:"GPCC Drought Index Product",gpp:"Carbon Mass Flux out of Atmosphere Due to Gross Primary Production on Land",gppc13:"Carbon-13 Mass Flux out of Atmosphere Due to Gross Primary Production on Land",gppc14:"Carbon-14 Mass Flux out of Atmosphere Due to Gross Primary Production on Land",gppgrass:"Gross Primary Production on Grass Tiles as Carbon Mass Flux",gpplut:"Gross Primary Production on Land-Use Tile as Carbon Mass Flux",gppshrub:"Gross Primary Production on Shrub Tiles as Carbon Mass Flux",gpptree:"Gross Primary Production on Tree Tiles as Carbon Mass Flux",grassfrac:"Natural Grass Area Percentage",grassfracc3:"C3 Natural Grass Area Percentage",grassfracc4:"C4 Natural Grass Area Percentage",graz:"Total Grazing of Phytoplankton by Zooplankton",grcongel:"Congelation Sea Ice Growth Rate",grfrazil:"Frazil Sea Ice Growth (Leads) Rate",gridspec:"grid specification",grlateral:"Lateral Sea Ice Growth Rate",grpllsprof:"Stratiform Graupel Flux",grplmxrat27:"Graupel Mixing Ratio",gsl:"Counted are the number of days per calendar year between the first occurrence of at least 6 consecutive days where the daily mean temperature is above 5 degree Celsius and the first occurrence of at least 6 consecutive days after 1st of July where the dai",gwd:"Gravity Wave Dissipation",h2o:"Mass Fraction of Water",hc300totalperarea:"total_ocean_heat_content 0 to 300m per area",hc700:"ocean_heat_content 0 to 700m",hc700perunitarea:"Ocean heat content anomaly calculated from objectively analyzed temperature anomaly fields over given depth limits",hc700rbek_ts:"",hc700total:"Ocean heat content anomaly calculated from objectively analyzed temperature anomaly fields over given depth limits",hc700totalperarea:"total_ocean_heat_content 0 to 700m per unit area",hcc:"High Cloud Cover",hcfc22global:"Global Mean Mole Fraction of HCFC22",hcho:"Formaldehyde Volume Mixing Ratio",hcice:"Sea Ice Heat Content",hcl:"HCl Volume Mixing Ratio",hcont300:"Heat content of upper 300 meters",hd:"Heating degree days relates the outside temperature with the room temperature during the heating period. It is the sum of the difference between room temperature X and daily mean temperature Y on days where Y is below a given constant A. X is 20 degree Ce",hfbasin:"Northward Ocean Heat Transport",hfbasinba:"Northward Ocean Heat Transport due to Bolus Advection",hfbasindiff:"Northward Ocean Heat Transport due to Diffussion",hfbasinpadv:"Northward Ocean Heat Transport Due to Parameterized Eddy Advection",hfbasinpmadv:"Northward Ocean Heat Transport Due to Parameterized Mesoscale Advection",hfbasinpmdiff:"Northward Ocean Heat Transport Due to Parameterized Mesoscale Diffusion",hfbasinpsmadv:"Northward Ocean Heat Transport Due to Parameterized Submesoscale Advection",hfcorr:"Heat Flux Correction",hfds:"Downward Heat Flux at Sea Water Surface",hfdsl:"Downward Heat Flux at Land Surface",hfdsn:"Downward Heat Flux into Snow Where Land over Land",hfdsnb:"Downward Heat Flux at Snow Base",hfevapds:"Temperature Flux Due to Evaporation Expressed as Heat Flux out of Sea Water",hfgeou:"Upward Geothermal Heat Flux at Sea Floor",hfgeoubed:"Geothermal Heat Flux Beneath Land Ice",hfibthermds:"Heat Flux into Sea Water due to Iceberg Thermodynamics",hfibthermds2d:"Heat Flux into Sea Water Due to Iceberg Thermodynamics",hfls:"Surface Upward Latent Heat Flux",hflsis:"Ice Sheet Surface Upward Latent Heat Flux",hflslut:"Latent Heat Flux on Land-Use Tile",hflsnumd:"Surface Upward Latent Heat Flux Number of Days with Observations",hflso:"Surface Downward Latent Heat Flux",hflssatm:"Surface Upward Latent Heat Flux Satellite Bit Mask",hflssi:"Surface Upward Latent Heat Flux over Sea Ice",hflsstddev:"Surface Upward Latent Heat Flux Standard Deviation",hfmlt:"Energy of Fusion",hfrainds:"Temperature Flux Due to Rainfall Expressed as Heat Flux into Sea Water",hfrs:"Heat Transferred to Snowpack by Rainfall",hfrunoffds:"Temperature Flux Due to Runoff Expressed as Heat Flux into Sea Water",hfrunoffds2d:"Temperature Flux Due to Runoff Expressed as Heat Flux into Sea Water",hfsbl:"Energy of Sublimation",hfsifrazil:"Heat Flux into Sea Water Due to Frazil Ice Formation",hfsifrazil2d:"Heat Flux into Sea Water Due to Frazil Ice Formation",hfsithermds:"Heat Flux into Sea Water due to Sea Ice Thermodynamics",hfsnthermds:"Heat Flux into Sea Water Due to Snow Thermodynamics",hfsnthermds2d:"Heat Flux into Sea Water Due to Snow Thermodynamics",hfss:"Surface Upward Sensible Heat Flux",hfssi:"Surface Upward Sensible Heat Flux over Sea Ice",hfssis:"Ice Sheet Surface Upward Sensible Heat Flux",hfsslut:"Sensible Heat Flux on Land-Use Tile",hfssnumd:"Surface Upward Sensible Heat Flux Number of Days with Observations",hfsso:"Surface Downward Sensible Heat Flux",hfsssatm:"Surface Upward Sensible Heat Flux Satellite Bit Mask",hfssstddev:"Surface Upward Sensible Heat Flux Standard Deviation",hfx:"Ocean Heat X Transport",hfxba:"Ocean Heat X Transport due to Bolus Advection",hfxdiff:"Ocean Heat X Transport due to Diffusion",hfy:"Ocean Heat Y Transport",hfyba:"Ocean Heat Y Transport due to Bolus Advection",hfydiff:"Ocean Heat Y Transport due to Diffussion",hno3:"HNO3 Volume Mixing Ratio",ho2:"HO2 Volume Mixing Ratio",hsea:"Sea-Surface Saturation Specific Humidity",hseanumd:"Sea-Surface Saturation Specific Humidity Number of Days with Observations",hseasatm:"Sea-Surface Saturation Specific Humidity Satellite Bit Mask",hseastddev:"Sea-Surface Saturation Specific Humidity Standard Deviation",htovgyre:"Northward Ocean Heat Transport Due to Gyre",htovovrt:"Northward Ocean Heat Transport Due to Overturning",hur:"Relative Humidity",hur1000:"Relative Humidity at 1000 hPa",hur200:"Relative Humidity at 200 hPa",hur250:"Relative Humidity at 250 hPa",hur300:"Relative Humidity at 300 hPa",hur500:"Relative Humidity at 500 hPa",hur700:"Relative Humidity at 700 hPa",hur800:"Relative Humidity at 800 hPa",hur850:"Relative Humidity at 850 hPa",hur900:"Relative Humidity at 900 hPa",hur925:"Relative Humidity at 925 hPa",hur950:"Relative Humidity at 950 hPa",hur975:"Relative Humidity at 975 hPa",hurmean:"Mean Relative Humidity in the 850-500-hPa layer",hurs:"Near-Surface Relative Humidity",hursmax:"Daily Maximum Near-Surface Relative Humidity",hursmin:"Daily Minimum Near-Surface Relative Humidity",hursmincrop:"Daily Minimum Near-Surface Relative Humidity over Crop Tile",hus:"Specific Humidity",hus1000:"Specific Humidity at 1000 hPa",hus200:"Specific Humidity at 200 hPa",hus250:"Specific Humidity at 250 hPa",hus27:"Specific Humidity",hus300:"Specific Humidity at 300 hPa",hus4:"Specific Humidity",hus400:"Specific Humidity at 400 hPa",hus500:"Specific Humidity at 500 hPa",hus600:"Specific Humidity at 600 hPa",hus700:"Specific Humidity at 700 hPa",hus7h:"Specific Humidity",hus800:"Specific Humidity at 800 hPa",hus850:"Specific Humidity at 850 hPa",hus900:"Specific Humidity at 900 hPa",hus925:"Specific Humidity at 925 hPa",hus950:"Specific Humidity at 950 hPa",hus975:"Specific Humidity at 975 hPa",husnobs:"Specific Humidity Number of Observations",huss:"Near-Surface Specific Humidity",husslut:"Near-Surface Specific Humidity on Land-Use Tile",hussnumd:"Near-Surface Specific Humidity Number of Days with Observations",husssatm:"Near-Surface Specific Humidity Satellite Bit Mask",hussstddev:"Near-Surface Specific Humidity Standard Deviation",husstderr:"Specific Humidity Standard Error",hwdi:"Heatwave day intensity yearsum",hwds:"Heatwave Days yearsum",ialb:"Sea Ice Albedo",iareafl:"Area Covered by Floating Ice Shelves",iareagr:"Area Covered by Grounded Ice Sheet",ibbihghatm90frac:"monthly blocking fraction based on daily instantaneous binary blocking index (ibbi) HGHA-TM90-2D",ibbitm90frac:"monthly blocking fraction based on daily instantaneous binary blocking index (ibbi) TM90-2D",icem:"Surface Ice Melt Flux",icemis:"Ice Sheet Surface Ice Melt Flux",icfriver:"Flux of Inorganic Carbon Into Ocean Surface by Runoff",id:"Ice days index is the number of days where maximum of temperature is below 0 degree Celsius. The time period should be defined by the bounds of the time coordinate.",idetccdi:"Number of Icing Days",inc:"Ice Crystal Number Concentration",indopacific_moc:"Indopacific ocean meridional overturning streamfunction",intdic:"Dissolved Inorganic Carbon Content",intdoc:"Dissolved Organic Carbon Content",intparag:"Aragonite Production",intpbfe:"Iron Production",intpbn:"Nitrogen Production",intpbp:"Phosphorus Production",intpbsi:"Silicon Production",intpcalc:"Net Primary Mole Productivity of Carbon by Calcareous Phytoplankton",intpcalcite:"Calcite Production",intpdiat:"Primary Organic Carbon Production by Diatoms",intpdiaz:"Net Primary Mole Productivity of Carbon by Diazotrophs",intpmisc:"Primary Organic Carbon Production by Other Phytoplankton",intpn2:"Nitrogen Fixation Rate in Ocean",intpnitrate:"Primary Organic Carbon Production by Phytoplankton Based on Nitrate Uptake Alone",intpoc:"Particulate Organic Carbon Content",intpp:"Primary Organic Carbon Production by All Types of Phytoplankton",intppcalc:"Net Primary Mole Productivity of Carbon by Calcareous Phytoplankton",intppdiat:"Net Primary Organic Carbon Production by Diatoms",intppdiaz:"Net Primary Mole Productivity of Carbon by Diazotrophs",intppico:"Net Primary Mole Productivity of Carbon by Picophytoplankton",intppmisc:"Net Primary Organic Carbon Production by Other Phytoplankton",intppnitrate:"Primary Organic Carbon Production by Phytoplankton Based on Nitrate Uptake Alone",intpppico:"Net Primary Mole Productivity of Carbon by Picophytoplankton",intuadse:"Vertically Integrated Eastward Dry Statice Energy Transport",intuaw:"Vertically Integrated Eastward Moisture Transport",intvadse:"Vertically Integrated Northward Dry Static Energy Transport",intvaw:"Vertically Integrated Northward Moisture Transport",ipo:"sea surface temperature","ipo-g":"sea surface temperature",irrlut:"Irrigation Flux Including any Irrigation for Crops, Trees, Pasture, or Urban Lawns",isop:"Isoprene Volume Mixing Ratio",isor:"Anisotropy of Sub-gridscale Orography",istl1:"Ice Temperature Layer 1",istl2:"Ice Temperature Layer 2",istl3:"Ice Temperature Layer 3",istl4:"Ice Temperature Layer 4",jno2:"Photolysis Rate of NO2",jo2:"Photolysis Rate of Diatomic Molecular Oxygen",jo3:"Photolysis Rate of Ozone (O3)",jpdftaureicemodis:"MODIS Joint Distribution of Optical Thickness and Particle Size, Ice",jpdftaureliqmodis:"MODIS Optical Thickness-Particle Size Joint Distribution, Liquid",kcyccg:"",kcyctd:"cyclone_track_density",ksat:"Saturated Hydraulic Conductivity",lai:"Leaf Area Index",lailut:"Leaf Area Index on Land-Use Tile",landcoverfrac:"Percentage of Area by Vegetation or Land-Cover Category",lara:"Temperature lapse rate between 850-hPa and 500-hPa",lcc:"Low Cloud Cover",ldai:"Large-scale Dynamical Activity Index",libmassbffl:"Basal Specific Mass Balance Flux of Floating Ice Shelf",libmassbfgr:"Basal Specific Mass Balance Flux of Grounded Ice Sheet",licalvf:"Land Ice Calving Flux",lifmassbf:"Land Ice Vertical Front Mass Balance Flux",lim:"Ice Sheet Mass",limfecalc:"Iron Limitation of Calcareous Phytoplankton",limfediat:"Iron Limitation of Diatoms",limfediaz:"Iron Limitation of Diazotrophs",limfemisc:"Iron Limitation of Other Phytoplankton",limfepico:"Iron Limitation of Picophytoplankton",limin:"Minimum LI of parcels lifted from 925-, 850-, and 700-hPa",limirrcalc:"Irradiance Limitation of Calcareous Phytoplankton",limirrdiat:"Irradiance Limitation of Diatoms",limirrdiaz:"Irradiance Limitation of Diazotrophs",limirrmisc:"Irradiance Limitation of Other Phytoplankton",limirrpico:"Irradiance Limitation of Picophytoplankton",limncalc:"Nitrogen Limitation of Calcareous Phytoplankton",limndiat:"Nitrogen Limitation of Diatoms",limndiaz:"Nitrogen Limitation of Diazotrophs",limnmisc:"Nitrogen Limitation of Other Phytoplankton",limnpico:"Nitrogen Limitation of Picophytoplankton",limnsw:"Ice Sheet Mass That Does not Displace Sea Water",litempbotfl:"Basal Temperature of Floating Ice Shelf",litempbotgr:"Basal Temperature of Grounded Ice Sheet",litemptop:"Temperature at Top of Ice Sheet Model",litemptopis:"Ice Sheet Temperature at Top of Ice Sheet Model",lithk:"Ice Sheet Thickness",lnsp:"Logarithm of Surface Pressure",loadbc:"Load of Black Carbon Aerosol",loaddust:"Load of Dust",loadnh4:"Load of NH4",loadno3:"Load of NO3",loadoa:"Load of Dry Aerosol Organic Matter",loadpoa:"Load of Dry Aerosol Primary Organic Matter",loadso4:"Load of SO4",loadsoa:"Load of Dry Aerosol Secondary Organic Matter",loadss:"Load of Sea-Salt Aerosol",lossch4:"Monthly Loss of Atmospheric Methane",lossco:"Monthly Loss of Atmospheric Carbon Monoxide",lossn2o:"Monthly Loss of Atmospheric Nitrous Oxide",lspf:"Large-scale Precipitation Fraction",lwp:"Liquid Water Path",lwsffluxaero:"Longwave Flux Due to Volcanic Aerosols at the Surface",lwsnl:"Liquid Water Content of Snow Layer",lwsrfasdust:"All-Sky Surface Longwave Radiative Flux Due to Dust",lwsrfcsdust:"Clear-Sky Surface Longwave Radiative Flux Due to Dust",lwtoaasdust:"TOA All-Sky Longwave Radiative Forcing Due to Dust",lwtoacsaer:"TOA Clear-Sky Longwave Radiative Forcing Due to Aerosols",lwtoacsdust:"TOA Clear-Sky Longwave Radiative Forcing Due to Dust",lwtoafluxaerocs:"TOA Outgoing Clear-Sky Longwave Flux Due to Volcanic Aerosols",masscello:"Ocean Grid-Cell Mass per Area",masso:"Sea Water Mass",maxpblz:"Maximum PBL Height",mc:"Convective Mass Flux",mcc:"Medium Cloud Cover",mcd:"Downdraft Convective Mass Flux",mcu:"Convective Updraft Mass Flux",md:"Wet Diameter Mode Coarse Insoluble",meanage:"Mean Age of Stratospheric Air",mfo:"Sea Water Transport",minpblz:"Minimum PBL Height",mlotst:"Ocean Mixed Layer Thickness Defined by Sigma T",mlotstmax:"Maximum Ocean Mixed Layer Thickness Defined by Sigma T",mlotstmin:"Minimum Ocean Mixed Layer Thickness Defined by Sigma T",mlotstsq:"Square of Ocean Mixed Layer Thickness Defined by Sigma T",mmraerh2o:"Aerosol Water Mass Mixing Ratio",mmrbc:"Elemental Carbon Mass Mixing Ratio",mmrdust:"Dust Aerosol Mass Mixing Ratio",mmrnh4:"NH4 Mass Mixing Ratio",mmrno3:"NO3 Aerosol Mass Mixing Ratio",mmroa:"Total Organic Aerosol Mass Mixing Ratio",mmrpm1:"PM1.0 Mass Mixing Ratio",mmrpm10:"PM10 Mass Mixing Ratio",mmrpm2p5:"PM2.5 Mass Mixing Ratio",mmrso4:"Aerosol Sulfate Mass Mixing Ratio",mmrsoa:"Secondary Organic Aerosol Mass Mixing Ratio",mmrss:"Sea-Salt Aerosol Mass Mixing Ratio",modelcellareai:"The Cell Area of the Ice Sheet Model",mrfso:"Soil Frozen Water Content",mrfsofr:"Average Layer Fraction of Frozen Moisture",mrlqso:"Average Layer Fraction of Liquid Moisture",mrlsl:"Water Content per Unit Area of Soil Layers",mrlso:"Soil Liquid Water Content",mrro:"Total Runoff",mrrob:"Subsurface Runoff",mrrois:"Ice Sheet Total Runoff",mrroli:"Land Ice Runoff Flux",mrrolut:"Total Runoff from Land-Use Tile",mrros:"Surface Runoff",mrsfl:"Frozen Water Content of Soil Layer",mrsll:"Liquid Water Content of Soil Layer",mrso:"Total Soil Moisture Content",mrso1:"Upper 1-Meter Soil Moisture Content",mrsofc:"Capacity of Soil to Store Water (Field Capacity)",mrsol:"Total Water Content of Soil Layer",mrsolut:"Total Soil Moisture",mrsopufc:"Total soil moisture content in percent of usable field capacity for soil layers in 0-0.6 m",mrsos:"Moisture in Upper Portion of Soil Column",mrsosat:"Soil Moisture Saturation Point",mrsoslut:"Moisture in Upper Portion of Soil Column of Land-Use Tile",mrsow:"Total Soil Wetness",mrsowp:"Soil Moisture Wilting Point",mrtws:"Terrestrial Water Storage",msftbarot:"Ocean Barotropic Mass Streamfunction",msftmrho:"Ocean Meridional Overturning Mass Streamfunction",msftmrhompa:"Ocean Meridional Overturning Mass Streamfunction Due to Parameterized Mesoscale Advection",msftmrhoz:"Ocean Meridional Overturning Mass Streamfunction",msftmrhozba:"Ocean Meridional Overturning Mass Streamfunction due to Bolus Advection",msftmyz:"Ocean Meridional Overturning Mass Streamfunction",msftmyzba:"Ocean Meridional Overturning Mass Streamfunction due to Bolus Advection",msftmz:"Ocean Meridional Overturning Mass Streamfunction",msftmzmpa:"Ocean Meridional Overturning Mass Streamfunction Due to Parameterized Mesoscale Advection",msftmzsmpa:"Ocean Meridional Overturning Mass Streamfunction Due to Parameterized Submesoscale Advection",msftyrho:"Ocean Y Overturning Mass Streamfunction",msftyrhompa:"Ocean Y Overturning Mass Streamfunction Due to Parameterized Mesoscale Advection",msftyrhoz:"Ocean Y Overturning Mass Streamfunction",msftyrhozba:"Ocean Y Overturning Mass Streamfunction due to Bolus Advection",msftyyz:"Ocean Y Overturning Mass Streamfunction",msftyyzba:"Ocean Y Overturning Mass Streamfunction due to Bolus Advection",msftyz:"Ocean Y Overturning Mass Streamfunction",msftyzmpa:"Ocean Y Overturning Mass Streamfunction Due to Parameterized Mesoscale Advection",msftyzsmpa:"Ocean Y Overturning Mass Streamfunction Due to Parameterized Submesoscale Advection",n2o:"Mole Fraction of N2O",n2oclim:"Mole Fraction of N2O",n2oglobal:"Global Mean Mole Fraction of N2O",n2oglobalclim:"Global Mean Mole Fraction of N2O",nbp:"Carbon Mass Flux out of Atmosphere Due to Net Biospheric Production on Land",necblut:"Net Carbon Mass Flux into Land-Use Tile",nep:"Net Carbon Mass Flux out of Atmosphere Due to Net Ecosystem Productivity on Land",netatmoslandc13flux:"Net Mass Flux of 13C Between Atmosphere and Land (Positive into Land) as a Result of All Processes",netatmoslandc14flux:"Net Mass Flux of 14C Between Atmosphere and Land (Positive into Land) as a Result of All Processes",netatmoslandco2flux:"Net Flux of CO2 Between Atmosphere and Land (Positive into Land) as a Result of All Processes",nh4:"Dissolved Ammonium Concentration",nh4os:"Surface Dissolved Ammonium Concentration",nh50:"Artificial Tracer with 50 Day Lifetime",nland:"Total Nitrogen in All Terrestrial Nitrogen Pools",nleaf:"Nitrogen Mass in Leaves",nlitter:"Nitrogen Mass in Litter Pool",nlittercwd:"Nitrogen Mass in Coarse Woody Debris",nlittersubsurf:"Nitrogen Mass in Below-Ground Litter (non CWD)",nlittersurf:"Nitrogen Mass in Above-Ground Litter (non CWD)",nmineral:"Mineral Nitrogen in the Soil",nmineralnh4:"Mineral Ammonium in the Soil",nmineralno3:"Mineral Nitrate in the Soil",no:"NO Volume Mixing Ratio",no2:"NO2 Volume Mixing Ratio",no3:"Dissolved Nitrate Concentration",no33d:"Dissolved Nitrate Concentration",no3os:"Surface Dissolved Nitrate Concentration",nother:"Nitrogen Mass in Vegetation Components Other than Leaves, Stem and Root",noy:"Total Reactive Nitrogen Volume Mixing Ratio",npp:"Net Primary Production on Land as Carbon Mass Flux",nppgrass:"Net Primary Production on Grass Tiles as Carbon Mass Flux",nppleaf:"Net Primary Production Allocated to Leaves as Carbon Mass Flux",npplut:"Net Primary Production on Land-Use Tile as Carbon Mass Flux",nppother:"Net Primary Production Allocated to Other Pools (not Leaves Stem or Roots) as Carbon Mass Flux",npproot:"Net Primary Production Allocated to Roots as Carbon Mass Flux",nppshrub:"Net Primary Production on Shrub Tiles as Carbon Mass Flux",nppstem:"Net Primary Production Allocated to Stem as Carbon Mass Flux",npptree:"Net Primary Production on Tree Tiles as Carbon Mass Flux",nppwood:"Net Primary Production Allocated to Wood as Carbon Mass Flux",nproduct:"Nitrogen Mass in Products of Land-Use Change",nroot:"Nitrogen Mass in Roots",nsoil:"Nitrogen Mass in Soil Pool",nstem:"Nitrogen Mass in Stem",nudgincsm:"Nudging Increment of Water in Soil Moisture",nudgincswe:"Nudging Increment of Water in Snow",nveg:"Nitrogen Mass in Vegetation",nwdfraclut:"Non-Woody Vegetation Percentage Cover",o2:"Dissolved Oxygen Concentration",o23d:"Dissolved Oxygen Concentration",o2min:"Oxygen Minimum Concentration",o2os:"Surface Dissolved Oxygen Concentration",o2sat:"Dissolved Oxygen Concentration at Saturation",o2satos:"Surface Dissolved Oxygen Concentration at Saturation",o3:"Ozone Mass Mixing Ratio",o3clim:"Mole Fraction of O3",o3loss:"O3 Destruction Rate",o3prod:"O3 Production Rate",o3ste:"Stratospheric Ozone Tracer Volume Mixing Ratio",obvfsq:"Square of Brunt Vaisala Frequency in Sea Water",ocfriver:"Flux of Organic Carbon into Ocean Surface by Runoff",ocontempdiff:"Tendency of Sea Water Conservative Temperature Expressed as Heat Content Due to Parameterized Dianeutral Mixing",ocontempmint:"Depth Integral of Product of Sea Water Density and Conservative Temperature",ocontemppadvect:"Tendency of Sea Water Conservative Temperature Expressed as Heat Content Due to Parameterized Eddy Advection",ocontemppmdiff:"Tendency of Sea Water Conservative Temperature Expressed as Heat Content Due to Parameterized Mesoscale Diffusion",ocontemppsmadvect:"Tendency of Sea Water Conservative Temperature Expressed as Heat Content Due to Parameterized Submesoscale Advection",ocontemprmadvect:"Tendency of Sea Water Conservative Temperature Expressed as Heat Content Due to Residual Mean Advection",ocontemptend:"Tendency of Sea Water Conservative Temperature Expressed as Heat Content",od440aer:"Ambient Aerosol Optical Thickness at 440nm",od443dust:"Optical Thickness at 443nm Dust",od550aer:"Ambient Aerosol Optical Thickness at 550nm",od550aerh2o:"Aerosol Water Optical Thickness at 550nm",od550aerso:"Stratospheric Optical Depth at 550nm (All Aerosols) 2D-Field (Stratosphere Only)",od550bb:"Aerosol Optical Depth at 550nm Due to Biomass Burning",od550bc:"Black Carbon Optical Thickness at 550nm",od550csaer:"Ambient Aerosol Optical Thickness at 550nm",od550dust:"Dust Optical Thickness at 550nm",od550lt1aer:"Ambient Fine Aerosol Optical Depth at 550nm",od550no3:"Nitrate Aerosol Optical Depth at 550nm",od550oa:"Total Organic Aerosol Optical Depth at 550nm",od550so4:"Sulfate Aerosol Optical Depth at 550nm",od550so4so:"Stratospheric Optical Depth at 550nm (Sulphate Only) 2D-Field (Stratosphere Only)",od550soa:"Particulate Organic Aerosol Optical Depth at 550nm",od550ss:"Sea-Salt Aerosol Optical Depth at 550nm",od865dust:"Dust Optical Depth at 865nm",od870aer:"Ambient Aerosol Optical Depth at 870nm",oh:"OH Volume Mixing Ratio",ohc2000m:"",ohc300m:"",ohc700m:"",omldamax:"Mean Daily Maximum Ocean Mixed Layer Thickness Defined by Mixing Scheme",omlmax:"Monthly Maximum Ocean Mixed Layer Thickness Defined by Mixing Scheme",opottempdiff:"Tendency of Sea Water Potential Temperature Expressed as Heat Content Due to Parameterized Dianeutral Mixing",opottempmint:"Integral with Respect to Depth of Product of Sea Water Density and Potential Temperature",opottemppadvect:"Tendency of Sea Water Potential Temperature Expressed as Heat Content Due to Parameterized Eddy Advection",opottemppmdiff:"Tendency of Sea Water Potential Temperature Expressed as Heat Content Due to Parameterized Mesoscale Diffusion",opottemppsmadvect:"Tendency of Sea Water Potential Temperature Expressed as Heat Content Due to Parameterized Submesoscale Advection",opottemprmadvect:"Tendency of Sea Water Potential Temperature Expressed as Heat Content Due to Residual Mean Advection",opottemptend:"Tendency of Sea Water Potential Temperature Expressed as Heat Content",orog:"Surface Altitude",orogis:"Ice Sheet Surface Altitude",osaltdiff:"Tendency of Sea Water Salinity Expressed as Salt Content Due to Parameterized Dianeutral Mixing",osaltpadvect:"Tendency of Sea Water Salinity Expressed as Salt Content Due to Parameterized Eddy Advection",osaltpmdiff:"Tendency of Sea Water Salinity Expressed as Salt Content Due to Parameterized Mesoscale Diffusion",osaltpsmadvect:"Tendency of Sea Water Salinity Expressed as Salt Content Due to Parameterized Submesoscale Advection",osaltrmadvect:"Tendency of Sea Water Salinity Expressed as Salt Content Due to Residual Mean Advection",osalttend:"Tendency of Sea Water Salinity Expressed as Salt Content",oxloss:"Total Odd Oxygen (Ox) Loss Rate",oxprod:"Total Odd Oxygen (Ox) Production Rate",pabigthetao:"Sea Water Added Conservative Temperature",pan:"PAN Volume Mixing Ratio",parag:"Aragonite Production",parasolrefl:"PARASOL Reflectance",pasturefrac:"Percentage of Land Which Is Anthropogenic Pasture",pasturefracc3:"C3 Pasture Area Percentage",pasturefracc4:"C4 Pasture Area Percentage",pathetao:"Sea Water Additional Potential Temperature",pbfe:"Biogenic Iron Production",pbo:"Sea Water Pressure at Sea Floor",pbsi:"Biogenic Silicon Production",pcalc:"Calcite Production",pco2_therm:"Sea Surface Temperature",pctisccp:"ISCCP Mean Cloud Top Pressure",pdi:"Diatom Primary Carbon Production",pflw:"Liquid Water Content of Permafrost Layer",pfull:"Pressure at Model Full-Levels",ph:"pH",ph3d:"pH",phabio:"Abiotic pH",phabioos:"Surface Abiotic pH",phail:"Mean 6-hr probability of hail (2 cm)",phalf:"Pressure on Model Half-Levels",phnat:"Natural pH",phnatos:"Surface Natural pH",phos:"Surface pH",photo1d:"Photolysis Rate of Ozone (O3) to Excited Atomic Oxygen (the Singlet D State, O1D)",phyc:"Phytoplankton Carbon Concentration",phyc3d:"Phytoplankton Carbon Concentration",phycalc:"Mole Concentration of Calcareous Phytoplankton Expressed as Carbon in Sea Water",phycalcos:"Surface Mole Concentration of Calcareous Phytoplankton Expressed as Carbon in Sea Water",phycos:"Sea Surface Phytoplankton Carbon Concentration",phydiat:"Mole Concentration of Diatoms Expressed as Carbon in Sea Water",phydiatos:"Surface Mole Concentration of Diatoms Expressed as Carbon in Sea Water",phydiaz:"Mole Concentration of Diazotrophs Expressed as Carbon in Sea Water",phydiazos:"Surface Mole Concentration of Diazotrophs Expressed as Carbon in Sea Water",phyfe:"Mole Concentration of Total Phytoplankton Expressed as Iron in Sea Water",phyfeos:"Surface Mole Concentration of Total Phytoplankton Expressed as Iron in Sea Water",phymisc:"Mole Concentration of Miscellaneous Phytoplankton Expressed as Carbon in Sea Water",phymiscos:"Surface Mole Concentration of Miscellaneous Phytoplankton Expressed as Carbon in Sea Water",phyn:"Mole Concentration of Total Phytoplankton Expressed as Nitrogen in Sea Water",phynos:"Surface Mole Concentration of Phytoplankton Nitrogen in Sea Water",phyp:"Mole Concentration of Total Phytoplankton Expressed as Phosphorus in Sea Water",phypico:"Mole Concentration of Picophytoplankton Expressed as Carbon in Sea Water",phypicoos:"Surface Mole Concentration of Picophytoplankton Expressed as Carbon in Sea Water",phypmisc:"Other Phytoplankton Carbon Production",phypos:"Surface Mole Concentration of Total Phytoplankton Expressed as Phosphorus in Sea Water",physi:"Mole Concentration of Total Phytoplankton Expressed as Silicon in Sea Water",physios:"Surface Mole Concentration of Total Phytoplankton Expressed as Silicon in Sea Water",plightning:"Mean 6-hr probability of lightning",pnitrate:"Primary Carbon Production by Phytoplankton Due to Nitrate Uptake Alone",po4:"Total Dissolved Inorganic Phosphorus Concentration",po43d:"Dissolved Phosphate Concentration",po4os:"Surface Total Dissolved Inorganic Phosphorus Concentration",pod0:"Phytotoxic Ozone Dose",pon:"Mole Concentration of Particulate Organic Matter Expressed as Nitrogen in Sea Water",ponos:"Surface Mole Concentration of Particulate Organic Matter Expressed as Nitrogen in Sea Water",pop:"Mole Concentration of Particulate Organic Matter Expressed as Phosphorus in Sea Water",popos:"Surface Mole Concentration of Particulate Organic Matter Expressed as Phosphorus in Sea Water",pp:"Primary Carbon Production by Phytoplankton",ppcalc:"Net Primary Mole Productivity of Carbon by Calcareous Phytoplankton",ppdiat:"Net Primary Organic Carbon Production by Diatoms",ppdiaz:"Net Primary Mole Productivity of Carbon by Diazotrophs",ppmisc:"Net Primary Organic Carbon Production by Other Phytoplankton",ppos:"Primary Carbon Production by Phytoplankton",pppico:"Net Primary Mole Productivity of Carbon by Picophytoplankton",pr:"Precipitation",pr17o:"Precipitation Flux of Water Containing Oxygen-17 (H2 17O)",pr18o:"Precipitation Flux of Water Containing Oxygen-18 (H2 18O)",pr2h:"Precipitation Flux of Water Containing Deuterium (1H 2H O)",pradjust:"Bias-Corrected Precipitation",prbigthetao:"Sea Water Redistributed Conservative Temperature",prc:"Convective Precipitation",prcprof:"Convective Rainfall Flux",prcptotetccdi:"Annual Total Precipitation in Wet Days",prcrop:"Precipitation over Crop Tile",prcsh:"Precipitation Flux from Shallow Convection",prhmax:"Maximum Hourly Precipitation Rate",prl:"Large Scale Precipitation",prlsns:"Stratiform Snowfall Flux",prlsprof:"Stratiform Rainfall Flux",prnobs:"number of observations (stations/satellite pixels) per grid cell",prnumd:"Precipitation Number of Days with Observations",probci:"",problimin:"Lightning Probability",prra:"Rainfall Flux",prrais:"Ice Sheet Rainfall Rate",prrc:"Convective Rainfall Rate",prrsn:"Fraction of Rainfall on Snow",prsatm:"Precipitation Satellite Bit Mask",prsn:"Snowfall Flux",prsn17o:"Precipitation Flux of Snow and Ice Containing Oxygen-17 (H2 17O)",prsn18o:"Precipitation Flux of Snow and Ice Containing Oxygen-18 (H2 18O)",prsn2h:"Precipitation Flux of Snow and Ice Containing Deuterium (1H 2H O)",prsnc:"Convective Snowfall Flux",prsnis:"Ice Sheet Snowfall Flux",prsnsn:"Fraction of Snowfall (Including Hail and Graupel) on Snow",prstddev:"Sea Surface Temperature Standard Deviation",prstderr:"Precipitation Standard Error",prthetao:"Sea Water Redistributed Potential Temperature",prveg:"Precipitation onto Canopy",prw:"Water Vapor Path",prw17o:"Mass of Water Vapor Containing Oxygen-17 (H2 17O) in Layer",prw18o:"Mass of Water Vapor Containing Oxygen-18 (H2 18O) in Layer",prw2h:"Mass of Water Containing Deuterium (1H 2H O) in Layer",prwerr:"Total Column Water Vapour Uncertainty of Retrieval Method",prwflag:"GlobVapour Area Type Flag",prwnumd:"Water Vapor Path Number of Days with Observations",prwsatm:"Water Vapor Path Satellite Bit Mask",prwstddev:"Water Vapor Path Standard Deviation",ps:"Surface Air Pressure",psitem:"Transformed Eulerian Mean Mass Streamfunction",psl:"Sea Level Pressure",psladjust:"Bias-Corrected Sea Level Pressure",pslensmax:"psl at 0,*,* Ensemble Maximum",pslensmean:"psl at 0,*,* Ensemble Mean",pslensmin:"psl at 0,*,* Ensemble Minimum",pslensminus:"psl at 0,*,* Ensemble Mean Minus 1 Standard Deviation",pslensplus:"psl at 0,*,* Ensemble Mean Plus 1 Standard Deviation",pslensrange:"psl at 0,*,* Ensemble Range",pslensstdev:"psl at 0,*,* Ensemble Standard Deviation",pslensvld:"psl at 0,*,* Ensemble Valid Data Count",pso:"Sea Water Pressure at Sea Water Surface",ptornado:"Mean 6-hr probability of tornado (F1)",ptp:"Tropopause Air Pressure",pv:"Potential Vorticity",pwind:"Mean 6-hr probability of severe wind (25 m s**-1)",qgwr:"Groundwater Recharge from Soil Layer",r1:"Max. precip. amount for one day)",r10mm:"Heavy precipitation days is the number of days per time period with daily precipitation sum exceeding 10 mm. The time period should be defined by the bounds of the time coordinate.",r10mmetccdi:"Annual Count of Days with At Least 10mm of Precipitation",r20mm:"Very heavy precipitation days is the number of days with daily precipitation sum exceeding 20 mm. The time period should be defined by the bounds of the time coordinate.",r20mmetccdi:"Annual Count of Days with At Least 20mm of Precipitation",r5:"Max. precip. amount for five consecutive days)",r95p:"The 95th percentile of days with RR >= 1 mm",r95p_tot:"Annual total precipitation when RR > R95p",r95petccdi:"Annual Total Precipitation when Daily Precipitation Exceeds the 95th Percentile of Wet Day Precipitation",r95ptot:"Annual Total Precipitation when Daily Precipitation Exceeds the 95th Percentile of Wet Day Precipitation",r99p:"The 99th percentile of days with RR >= 1 mm",r99p_tot:"Annual total precipitation when RR > R99p",r99petccdi:"Annual Total Precipitation when Daily Precipitation Exceeds the 99th Percentile of Wet Day Precipitation",r99ptot:"Annual total precipitation when RR > R99p",ra:"Carbon Mass Flux into Atmosphere Due to Autotrophic (Plant) Respiration on Land",rac13:"Carbon-13 Mass Flux into Atmosphere Due to Autotrophic (Plant) Respiration on Land",rac14:"Carbon-14 Mass Flux into Atmosphere Due to Autotrophic (Plant) Respiration on Land",ragrass:"Autotrophic Respiration on Grass Tiles as Carbon Mass Flux",rainmxrat:"Mass Fraction of Rain in Air",rainmxrat27:"Mass Fraction of Rain in Air",raleaf:"Total Respiration from Leaves",ralut:"Autotrophic Respiration on Land-Use Tile as Carbon Mass Flux",raother:"Total Respiration from Other Pools (not Leaves Stem or Roots) as Carbon Mass Flux",raroot:"Total Respiration from Roots as Carbon Mass Flux",rashrub:"Autotrophic Respiration on Shrub Tiles as Carbon Mass Flux",rastem:"Total Respiration from Stem as Carbon Mass Flux",ratree:"Autotrophic Respiration on Tree Tiles as Carbon Mass Flux",reffcclwtop:"Cloud-Top Effective Droplet Radius in Convective Cloud",reffclic:"Hydrometeor Effective Radius of Convective Cloud Ice",reffclis:"Hydrometeor Effective Radius of Stratiform Cloud Ice",reffclwc:"Convective Cloud Liquid Droplet Effective Radius",reffclws:"Stratiform Cloud Liquid Droplet Effective Radius",reffclwtop:"Cloud-Top Effective Droplet Radius",reffgrpls:"Hydrometeor Effective Radius of Stratiform Graupel",reffrainc:"Hydrometeor Effective Radius of Convective Rainfall",reffrains:"Hydrometeor Effective Radius of Stratiform Rainfall",reffsclwtop:"Cloud-Top Effective Droplet Radius in Stratiform Cloud",reffsnowc:"Hydrometeor Effective Radius of Convective Snowfall",reffsnows:"Hydrometeor Effective Radius of Stratiform Snowfall",remoc:"Remineralization of Organic Carbon",residualfrac:"Percentage of Grid Cell That Is Land but neither Vegetation Covered nor Bare Soil",rgrowth:"Total Autotrophic Respiration on Land as Carbon Mass Flux",rh:"Total Heterotrophic Respiration on Land as Carbon Mass Flux",rhc13:"Carbon-13 Mass Flux into Atmosphere Due to Heterotrophic Respiration on Land",rhc14:"Carbon-14 Mass Flux into Atmosphere Due to Heterotrophic Respiration on Land",rhgrass:"Heterotrophic Respiration on Grass Tiles as Carbon Mass Flux",rhlitter:"Carbon Mass Flux into Atmosphere Due to Heterotrophic Respiration from Litter on Land",rhlut:"Heterotrophic Respiration on Land-Use Tile as Carbon Mass Flux",rhopoto:"Sea Water Potential Density",rhs:"Near-Surface Relative Humidity",rhshrub:"Heterotrophic Respiration on Shrub Tiles as Carbon Mass Flux",rhsmax:"Surface Daily Maximum Relative Humidity",rhsmin:"Surface Daily Minimum Relative Humidity",rhsoil:"Carbon Mass Flux into Atmosphere Due to Heterotrophic Respiration from Soil on Land",rhtree:"Heterotrophic Respiration on Tree Tiles as Carbon Mass Flux",ridgice:"Sea Ice Ridging Rate",rivi:"River Inflow",rivo:"River Discharge",rld:"Downwelling Longwave Radiation",rld4co2:"Downwelling Longwave Radiation 4XCO2 Atmosphere",rldcs:"Downwelling Clear-Sky Longwave Radiation",rldcs4co2:"Downwelling Clear-Sky Longwave Radiation 4XCO2 Atmosphere",rlds:"Surface Downwelling Longwave Radiation",rldscs:"Surface Downwelling Clear-Sky Longwave Radiation",rldsis:"Ice Sheet Surface Downwelling Longwave Radiation",rldssi:"Downwelling Long Wave over Sea Ice",rldsstddev:"Surface Downwelling Longwave Radiation Standard Deviation",rlntds:"Surface Net Downward Longwave Radiation",rls:"Net Longwave Surface Radiation",rlscs:"Surface Net Downward Longwave Flux Assuming Clear Sky",rlu:"Upwelling Longwave Radiation",rlu4co2:"Upwelling Longwave Radiation 4XCO2 Atmosphere",rlucs:"Upwelling Clear-Sky Longwave Radiation",rlucs4co2:"Upwelling Clear-Sky Longwave Radiation 4XCO2 Atmosphere",rlus:"Surface Upwelling Longwave Radiation",rlusis:"Ice Sheet Surface Upwelling Longwave Radiation",rluslut:"Surface Upwelling Longwave on Land-Use Tile",rlusnumd:"Surface Upwelling Longwave Radiation Number of Days with Observations",rlussatm:"Surface Upwelling Longwave Radiation Satellite Bit Mask",rlussi:"Upward Long Wave over Sea Ice",rlusstddev:"Surface Upwelling Longwave Radiation Standard Deviation",rlut:"TOA Outgoing Longwave Radiation",rlut4co2:"TOA Outgoing Longwave Radiation 4XCO2 Atmosphere",rlutaf:"TOA Outgoing Aerosol-Free Longwave Radiation",rlutcs:"TOA Outgoing Clear-Sky Longwave Radiation",rlutcs4co2:"TOA Outgoing Clear-Sky Longwave Radiation 4XCO2 Atmosphere",rlutcsaf:"TOA Outgoing Clear-Sky, Aerosol-Free Longwave Radiation",rm7p80:"80th percentile of 7-day running mean of RR Precipitation amount",rm7p95:"95th percentile of 7-day running mean of RR Precipitation amount",rm7p98:"98th percentile of 7-day running mean of RR Precipitation amount",rmaint:"Carbon Mass Flux into Atmosphere Due to Maintenance Autotrophic Respiration on Land",rnn:"# days with RR > $lev mm.",rnn_tot:"Annual tot. precip. from days with RR > $lev mm",rnntot:"Annual tot. precip. from days with RR > $lev mm",rootd:"Maximum Root Depth",rootdsl:"Root Distribution",rr1:"Wet days index is the number of days per time period with daily precipitation of at least 1 mm. The time period should be defined by the bounds of the time coordinate.",rsd:"Downwelling Shortwave Radiation",rsd4co2:"Downwelling Shortwave Radiation 4XCO2 Atmosphere",rsdcs:"Downwelling Clear-Sky Shortwave Radiation",rsdcs4co2:"Downwelling Clear-Sky Shortwave Radiation 4XCO2 Atmosphere",rsdcsaf:"Downwelling Clear-Sky, Aerosol-Free Shortwave Radiation",rsdcsafbnd:"Downwelling Clear-Sky, Aerosol-Free, Shortwave Radiation in Bands",rsdcsbnd:"Downwelling Clear-Sky Shortwave Radiation at Each Level for Each Band",rsdo:"Downwelling Shortwave Radiation in Sea Water",rsdoabsorb:"Net Rate of Absorption of Shortwave Energy in Ocean Layer",rsds:"Surface Downwelling Shortwave Radiation",rsdscs:"Surface Downwelling Clear-Sky Shortwave Radiation",rsdscsaf:"Surface Downwelling Clear-Sky, Aerosol-Free Shortwave Radiation",rsdscsafbnd:"Surface Downwelling Clear-Sky, Aerosol-Free Shortwave Radiation in Bands",rsdscsbnd:"Surface Downwelling Clear-Sky Shortwave Radiation for Each Band",rsdscsdiff:"Surface Diffuse Downwelling Clear Sky Shortwave Radiation",rsdscsdms:"Surface Downwelling Clear-Sky Shortwave Radiation Number of Days with Observations",rsdscsstddev:"Surface Downwelling Clear-Sky Shortwave Radiation Standard Deviation",rsdsdiff:"Surface Diffuse Downwelling Shortwave Radiation",rsdsdms:"Surface Downwelling Shortwave Radiation Number of Days with Observations",rsdsis:"Ice Sheet Surface Downwelling Shortwave Radiation",rsdssi:"Downwelling Shortwave over Sea Ice",rsdsstddev:"Surface Downwelling Shortwave Radiation Standard Deviation",rsdt:"TOA Incident Shortwave Radiation",rsn:"Snow Density",rsntds:"Net Downward Shortwave Radiation at Sea Water Surface",rss:"Net Shortwave Surface Radiation",rsscs:"Surface Net Downward Shortwave Flux Assuming Clear Sky",rst:"TOA Net Downward Shortwave Flux",rstcs:"TOA Net Downward Shortwave Flux Assuming Clear Sky",rsu:"Upwelling Shortwave Radiation",rsu4co2:"Upwelling Shortwave Radiation 4XCO2 Atmosphere",rsucs:"Upwelling Clear-Sky Shortwave Radiation",rsucs4co2:"Upwelling Clear-Sky Shortwave Radiation 4XCO2 Atmosphere",rsucsaf:"Upwelling Clear-Sky, Aerosol-Free Shortwave Radiation",rsucsafbnd:"Upwelling Clear-Sky, Aerosol-Free Shortwave Radiation in Bands",rsucsbnd:"Upwelling Clear-Sky Shortwave Radiation at Each Level for Each Band",rsus:"Surface Upwelling Shortwave Radiation",rsuscs:"Surface Upwelling Clear-Sky Shortwave Radiation",rsuscsaf:"Surface Upwelling Clean Clear-Sky Shortwave Radiation",rsuscsafbnd:"Surface Upwelling Clear-Sky, Aerosol-Free Shortwave Radiation in Bands",rsuscsbnd:"Surface Upwelling Clear-Sky Shortwave Radiation for Each Band",rsusis:"Ice Sheet Surface Upwelling Shortwave Radiation",rsuslut:"Surface Upwelling Shortwave on Land-Use Tile",rsussi:"Upward Shortwave over Sea Ice",rsut:"TOA Outgoing Shortwave Radiation",rsut4co2:"TOA Outgoing Shortwave Radiation in 4XCO2 Atmosphere",rsutaf:"TOA Outgoing Aerosol-Free Shortwave Radiation",rsutcs:"TOA Outgoing Clear-Sky Shortwave Radiation",rsutcs4co2:"TOA Outgoing Clear-Sky Shortwave Radiation 4XCO2 Atmosphere",rsutcsaf:"TOA Outgoing Clear-Sky, Aerosol-Free Shortwave Radiation",rsutcsafbnd:"TOA Outgoing Clear-Sky, Aerosol-Free Shortwave Radiation in Bands",rsutcsbnd:"TOA Outgoing Clear-Sky Shortwave Radiation for Each Band",rtmt:"Net Downward Radiative Flux at Top of Model",rv:"Relative Vorticity",rv850:"Relative Vorticity at 850 hPa",rx1:"Max. precip. amount for one day)",rx1dayetccdi:"Annual Maximum 1-day Precipitation",rx1dayetccdiyrmax:"",rx5:"Max. precip. amount for five consecutive days)",rx5dayetccdi:"Annual Maximum Consecutive 5-day Precipitation",rzwc:"Root Zone Soil Moisture",sal:"Planetary Albedo",salstddev:"Planetary Albedo Standard Deviation",sandfrac:"Sand Fraction",sbl:"Surface Snow and Ice Sublimation Flux",sblis:"Ice Sheet Surface Snow and Ice Sublimation Flux",sblnosn:"Sublimation of the Snow Free Area",sblsi:"Sublimation over Sea Ice",sci:"Fraction of Time Shallow Convection Occurs",scldncl:"Cloud Droplet Number Concentration of Stratiform Cloud Tops",sconcbc:"Surface Concentration of Black Carbon Aerosol",sconcdust:"Surface Concentration of Dust",sconcnh4:"Surface Concentration of NH4",sconcno3:"Surface Concentration of NO3",sconcoa:"Surface Concentration of Dry Aerosol Organic Matter",sconcpoa:"Surface Concentration of Dry Aerosol Primary Organic Matter",sconcso4:"Surface Concentration of SO4",sconcsoa:"Surface Concentration of Dry Aerosol Secondary Organic Matter",sconcss:"Surface Concentration of Sea-Salt Aerosol",sdii:"Simple precipitation intensity index",sdiietccdi:"Simple Precipitation Intensity Index",sdor:"Standard Deviation of Orography",sds:"Storm Days is the number of days per year where the maximum gust speed is above 17.2m s-1",sdsw:"Winter Storm Days is the number of days per winter ONDJFM where the maximum gust speed is above 17.2m s-1",seaice:"Sea Ice Area Fraction",sedustci:"Sedimentation Flux of Dust Mode Coarse Insoluble",sf6:"Mole Concentration of SF6 in sea water",sfcwind:"Near-Surface Wind Speed",sfcwind98:"Number of days per year where the wind speed is above the 98th percentile",sfcwind98w:"Number of days per winter ONDJFM where the wind speed is above the 98th percentile",sfcwindiqr:"Interquartile Range of Daily-Mean Near-Surface Wind Speed",sfcwindmax:"Daily Maximum Near-Surface Wind Speed",sfcwindnobs:"Near-Surface Wind Speed Number of Observations",sfcwindnumd:"Near-Surface Wind Speed Number of Days with Observations",sfcwindsatm:"Near-Surface Wind Speed Satellite Bit Mask",sfcwindstddev:"Near-Surface Wind Speed Standard Deviation",sfcwindstderr:"Near-Surface Wind Speed Standard Error",sfdsi:"Downward Sea Ice Basal Salt Flux",sfno2:"NO2 Volume Mixing Ratio in Lowest Model Layer",sfo3:"O3 Volume Mixing Ratio in Lowest Model Layer",sfo3max:"Daily Maximum O3 Volume Mixing Ratio in Lowest Model Layer",sfpm25:"PM2.5 Mass Mixing Ratio in Lowest Model Layer",sfriver:"Salt Flux into Sea Water from Rivers",sftflf:"Floating Ice Shelf Area Percentage",sftgif:"Land Ice Area Percentage",sftgrf:"Grounded Ice Sheet Area Percentage",sftlf:"Percentage of the Grid Cell Occupied by Land (Including Lakes)",sftof:"Sea Area Percentage",shrubfrac:"Percentage Cover by Shrub",si:"Total Dissolved Inorganic Silicon Concentration",si3d:"Dissolved Silicate Concentration",siage:"Age of Sea Ice",siareaacrossline:"Sea-Ice Area Flux Through Straits",siarean:"Sea-Ice Area North",siareas:"Sea-Ice Area South",sic:"Sea Ice Area Fraction",sicompstren:"Compressive Sea Ice Strength",siconc:"Sea-Ice Area Percentage (Ocean Grid)",siconca:"Sea-Ice Area Percentage (Atmospheric Grid)",sicstderr:"sea ice concentration standard error",sidconcdyn:"Sea-Ice Area Percentage Tendency Due to Dynamics",sidconcth:"Sea-Ice Area Percentage Tendency Due to Thermodynamics",sidivvel:"Divergence of the Sea-Ice Velocity Field",sidmassdyn:"Sea-Ice Mass Change from Dynamics",sidmassevapsubl:"Sea-Ice Mass Change Through Evaporation and Sublimation",sidmassgrowthbot:"Sea-Ice Mass Change Through Basal Growth",sidmassgrowthwat:"Sea-Ice Mass Change Through Growth in Supercooled Open Water (Frazil)",sidmasslat:"Lateral Sea Ice Melt Rate",sidmassmeltbot:"Sea-Ice Mass Change Through Bottom Melting",sidmassmelttop:"Sea-Ice Mass Change Through Surface Melting",sidmasssi:"Sea-Ice Mass Change Through Snow-to-Ice Conversion",sidmassth:"Sea-Ice Mass Change from Thermodynamics",sidmasstranx:"X-Component of Sea-Ice Mass Transport",sidmasstrany:"Y-Component of Sea-Ice Mass Transport",sidragbot:"Ocean Drag Coefficient",sidragtop:"Atmospheric Drag Coefficient",siextentn:"Sea-Ice Extent North",siextents:"Sea-Ice Extent South",sifb:"Sea-Ice Freeboard",siflcondbot:"Net Conductive Heat Fluxes in Ice at the Bottom",siflcondtop:"Net Conductive Heat Flux in Ice at the Surface",siflfwbot:"Freshwater Flux from Sea Ice",siflfwdrain:"Freshwater Flux from Sea-Ice Surface",sifllatstop:"Net Latent Heat Flux over Sea Ice",sifllwdtop:"Downwelling Longwave Flux over Sea Ice",sifllwutop:"Upwelling Longwave Flux over Sea Ice",siflsenstop:"Net Upward Sensible Heat Flux over Sea Ice",siflsensupbot:"Net Upward Sensible Heat Flux Under Sea Ice",siflswdbot:"Downwelling Shortwave Flux Under Sea Ice",siflswdtop:"Downwelling Shortwave Flux over Sea Ice",siflswutop:"Upwelling Shortwave Flux over Sea Ice",siforcecoriolx:"Coriolis Force Term in Force Balance (X-Component)",siforcecorioly:"Coriolis Force Term in Force Balance (Y-Component)",siforceintstrx:"Internal Stress Term in Force Balance (X-Component)",siforceintstry:"Internal Stress Term in Force Balance (Y-Component)",siforcetiltx:"Sea-Surface Tilt Term in Force Balance (X-Component)",siforcetilty:"Sea-Surface Tilt Term in Force Balance (Y-Component)",sihc:"Sea-Ice Heat Content per Unit Area",siitdconc:"Sea-Ice Area Percentages in Thickness Categories",siitdsnconc:"Snow Area Percentages in Ice Thickness Categories",siitdsnthick:"Snow thickness in thickness categories",siitdthick:"Sea-Ice Thickness in Thickness Categories",siltfrac:"Silt Fraction",sim:"Sea Ice Plus Surface Snow Amount",simass:"Sea-Ice Mass per Area",simassacrossline:"Sea Mass Area Flux Through Straits",simpconc:"Percentage Cover of Sea Ice by Meltpond",simpmass:"Meltpond Mass per Unit Area (as Depth)",simprefrozen:"Thickness of Refrozen Ice on Melt Pond",sios:"Surface Total Dissolved Inorganic Silicon Concentration",sipr:"Rainfall Rate over Sea Ice",sirdgconc:"Percentage Cover of Sea Ice by Ridging",sirdgthick:"Ridged Ice Thickness",sisali:"Sea Ice Salinity",sisaltmass:"Mass of Salt in Sea Ice per Area",sishevel:"Maximum Shear of Sea-Ice Velocity Field",sisnconc:"Snow Area Percentage",sisnhc:"Snow Heat Content per Unit Area",sisnmass:"Snow Mass per Area",sisnthick:"Snow Thickness",sispeed:"Sea-Ice Speed",sistremax:"Maximum Shear Stress in Sea Ice",sistresave:"Average Normal Stress in Sea Ice",sistrxdtop:"X-Component of Atmospheric Stress on Sea Ice",sistrxubot:"X-Component of Ocean Stress on Sea Ice",sistrydtop:"Y-Component of Atmospheric Stress on Sea Ice",sistryubot:"Y-Component of Ocean Stress on Sea Ice",sit:"Sea Ice Thickness",sitempbot:"Temperature at Ice-Ocean Interface",sitempsnic:"Temperature at Snow-Ice Interface",sitemptop:"Surface Temperature of Sea Ice",sithick:"Sea Ice Thickness",sitimefrac:"Fraction of Time Steps with Sea Ice",sitstderr:"sea ice thickness estimated uncertainty",siu:"X-Component of Sea-Ice Velocity",siv:"Y-Component of Sea-Ice Velocity",sivol:"Sea-Ice Volume per Area",sivoln:"Sea-Ice Volume North",sivols:"Sea-Ice Volume South",skt:"Skin Temperature",sli:"Surface Lifted Index",slor:"Slope of Sub-gridscale Orography",sltbasin:"Northward Ocean Salt Transport",slthick:"Thickness of Soil Layers",sltnortha:"Atlantic Northward Ocean Salt Transport",sltovgyre:"Northward Ocean Salt Transport Due to Gyre",sltovovrt:"Northward Ocean Salt Transport Due to Overturning",smc:"Shallow Convective Mass Flux",snc:"Snow Area Percentage",sncis:"Ice Sheet Snow Cover Percentage",snd:"Snow Depth",sndmassdyn:"Snow Mass Rate of Change Through Advection by Sea-Ice Dynamics",sndmassmelt:"Snow Mass Rate of Change Through Melt",sndmasssi:"Snow Mass Rate of Change Through Snow-to-Ice Conversion",sndmasssnf:"Snow Mass Change Through Snow Fall",sndmasssubl:"Snow Mass Rate of Change Through Evaporation or Sublimation",sndmasswindrif:"Snow Mass Rate of Change Through Wind Drift of Snow",snicefreez:"Surface Snow and Ice Refreeze Flux",snicefreezis:"Ice Sheet Surface Snow and Ice Refreeze Flux",snicem:"Surface Snow and Ice Melt Flux",snicemis:"Ice Sheet Surface Snow and Ice Melt Flux",snl:"Surface Net Longwave Radiation",snlstddev:"Surface Net Longwave Radiation Standard Deviation",snm:"Surface Snow Melt",snmassacrossline:"Snow Mass Flux Through Straits",snmis:"Ice Sheet Surface Snow Melt",snmsl:"Water Flowing out of Snowpack",snomelt:"Snow Melt Rate",snotoice:"Snow-Ice Formation Rate",snowmxrat:"Mass Fraction of Snow in Air",snowmxrat27:"Mass Fraction of Snow in Air",snrefr:"Refreezing of Water in the Snow",snw:"Surface Snow Amount",snwc:"Snow Water Equivalent Intercepted by the Vegetation",so:"Sea Water Salinity",so2:"SO2 Volume Mixing Ratio",sob:"Sea Water Salinity at Sea Floor",soga:"Global Mean Sea Water Salinity",solbnd:"TOA Solar Irradiance for Each Band",somint:"Depth Integral of Product of Sea Water Density and Prognostic Salinity",sootsn:"Snow Soot Content",sos:"Sea Surface Salinity",sosga:"Global Average Sea Surface Salinity",sossq:"Square of Sea Surface Salinity",spco2:"Surface Aqueous Partial Pressure of CO2",spco2abio:"Abiotic Surface Aqueous Partial Pressure of CO2",spco2nat:"Natural Surface Aqueous Partial Pressure of CO2",spei:"spei",spi:"spi",src:"Skin Reservoir Content",ssi:"Sea Ice Salinity",sst:"sea surface temperature",stormtd:"track density of storms per season within 1000km radius",strack:"",strairx:"X-Component of Atmospheric Stress On Sea Ice",strairy:"Y-Component of Atmospheric Stress On Sea Ice",strbasemag:"Land Ice Basal Drag",streng:"Compressive Sea Ice Strength",strocnx:"X-Component of Ocean Stress On Sea Ice",strocny:"Y-Component of Ocean Stress On Sea Ice",su:"Summer days index is the number of days where maximum of temperature is above 25 degree Celsius. The time period should be defined by the bounds of the time coordinate.",suetccdi:"Number of Summer Days",sund:"Duration of Sunshine",sw:"Surface Water Storage",sw17o:"Isotopic Ratio of Oxygen-17 in Sea Water",sw18o:"Isotopic Ratio of Oxygen-18 in Sea Water",sw2h:"Isotopic Ratio of Deuterium in Sea Water",swelut:"Snow Water Equivalent on Land-Use Tile",swsffluxaero:"Shortwave Heating Rate Due to Volcanic Aerosols",swsrfasdust:"All-Sky Surface Shortwave Radiative Flux Due to Dust",swsrfcsdust:"Clear-Sky Surface Shortwave Radiative Flux Due to Dust",swtoaasdust:"All-Sky Shortwave Flux Due to Dust at Toa",swtoacsdust:"Clear Sky Shortwave Flux Due to Dust at Toa",swtoafluxaerocs:"TOA Outgoing Clear-Sky Shortwave Flux Due to Volcanic Aerosols",swvl1:"Volumetric Soil Water Layer 1",swvl2:"Volumetric Soil Water Layer 2",swvl3:"Volumetric Soil Water Layer 3",swvl4:"Volumetric Soil Water Layer 4",sza:"Solar Zenith Angle",t2:"Mean-Squared Air Temperature",t20d:"Depth of 20 degree Celsius Isotherm",ta:"Air Temperature",ta1000:"Air Temperature at 1000 hPa",ta200:"Air Temperature at 200 hPa",ta20000:"",ta250:"Air Temperature at 250 hPa",ta27:"Air Temperature",ta300:"Air Temperature at 300 hPa",ta400:"Air Temperature at 400 hPa",ta500:"Air Temperature",ta600:"Air Temperature at 600 hPa",ta700:"Air Temperature",ta7h:"Air Temperature",ta800:"Air Temperature at 800 hPa",ta850:"Air Temperature",ta900:"Air Temperature at 900 hPa",ta925:"Air Temperature at 925 hPa",ta950:"Air Temperature at 950 hPa",ta975:"Air Temperature at 975 hPa",talk:"Total Alkalinity",talk3d:"Total Alkalinity",talknat:"Natural Total Alkalinity",talknatos:"Surface Natural Total Alkalinity",talkos:"Surface Total Alkalinity",tanobs:"Air Temperature Number of Observations",tas:"Near-Surface Air Temperature","tas-reg11":"near_surface_temperature_anomaly","tas-reg12":"near_surface_temperature_anomaly","tas-reg13":"Near-Surface Air Temperature",tasadjust:"Bias-Corrected Near-Surface Air Temperature",tasensmax:"tas at 0,*,* Ensemble Maximum",tasensmean:"tas at 0,*,* Ensemble Mean",tasensmin:"tas at 0,*,* Ensemble Minimum",tasensminus:"tas at 0,*,* Ensemble Mean Minus 1 Standard Deviation",tasensplus:"tas at 0,*,* Ensemble Mean Plus 1 Standard Deviation",tasensrange:"tas at 0,*,* Ensemble Range",tasensstdev:"tas at 0,*,* Ensemble Standard Deviation",tasensvld:"tas at 0,*,* Ensemble Valid Data Count",tasiqr:"Interquartile Range of Near-Surface Air Temperature",tasis:"Ice Sheet Near-Surface Air Temperature",taslut:"Near-Surface Air Temperature on Land Use Tile",tasmax:"Maximum Near-Surface Air Temperature",tasmaxcrop:"Daily Maximum Near-Surface Air Temperature over Crop Tile",tasmaxiqr:"Interquartile Range of Daily Maximum Near-Surface Air Temperature",tasmaxx:"Daily Maximum Near-Surface Air Temperature",tasmin:"Minimum Near-Surface Air Temperature",tasmincrop:"Daily Minimum Near-Surface Air Temperature over Crop Tile",tasminiqr:"Interquartile Range of Daily Minimum Near-Surface Air Temperature",tastderr:"Air Temperature Standard Error",tatp:"Tropopause Air Temperature",tau:"Momentum Flux",tauu:"Surface Downward Eastward Wind Stress",tauucorr:"Surface Downward X Stress Correction",tauuo:"Sea Water Surface Downward X Stress",tauupbl:"Eastward Surface Stress from Planetary Boundary Layer Scheme",tauv:"Surface Downward Northward Wind Stress",tauvcorr:"Surface Downward Y Stress Correction",tauvo:"Sea Water Surface Downward Y Stress",tauvpbl:"Northward Surface Stress from Planetary Boundary Layer Scheme",tco3:"Total Column Ozone",tcs:"Vegetation Canopy Temperature",tcw:"Water Path",td2m:"Near-Surface Dew Point Temperature",tdps:"2m Dewpoint Temperature",tendacabf:"Total Surface Mass Balance Flux",tendlibmassbf:"Total Basal Mass Balance Flux",tendlicalvf:"Total Calving Flux",tgs:"Temperature of Bare Soil",thetao:"Sea Water Potential Temperature",thetaoga:"Global Average Sea Water Potential Temperature",thetaot:"Vertically Averaged Sea Water Potential Temperature",thetaot2000:"Depth Average Potential Temperature of Upper 2000m",thetaot300:"Depth Average Potential Temperature of Upper 300m",thetaot700:"Depth Average Potential Temperature of Upper 700m",thkcello:"Ocean Model Cell Thickness",tmelt:"Rate of Melt at Upper Surface of Sea Ice",tn10petccdi:"Percentage of Days when Daily Minimum Temperature is Below the 10th Percentile",tn90petccdi:"Percentage of Days when Daily Minimum Temperature is Above the 90th Percentile",tnhus:"Tendency of Specific Humidity",tnhusa:"Tendency of Specific Humidity Due to Advection",tnhusc:"Tendency of Specific Humidity Due to Convection",tnhusd:"Tendency of Specific Humidity Due to Numerical Diffusion",tnhusmp:"Tendency of Specific Humidity Due to Model Physics",tnhuspbl:"Tendency of Specific Humidity Due to Boundary Layer Mixing",tnhusscp:"Tendency of Specific Humidity Due to Stratiform Clouds and Precipitation",tnhusscpbl:"Tendency of Specific Humidity Due to Stratiform Cloud and Precipitation and Boundary Layer Mixing",tnkebto:"Tendency of Ocean Eddy Kinetic Energy Content Due to Parameterized Eddy Advection",tnkebto2d:"Tendency of Ocean Eddy Kinetic Energy Content Due to Parameterized Eddy Advection",tnnetccdi:"Annual Minimum of Daily Minimum Temperature",tnnetccdiyrmin:"",tnpeo:"Tendency of Ocean Potential Energy Content",tnpeot:"Tendency of Ocean Potential Energy Content Due to Tides",tnpeotb:"Tendency of Ocean Potential Energy Content Due to Background",tnsccw:"Tendency of Mass Fraction of Stratiform Cloud Condensed Water In Air",tnsccwa:"Tendency of Mass Fraction of Stratiform Cloud Condensed Water due to Advection",tnsccwacr:"Tendency of Mass Fraction of Stratiform Cloud Condensed Water due to Autoconversion to Rain",tnsccwacs:"Tendency of Mass Fraction of Stratiform Cloud Condensed Water due to Autoconversion to Snow",tnsccwbl:"Tendency of Mass Fraction of Stratiform Cloud Condensed Water In Air Due To Boundary Layer Mixing",tnsccwce:"Tendency of Mass Fraction of Stratiform Cloud Condensed Water due to Condensation and Evaporation",tnsccwcm:"Tendency of Mass Fraction of Stratiform Cloud Condensed Water In Air Due To Cloud Microphysics",tnsccwif:"Tendency of Mass Fraction of Stratiform Cloud Condensed Water due to Icefall",tnscli:"Tendency of Mass Fraction of Stratiform Cloud Ice In Air",tnsclia:"Tendency of Mass Fraction of Stratiform Cloud Ice due to Advection",tnscliag:"Tendency of Mass Fraction of Stratiform Cloud Ice due to Aggregation",tnsclias:"Tendency of Mass Fraction of Stratiform Cloud Ice due to Accretion to Snow",tnsclibfpcl:"Tendency of Mass Fraction of Stratiform Cloud Ice In Air Due To Bergeron Findeisen Process from Cloud Liquid",tnsclibl:"Tendency of Mass Fraction of Stratiform Cloud Ice In Air Due To Boundary Layer Mixing",tnsclicd:"Tendency of Mass Fraction of Stratiform Cloud Ice Due Convective Detrainment",tnsclicm:"Tendency of Mass Fraction of Stratiform Cloud Ice In Air Due To Cloud Microphysics",tnsclids:"Tendency of Mass Fraction of Stratiform Cloud Ice due to Deposition and Sublimation",tnscliemi:"Tendency of Mass Fraction of Stratiform Cloud Ice due to Evaporation of Melting Ice",tnsclihencl:"Tendency of Mass Fraction of Stratiform Cloud Ice due to Heterogeneous Nucleation From Cloud Liquid",tnsclihenv:"Tendency of Mass Fraction of Stratiform Cloud Ice due to Heterogeneous Nucleation From Water Vapor",tnsclihon:"Tendency of Mass Fraction of Stratiform Cloud Ice due to Homogeneous Nucleation",tnscliif:"Tendency of Mass Fraction of Stratiform Cloud Ice due to Icefall",tnsclimcl:"Tendency of Mass Fraction of Stratiform Cloud Ice due to Melting to Cloud Liquid",tnsclimr:"Tendency of Mass Fraction of Stratiform Cloud Ice due to Melting to Rain",tnscliricl:"Tendency of Mass Fraction of Stratiform Cloud Ice due to Riming From Cloud Liquid",tnsclirir:"Tendency of Mass Fraction of Stratiform Cloud Ice due to Riming From Rain",tnsclw:"Tendency of Mass Fraction of Stratiform Cloud Liquid Water In Air",tnsclwa:"Tendency of Mass Fraction of Stratiform Cloud Liquid Water due to Advection",tnsclwac:"Tendency of Mass Fraction of Stratiform Cloud Liquid Water due to Autoconversion",tnsclwar:"Tendency of Mass Fraction of Stratiform Cloud Liquid Water due to Accretion to Rain",tnsclwas:"Tendency of Mass Fraction of Stratiform Cloud Liquid Water due to Accretion to Snow",tnsclwbfpcli:"Tendency of Mass Fraction of Stratiform Cloud Liquid Water In Air Due To Bergeron Findeisen Process To Cloud Ice",tnsclwcd:"Tendency of Mass Fraction of Stratiform Cloud Liquid Water Due to Convective Detrainment",tnsclwce:"Tendency of Mass Fraction of Stratiform Cloud Liquid Water due to Condensation and Evaporation",tnsclwcm:"Tendency of Mass Fraction of Stratiform Cloud Liquid Water In Air Due To Cloud Microphysics",tnsclwhen:"Tendency of Mass Fraction of Stratiform Cloud Liquid Water due to Heterogeneous Nucleation",tnsclwhon:"Tendency of Mass Fraction of Stratiform Cloud Liquid Water due to Homogeneous Nucleation",tnsclwmi:"Tendency of Mass Fraction of Stratiform Cloud Liquid Water due to Melting From Cloud Ice",tnsclwri:"Tendency of Mass Fraction of Stratiform Cloud Liquid Water due to Riming",tnt:"Tendency of Air Temperature",tnta:"Tendency of Air Temperature Due to Advection",tntc:"Tendency of Air Temperature Due to Convection",tntd:"Tendency of Air Temperature Due to Numerical Diffusion",tntmp:"Tendency of Air Temperature Due to Model Physics",tntmp27:"Tendency of Air Temperature Due to Model Physics",tntnogw:"Temperature Tendency Due to Non-Orographic Gravity Wave Dissipation",tntogw:"Temperature Tendency Due to Orographic Gravity Wave Dissipation",tntpbl:"Tendency of Air Temperature Due to Boundary Layer Mixing",tntr:"Tendency of Air Temperature Due to Radiative Heating",tntr27:"Tendency of Air Temperature Due to Radiative Heating",tntr_lw:"Tendency of Air Temperature due to Longwave Radiative Heating",tntr_sw:"Tendency of Air Temperature due to Shortwave Radiative Heating",tntrl:"Tendency of Air Temperature Due to Longwave Radiative Heating",tntrl27:"Tendency of Air Temperature Due to Longwave Radiative Heating",tntrlcs:"Tendency of Air Temperature due to Clear Sky Longwave Radiative Heating",tntrs:"Tendency of Air Temperature Due to Shortwave Radiative Heating",tntrs27:"Tendency of Air Temperature Due to Shortwave Radiative Heating",tntrscs:"Tendency of Air Temperature due to Clear Sky Shortwave Radiative Heating",tntscp:"Tendency of Air Temperature Due to Stratiform Clouds and Precipitation",tntscpbl:"Tendency of Air Temperature Due to Stratiform Cloud and Precipitation and Boundary Layer Mixing",tnxetccdi:"Annual Maximum of Daily Minimum Temperature",tob:"Sea Water Potential Temperature at Sea Floor",toffset:"Offset Time",tomint:"integral wrt depth of product of sea water density and prognostic temperature",topg:"Bedrock Altitude",tos:"Sea Surface Temperature",tosanom:"Sea Surface Temperature Anomaly",tosga:"Global Average Sea Surface Temperature",tosnobs:"Sea Surface Temperature Number of Observations",tosnumd:"Sea Surface Temperature Number of Days with Observations",tossatm:"Sea Surface Temperature Satellite Bit Mask",tossq:"Square of Sea Surface Temperature",tosstddev:"Sea Surface Temperature Standard Deviation",tosstderr:"Sea Surface Temperature Standard Error",toz:"Total Column Ozone",tpf:"Permafrost Layer Thickness",tr:"Surface Radiative Temperature",tran:"Transpiration",transifs:"Sea Ice Mass Transport Through Fram Strait",transix:"X-Component of Sea Ice Mass Transport",transiy:"Y-Component of Sea Ice Mass Transport",trce:"Latent Heat Transfer Coefficient",trcenumd:"Latent Heat Transfer Coefficient Number of Days with Observations",trcesatm:"Latent Heat Transfer Coefficient Satellite Bit Mask",trcestddev:"Latent Heat Transfer Coefficient Standard Deviation",treefrac:"Tree Cover Percentage",treefracbdldcd:"Broadleaf Deciduous Tree Area Percentage",treefracbdlevg:"Broadleaf Evergreen Tree Area Percentage",treefracndldcd:"Needleleaf Deciduous Tree Area Percentage",treefracndlevg:"Needleleaf Evergreen Tree Area Percentage",treefracprimdec:"Percentage Cover by Primary Deciduous Tree",treefracprimever:"Percentage Cover by Primary Evergreen Trees",treefracsecdec:"Percentage Cover by Secondary Deciduous Trees",treefracsecever:"Percentage Cover by Secondary Evergreen Trees",trefht:"Reference height temperature",tretccdi:"Number of Tropical Nights",tro3:"Mole Fraction of O3",tro3nobs:"Mole Fraction of O3 Number of Observations",tro3stderr:"Mole Fraction of O3 Standard Error",tropoz:"Tropospheric Ozone Column",ts:"Surface Temperature",tsadjust:"Bias-Corrected Surface Temperature",tsice:"Surface Temperature of Sea Ice",tsis:"Ice Sheet Surface Temperature",tsl:"Temperature of Soil",tsl1:"Temperature of Soil 1",tsl2:"Temperature of Soil 2",tsl3:"Temperature of Soil 3",tsl4:"Temperature of Soil 4",tsland:"Land Surface Temperature",tslsi:"Surface Temperature Where Land or Sea Ice",tslsilut:"Surface Temperature on Landuse Tile",tsn:"Snow Internal Temperature",tsnint:"Temperature at Interface Between Sea Ice and Snow",tsnis:"Ice Sheet Snow Internal Temperature",tsns:"Snow Surface Temperature",tso:"Sea Surface Temperature",tsoilpools:"Turnover Rate of Each Model Soil Carbon Pool",ttop:"air temperature at cloud top",tvh:"Type of High Vegetation",tvl:"Type of Low Vegetation",twap:"Product of Air Temperature and Omega",twp:"Condensed Water Path",tx10petccdi:"Percentage of Days when Daily Maximum Temperature is Below the 10th Percentile",tx90petccdi:"Percentage of Days when Daily Maximum Temperature is Above the 90th Percentile",txnetccdi:"Annual Minimum of Daily Maximum Temperature",txxetccdi:"Annual Maximum of Daily Maximum Temperature",txxetccdiyrmax:"",u2:"Mean-Squared Eastward Wind Speed",ua:"Eastward Wind",ua10:"Eastward Wind at 10hPa",ua1000:"Eastward Wind at 1000 hPa",ua100m:"Eastward Wind at 100m",ua200:"Eastward Wind at 200 hPa",ua250:"Eastward Wind at 250 hPa",ua27:"Eastward Wind",ua300:"Eastward Wind at 300 hPa",ua400:"Eastward Wind at 400 hPa",ua500:"Eastward Wind at 500 hPa",ua600:"Eastward Wind at 600 hPa",ua700:"Eastward Wind at 700 hPa",ua7h:"Eastward Wind",ua800:"Eastward Wind at 800 hPa",ua850:"Eastward Wind at 850 hPa",ua900:"Eastward Wind at 900 hPa",ua925:"Eastward Wind at 925 hPa",ua950:"Eastward Wind at 950 hPa",ua975:"Eastward Wind at 975 hPa",uas:"Eastward Near-Surface Wind",uasnobs:"Eastward Near-Surface Wind Number of Observations",uasstderr:"Eastward Near-Surface Wind Standard Error",ugrido:"UGRID Grid Specification",umo:"Ocean Mass X Transport",uncalipso:"CALIPSO 3D Undefined fraction",uo:"Sea Water X Velocity",uqint:"Eastward Humidity Transport",usi:"X-Component of Sea Ice Velocity",ut:"Product of Air Temperature and Eastward Wind",utendepfd:"Tendency of Eastward Wind Due to Eliassen-Palm Flux Divergence",utendnogw:"Eastward Acceleration Due to Non-Orographic Gravity Wave Drag",utendnogw27:"Eastward Acceleration Due to Non-Orographic Gravity Wave Drag",utendogw:"Eastward Acceleration Due to Orographic Gravity Wave Drag",utendvtem:"Tendency of Eastward Wind Due to TEM Northward Advection and Coriolis Term",utendwtem:"Tendency of Eastward Wind Due to TEM Upward Advection",uv:"Product of Eastward Wind and Northward Wind",uvb:"Downward UV Radiation at the Surface",uwap:"Product of Eastward Wind and Omega",v2:"Mean-Squared Northward Wind Speed",va:"Northward Wind",va1000:"Northward Wind at 1000 hPa",va100m:"Northward Wind at 100m",va200:"Northward Wind at 200 hPa",va250:"Northward Wind at 250 hPa",va27:"Northward Wind",va300:"Northward Wind at 300 hPa",va400:"Northward Wind at 400 hPa",va500:"Northward Wind at 500 hPa",va600:"Northward Wind at 600 hPa",va700:"Northward Wind at 700 hPa",va7h:"Northward Wind",va800:"Northward Wind at 800 hPa",va850:"Northward Wind at 850 hPa",va900:"Northward Wind at 900 hPa",va925:"Northward Wind at 925 hPa",va950:"Northward Wind at 950 hPa",va975:"Northward Wind at 975 hPa",vas:"Northward Near-Surface Wind",vasnobs:"Northward Near-Surface Wind Number of Observations",vasstderr:"Northward Near-Surface Wind Standard Error",vegfrac:"Total Vegetated Percentage Cover",vegheight:"Height of the Vegetation Canopy",vegheightcrop:"Height of Crops",vegheightgrass:"Height of Grass",vegheightpasture:"Height of Pastures",vegheightshrub:"Height of Shrubs",vegheighttree:"Height of Trees",vmo:"Ocean Mass Y Transport",vmrox:"Mole Fraction of Odd Oxygen (O, O3 and O1D)",vo:"Sea Water Y Velocity",volcello:"Ocean Grid-Cell Volume",volo:"Sea Water Volume",vortmean:"Relative Vorticity",vqint:"Northward Humidity Transport",vsf:"Virtual Salt Flux into Sea Water",vsfcorr:"Virtual Salt Flux Correction",vsfevap:"Virtual Salt Flux into Sea Water Due to Evaporation",vsfpr:"Virtual Salt Flux into Sea Water Due to Rainfall",vsfriver:"Virtual Salt Flux into Sea Water from Rivers",vsfsit:"Virtual Salt Flux into Sea Water Due to Sea Ice Thermodynamics",vsi:"Y-Component of Sea Ice Velocity",vt:"Product of Air Temperature and Northward Wind",vt100:"Northward heat flux due to eddies",vtem:"Transformed Eulerian Mean Northward Wind",vtendnogw:"Northward Acceleration Due to Non-Orographic Gravity Wave Drag",vtendnogw27:"Northward Acceleration Due to Non-Orographic Gravity Wave Drag",vtendogw:"Northward Acceleration Due to Orographic Gravity Wave Drag",vwap:"Product of Northward Wind and Omega",wa:"Upward Air Velocity",wap:"Omega (=dp/dt)",wap2:"Mean-Squared Vertical Velocity (Omega)",wap27:"Omega (=dp/dt)",wap4:"Omega (=dp/dt)",wap500:"omega (=dp/dt)",wap7h:"Omega (=dp/dt)",wbptemp:"wet_bulb_potential_temperature",wbptemp7h:"Wet Bulb Potential Temperature",wetbc:"Wet Deposition Rate of Black Carbon Aerosol Mass",wetdms:"Wet Deposition Rate of DMS",wetdust:"Wet Deposition Rate of Dust",wetlandch4:"Grid Averaged Methane Emissions from Wetlands",wetlandch4cons:"Grid Averaged Methane Consumption (Methanotrophy) from Wetlands",wetlandch4prod:"Grid Averaged Methane Production (Methanogenesis) from Wetlands",wetlandfrac:"Wetland Percentage Cover",wetnh3:"Wet Deposition Rate of NH3",wetnh4:"Wet Deposition Rate of NH4",wetnoy:"Wet Deposition Rate of NOy Including Aerosol Nitrate",wetoa:"Wet Deposition Rate of Dry Aerosol Total Organic Matter",wetpoa:"Wet Deposition Rate of Dry Aerosol Primary Organic Matter",wetso2:"Wet Deposition Rate of SO2",wetso4:"Wet Deposition Rate of SO4",wetsoa:"Wet Deposition Rate of Dry Aerosol Secondary Organic Matter",wetss:"Wet Deposition Rate of Sea-Salt Aerosol",wfcorr:"Water Flux Correction",wfo:"Water Flux into Sea Water",wfonocorr:"Water Flux into Sea Water Without Flux Correction",wilt:"Wilting Point",wmo:"Upward Ocean Mass Transport",wmosq:"Square of Upward Ocean Mass Transport",wo:"Sea Water Vertical Velocity",ws:"Eastward Wind",ws20000:"",wsgmax100m:"Maximum Wind Speed of Gust at 100m",wsgmax10m:"Maximum Wind Speed of Gust at 10m",wsgsmax:"Maximum Near-Surface Wind Speed of Gust",wtd:"Water Table Depth",wtem:"Transformed Eulerian Mean Upward Wind",xgwdparam:"Eastward Gravity Wave Drag",xvelbase:"X-Component of Land Ice Basal Velocity",xvelmean:"X-Component of Land Ice Vertical Mean Velocity",xvelsurf:"X-Component of Land Ice Surface Velocity",ygwdparam:"Northward Gravity Wave Drag",yvelbase:"Y-Component of Land Ice Basal Velocity",yvelmean:"Y-Component of Land Ice Vertical Mean Velocity",yvelsurf:"Y-Component of Land Ice Surface Velocity",z:"Geopotential",zfull:"Altitude of Model Full-Levels",zfullo:"Depth Below Geoid of Ocean Layer",zg:"Geopotential Height",zg10:"Geopotential Height at 10 hPa",zg100:"Geopotential Height at 100 hPa",zg1000:"Geopotential Height at 1000 hPa",zg200:"Geopotential Height at 250 hPa",zg20000:"",zg250:"Geopotential Height at 250 hPa",zg27:"Geopotential Height",zg300:"Geopotential Height at 300 hPa",zg400:"Geopotential Height at 400 hPa",zg500:"Geopotential Height at 500 hPa",zg600:"Geopotential Height at 600 hPa",zg700:"Geopotential Height at 700 hPa",zg7h:"Geopotential Height",zg800:"Geopotential Height at 800 hPa",zg850:"Geopotential Height at 850 hPa",zg900:"Geopotential Height at 900 hPa",zg925:"Geopotential Height at 925 hPa",zg950:"Geopotential Height at 950 hPa",zg975:"Geopotential Height at 975 hPa",zhalf:"Altitude of Model Half-Levels",zhalfo:"Depth Below Geoid of Interfaces Between Ocean Layers",zmeso:"Mole Concentration of Mesozooplankton Expressed as Carbon in Sea Water",zmesoos:"Surface Mole Concentration of Mesozooplankton Expressed as Carbon in Sea Water",zmicro:"Mole Concentration of Microzooplankton Expressed as Carbon in Sea Water",zmicroos:"Surface Mole Concentration of Microzooplankton expressed as Carbon in sea water",zmisc:"Mole Concentration of Other Zooplankton Expressed as Carbon in Sea Water",zmiscos:"Surface Mole Concentration of Other Zooplankton Expressed as Carbon in Sea Water",zmla:"Height of Boundary Layer",zmlwaero:"Zonal Mean Longwave Heating Rate Due to Volcanic Aerosols",zmswaero:"Zonal Mean Shortwave Heating Rate Due to Volcanic Aerosols",zmtnt:"Zonal Mean Diabatic Heating Rates",zo2min:"Depth of Oxygen Minimum Concentration",zooc:"Zooplankton Carbon Concentration",zooc3d:"Zooplankton Carbon Concentration",zoocmisc:"Other Zooplankton Carbon Concentration at Surface",zoocos:"Surface Zooplankton Carbon Concentration",zos:"Sea Surface Height Above Geoid",zosga:"Global Average Sea Level Change",zosnobs:"Sea Surface Height Above Geoid Number of Observations",zossga:"Global Average Steric Sea Level Change",zossq:"Square of Sea Surface Height Above Geoid",zosstderr:"Sea Surface Height Above Geoid Standard Error",zostoga:"Global Average Thermosteric Sea Level Change",zsatarag:"Aragonite Saturation Depth",zsatcalc:"Calcite Saturation Depth",ztp:"Tropopause Altitude above Geoid",zvelbase:"Upward Component of Land-Ice Basal Velocity",zvelsurf:"Upward Component of Land-Ice Surface Velocity"})})});var Ai={};zr(Ai,{AggregationConfigElement:()=>Ha,DataInspectorElement:()=>Na,NcDumpDialogState:()=>it,ZarrLoadingStepsElement:()=>Ua,ZarrPoller:()=>qo,buildXarrayRepr:()=>mi,detectZarrStore:()=>Rl,injectXarrayCss:()=>gi,loadZarrMetadataHtml:()=>Jl,openDatasetMeta:()=>fi});function Uo(e){return`${El}#${e}`}function Dl(e){var t;let a=navigator;return(t=a.clipboard)!=null&&t.writeText?a.clipboard.writeText(e):new Promise((o,r)=>{try{let n=document.createElement("textarea");n.value=e,n.style.position="fixed",n.style.opacity="0",document.body.appendChild(n),n.focus(),n.select(),document.execCommand("copy"),document.body.removeChild(n),o()}catch(n){r(n instanceof Error?n:new Error(String(n)))}})}function Se(e,t=!1){return`<svg viewBox="0 0 24 24" width="1em" height="1em" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" focusable="false" style="vertical-align:-0.14em;flex-shrink:0${t?";margin-right:.5em":""}">${e}</svg>`}function Ol(){if($n||typeof document>"u")return;$n=!0;try{if(typeof CSSStyleSheet=="function"&&Array.isArray(document.adoptedStyleSheets)){let t=new CSSStyleSheet;t.replaceSync(_n),document.adoptedStyleSheets=[...document.adoptedStyleSheets,t];return}}catch{}let e=document.createElement("style");e.setAttribute("data-data-inspector","1"),e.textContent=_n,document.head.appendChild(e)}function Fa(e){return e===0?3:e===4?2:e===3?1:0}function Ll(){if(!document.getElementById("zarr-loading-keyframes")){let e=document.createElement("style");e.id="zarr-loading-keyframes",e.textContent=Il,document.head.appendChild(e)}}function oi(e,t,a){let o=e<t,r=e===t,n=o?`<svg width="10" height="8" viewBox="0 0 10 8" fill="none">
        <path d="M1 4L3.8 7L9 1" stroke="#fff" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
       </svg>`:r?`<svg width="16" height="16" viewBox="0 0 16 16" class="zarr-spin" style="position:absolute;">
        <circle cx="8" cy="8" r="5.5" fill="none" stroke="${Ce.active}" stroke-width="2"
          stroke-dasharray="18 17" stroke-linecap="round"/>
       </svg>`:`<div style="width:5px;height:5px;border-radius:50%;background:${Ce.pending};"></div>`,i=r?`<div style="position:absolute;width:30px;height:30px;border-radius:50%;
         border:2px solid ${Ce.active};animation:zarrPulseRing 2s ease-out infinite;pointer-events:none;"></div>`:"";return`
    ${e>0?`
    <div style="flex:1;height:2px;background:${o?`linear-gradient(90deg,${Ce.done},${Ce.active})`:Ce.track};
      transition:background 0.5s ease;position:relative;overflow:hidden;">
      ${r?`<div style="position:absolute;top:0;left:0;height:100%;
        background:linear-gradient(90deg,${Ce.done},${Ce.active});
        animation:zarrTrackFill 0.6s ease forwards;"></div>`:""}
    </div>`:""}
    <div style="position:relative;display:flex;flex-direction:column;align-items:center;">
      ${i}
      <div style="width:22px;height:22px;border-radius:50%;
        background:${o?Ce.done:r?"#fff":"#f9fafb"};
        border:2px solid ${o?Ce.done:r?Ce.active:Ce.pending};
        display:flex;align-items:center;justify-content:center;
        position:relative;z-index:1;
        box-shadow:${r?"0 0 0 4px rgba(20,184,166,0.12)":"none"};
        transition:all 0.4s ease;">
        ${n}
      </div>
      <span style="position:absolute;top:28px;font-size:10.5px;
        font-weight:${r?600:400};
        color:${o?Ce.done:r?Ce.textOn:Ce.textOff};
        white-space:nowrap;letter-spacing:0.03em;text-transform:uppercase;
        transition:color 0.4s ease;">
        ${a}
      </span>
    </div>`}function jo(){if(typeof document>"u")return{};let e=document.cookie.split(";").find(t=>t.trim().startsWith(Bl));if(!e)return{};try{let t=e.substring(e.indexOf("=")+1).trim();return t.startsWith('"')&&t.endsWith('"')&&(t=t.slice(1,-1)),t?{Authorization:`Bearer ${t}`}:{}}catch{return{}}}function pi(e){if(typeof e!="string")return e;let t=e.trim();for(;/^https?%(25)*3a/i.test(t);)try{let a=decodeURIComponent(t);if(a===t)break;t=a}catch{break}return t}function Pl(e){return`/api/freva-nextgen/data-portal/zarr-utils/status?url=${e}&timeout=1`}async function Rl(e,t={}){if(!e)return{...ri};let a=pi(e).replace(/\/$/,""),o={credentials:"same-origin",headers:(t.getAuthHeaders??jo)(),signal:AbortSignal.timeout(t.timeoutMs??5e3)};try{if((await fetch(`${a}/.zmetadata`,o)).ok)return{isZarr:!0,version:2,consolidated:!0}}catch{}try{let r=await fetch(`${a}/zarr.json`,o);if(r.ok){let n=await r.json(),i=n.zarr_format;if(i===2||i===3){let l=i===3?"consolidated_metadata"in n:!1;return{isZarr:!0,version:i,consolidated:l}}}}catch{}return{...ri}}function Fl(e){return{f2:"float16",f4:"float32",f8:"float64",i1:"int8",i2:"int16",i4:"int32",i8:"int64",u1:"uint8",u2:"uint16",u4:"uint32",u8:"uint64",b1:"bool"}[e.slice(1)]??e}function zl(e){let t={...e};return delete t._ARRAY_DIMENSIONS,t}function ui(e,t,a){let o={},r={};for(let[p,u]of Object.entries(e)){let{shape:s,chunks:g,dtype:m,dims:A,attrs:v}=a(u);A.forEach((M,B)=>{M in o||(o[M]=s[B]??0)});let b=v.units,k=A.length===1&&A[0]===p&&(String(b??"").includes("since")||p==="time");r[p]={shape:s,chunks:g,dtype:m,dims:A,attrs:v,_isTimeCoord:k}}let n=new Set,i=p=>String(p??"").split(/[\s,]+/).filter(Boolean);for(let[p,u]of Object.entries(r))u.dims.length===1&&u.dims[0]===p&&n.add(p);i(t.coordinates).forEach(p=>n.add(p));for(let p of Object.values(r))i(p.attrs.coordinates).forEach(u=>n.add(u));let l={},d={};for(let[p,u]of Object.entries(r))(n.has(p)?l:d)[p]=u;return{dims:o,coords:l,data_vars:d,attrs:t}}function Ut(e){return Object.keys(e.coords).length+Object.keys(e.data_vars).length>0}function Qo(e,t){let a=e[`${t}.zattrs`]??{},o={};for(let[r,n]of Object.entries(e)){if(!r.startsWith(t))continue;let i=r.slice(t.length);if(i.endsWith("/.zarray")){let l=i.slice(0,-8);if(l.includes("/"))continue;o[l]??(o[l]={}),o[l].zarray=n}else if(i.endsWith("/.zattrs")){let l=i.slice(0,-8);if(!l||l.includes("/"))continue;o[l]??(o[l]={}),o[l].zattrs=n}}for(let r of Object.keys(o))o[r].zarray||delete o[r];return ui(o,a,r=>{var n,i,l,d;let p=r.zattrs??{};return{shape:((n=r.zarray)==null?void 0:n.shape)??[],chunks:((i=r.zarray)==null?void 0:i.chunks)??((l=r.zarray)==null?void 0:l.shape)??[],dtype:Fl(((d=r.zarray)==null?void 0:d.dtype)??"|u1"),dims:p._ARRAY_DIMENSIONS??[],attrs:zl(p)}})}function Nl(e){let t=e.metadata??{},a=new Set;for(let n of Object.keys(t))n===".zgroup"||n===".zattrs"||n.endsWith("/.zgroup")&&a.add(n.slice(0,-8));if(a.size===0){let n=Qo(t,"");if(!Ut(n))throw new Error("No arrays found in .zmetadata");return{groups:null,...n}}let o={},r=Qo(t,"");Ut(r)&&(o["/"]=r);for(let n of[...a].sort()){let i=Qo(t,`${n}/`);Ut(i)&&(o[n]=i)}if(!Object.keys(o).length)throw new Error("No arrays found in .zmetadata");return{groups:o}}function Wo(e,t){let a=(e[t||""]??{}).attributes??{},o={};for(let[r,n]of Object.entries(e))if(n.node_type==="array")if(t){if(!r.startsWith(`${t}/`))continue;let i=r.slice(t.length+1);if(i.includes("/"))continue;o[i]={zarray:n}}else{if(!r||r.includes("/"))continue;o[r]={zarray:n}}return ui(o,a,r=>{var n,i;let l=r.zarray;return{shape:l.shape??[],chunks:((i=(n=l.chunk_grid)==null?void 0:n.configuration)==null?void 0:i.chunk_shape)??l.shape??[],dtype:l.data_type??"float32",dims:l.dimension_names??[],attrs:l.attributes??{}}})}function Hl(e){var t;let a=((t=e.consolidated_metadata)==null?void 0:t.metadata)??{};if(!Object.keys(a).length)throw new Error("zarr.json has no consolidated_metadata");let o=new Set;for(let[i,l]of Object.entries(a))!i||l.node_type!=="group"||o.add(i);if(o.size===0){let i=Wo(a,"");if(!Ut(i))throw new Error("No arrays found in zarr.json");return{groups:null,...i}}let r={},n=Wo(a,"");Ut(n)&&(r["/"]=n);for(let i of[...o].sort()){let l=Wo(a,i);Ut(l)&&(r[i]=l)}if(!Object.keys(r).length)throw new Error("No arrays found in zarr.json");return{groups:r}}async function fi(e,t={}){let a=pi(e).replace(/\/$/,""),o={credentials:"same-origin",headers:(t.getAuthHeaders??jo)()},r=null,n=0;try{let i=await fetch(`${a}/.zmetadata`,o);i.ok&&(r=await i.json(),n=2)}catch{}if(!r)try{let i=await fetch(`${a}/zarr.json`,o);i.ok&&(r=await i.json(),n=3)}catch{}if(!r)throw new Error("Could not read zarr metadata (.zmetadata or zarr.json)");return n===2?Nl(r):Hl(r)}function $e(e){return String(e).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;")}function Vo(){return`xr${++Ul}`}function ni(e){return`<svg class="icon xr-${e}"><use xlink:href="#${e}"></use></svg>`}function Gl(e,t){return Object.keys(e).length?`<ul class='xr-dim-list'>${Object.entries(e).map(([a,o])=>`<li><span${t.has(a)?" class='xr-has-index'":""}>${$e(a)}</span>: ${o}</li>`).join("")}</ul>`:""}function hi(e){let t=Object.entries(e);return t.length?`<dl class='xr-attrs'>${t.map(([a,o])=>`<dt><span>${$e(a)} :</span></dt><dd>${$e(String(o))}</dd>`).join("")}</dl>`:"<em>No attributes</em>"}function Ql(e){let t=e.shape.reduce((a,o)=>a*o,1);return t===0?"[]":`${e.dtype} (${e.shape.join(" \xD7 ")} = ${t.toLocaleString()})`}function ii(e){return e<1024?e+" B":e<1024**2?(e/1024).toFixed(2)+" KiB":e<1024**3?(e/1024**2).toFixed(2)+" MiB":(e/1024**3).toFixed(2)+" GiB"}function Wl(e){if(!e.length)return"";let t=Math.min(e.length,3),a=e.slice(-t),o=M=>Math.max(20,Math.min(110,20+Math.log10(Math.max(1,M))*30)),r=M=>M.toLocaleString(),n="font-size:11px;fill:var(--xr-font-color2);font-family:monospace";if(t<3){let M=o(a[t-1]),B=t===2?o(a[0]):12,D=t===2?42:2,P=16,U=D+M+4,q=B+P;return`<svg width="${Math.ceil(U)}" height="${Math.ceil(q)}"
        viewBox="0 0 ${Math.ceil(U)} ${Math.ceil(q)}"
        style="overflow:visible;display:block;flex-shrink:0">
      <rect x="${D}" y="0" width="${M}" height="${B}"
            style="fill:var(--xr-chunk-face);stroke:var(--xr-chunk-edge);stroke-width:0.8"/>
      ${t===2?`<text x="${D-5}" y="${B/2+4}"
            text-anchor="end" style="${n}">${r(a[0])}</text>`:""}
      <text x="${D+M/2}" y="${B+P-3}"
            text-anchor="middle" style="${n}">${r(a[t-1])}</text>
    </svg>`}let i=o(a[2]),l=o(a[1]),d=o(a[0]),p=.5,u=d*p,s=d*p*.45,g=16,m=16,A=i+u+52,v=g+l+s+m,b=2,k=g+l+s;return`<svg width="${Math.ceil(A)}" height="${Math.ceil(v)}"
      viewBox="0 0 ${Math.ceil(A)} ${Math.ceil(v)}"
      style="overflow:visible;display:block;flex-shrink:0">
    <polygon points="${b},${k} ${b+i},${k} ${b+i},${k-l} ${b},${k-l}"
             style="fill:var(--xr-chunk-face);stroke:var(--xr-chunk-edge);stroke-width:0.8"/>
    <polygon points="${b},${k-l} ${b+i},${k-l} ${b+i+u},${k-l-s} ${b+u},${k-l-s}"
             style="fill:var(--xr-chunk-top);stroke:var(--xr-chunk-edge);stroke-width:0.8"/>
    <polygon points="${b+i},${k} ${b+i+u},${k-s} ${b+i+u},${k-l-s} ${b+i},${k-l}"
             style="fill:var(--xr-chunk-side);stroke:var(--xr-chunk-edge);stroke-width:0.8"/>
    <text x="${b+i/2}" y="${k+m-3}"
          text-anchor="middle" style="${n}">${r(a[2])}</text>
    <text x="${b+i/2+u/2}" y="${g-3}"
          text-anchor="middle" style="${n}">${r(a[1])}</text>
    <text x="${b+i+u+5}" y="${k-l/2-s/2+4}"
          text-anchor="start" style="${n}">${r(a[0])}</text>
  </svg>`}function ql(e){let{shape:t,chunks:a,dtype:o}=e,r={int8:1,uint8:1,bool:1,int16:2,uint16:2,int32:4,uint32:4,float32:4,int64:8,uint64:8,float64:8}[o]??4,n=t.reduce((g,m)=>g*m,1)*r,i=null,l=null;a&&a.length===t.length&&(i=a.reduce((g,m)=>g*m,1)*r,l=t.reduce((g,m,A)=>g*Math.ceil(m/a[A]),1));let d='style="color:var(--xr-font-color3);padding:2px 16px 2px 0;white-space:nowrap;vertical-align:top"',p='style="padding:2px 16px 2px 0;white-space:nowrap;vertical-align:top"',u='style="padding:2px 0;white-space:nowrap;color:var(--xr-font-color2);vertical-align:top"',s=(g,m,A="")=>`<tr><td ${d}>${g}</td><td ${p}>${m}</td><td ${u}>${A}</td></tr>`;return`<table style="border-collapse:collapse;padding:6px 0 12px"><tr>
    <td style="vertical-align:top;padding:0">
      <table style="font-size:12px;font-family:monospace;border-collapse:collapse;line-height:1.75">
        <thead><tr>
          <th style="font-weight:400;padding:0 16px 5px 0;text-align:left"></th>
          <th style="font-weight:600;padding:0 16px 5px 0;text-align:left">Array</th>
          <th style="font-weight:600;padding:0 0 5px 0;text-align:left">Chunk</th>
        </tr></thead><tbody>
          ${s("Bytes",ii(n),i!==null?ii(i):"\u2014")}
          ${s("Shape","("+t.join(", ")+")",a?"("+a.join(", ")+")":"\u2014")}
          ${l!==null?s("Chunks",l.toLocaleString()+" chunks"):""}
          ${s("dtype",$e(o))}
          ${s("dims","("+e.dims.join(", ")+")")}
        </tbody>
      </table>
    </td>
    <td style="vertical-align:middle;padding:0 0 0 32px">${Wl(t)}</td>
  </tr></table>`}function Vl(e,t,a){let o=Vo(),r=Vo(),n=Object.keys(t.attrs).length>0;return`
    <div class='xr-var-name'><span${a?" class='xr-has-index'":""}>${$e(e)}</span></div>
    <div class='xr-var-dims'>(${t.dims.map($e).join(", ")})</div>
    <div class='xr-var-dtype'>${$e(t.dtype)}</div>
    <div class='xr-var-preview xr-preview'>${$e(Ql(t))}</div>
    <input id='${o}' class='xr-var-attrs-in' type='checkbox'${n?"":" disabled"}>
    <label for='${o}' title='Show/Hide attributes'>${ni("icon-file-text2")}</label>
    <input id='${r}' class='xr-var-data-in' type='checkbox'>
    <label for='${r}' title='Show/Hide data repr'>${ni("icon-database")}</label>
    <div class='xr-var-attrs'>${hi(t.attrs)}</div>
    <div class='xr-var-data'>${ql(t)}</div>
  `}function si(e,t){return`<ul class='xr-var-list'>${Object.entries(e).map(([a,o])=>`<li class='xr-var-item'>${Vl(a,o,t.has(a))}</li>`).join("")}</ul>`}function za(e,t,a,o,r,n){let i=Vo(),l=(o??0)>0,d=o!==null?` <span>(${o})</span>`:"",p=r&&l?"":" disabled";return`
    <input id='${i}' class='xr-section-summary-in' type='checkbox'${p}${n||!l?"":" checked"} />
    <label for='${i}' class='xr-section-summary'${p===""?" title='Expand/collapse section'":""}>${e}${d}</label>
    <div class='xr-section-inline-details'>${t}</div>
    ${a?`<div class='xr-section-details'>${a}</div>`:""}
  `}function ci(e){let t=new Set(Object.keys(e.coords)),a=[];return a.push(za("Dimensions:",Gl(e.dims,t),"",Object.keys(e.dims).length,!1,!0)),Object.keys(e.coords).length&&a.push(za("Coordinates:","",si(e.coords,t),Object.keys(e.coords).length,!0,!1)),a.push(za("Data variables:","",si(e.data_vars,new Set),Object.keys(e.data_vars).length,!0,!1)),Object.keys(e.attrs).length&&a.push(za("Attributes:","",hi(e.attrs),Object.keys(e.attrs).length,!0,!0)),`<div class='xr-root'>
    <div class='xr-wrap'>
      <div class='xr-header'><div class='xr-obj-type'>xarray.Dataset</div></div>
      <ul class='xr-sections'>${a.map(o=>`<li class='xr-section-item'>${o}</li>`).join("")}</ul>
    </div>
  </div>`}function mi(e){if(!e.groups)return`${li}${ci(e)}`;let t=Object.entries(e.groups).map(([a,o])=>`
    <details open style="margin-bottom:10px;border:1px solid var(--xr-border-color);border-radius:4px;overflow:hidden">
      <summary style="padding:8px 12px;font-weight:600;cursor:pointer;background:var(--xr-background-color-row-odd);list-style:none;display:flex;align-items:center;gap:8px">
        <span style="font-size:11px;color:var(--xr-font-color2)">\u25B6</span>
        <span>Group: ${$e(a)}</span>
      </summary>
      <div style="padding:0 12px 8px">${ci(o)}</div>
    </details>
  `).join("");return`${li}<div style="font-family:monospace">${t}</div>`}function gi(e={}){if(di||typeof document>"u")return;di=!0;let t=typeof window<"u"?window.MAIN_COLOR:void 0,a=(e.mainColor??t??"#9b7a52").replace(/^#/,""),o=parseInt(a.slice(0,2),16),r=parseInt(a.slice(2,4),16),n=parseInt(a.slice(4,6),16),i=s=>`rgb(${Math.min(255,o*s|0)},${Math.min(255,r*s|0)},${Math.min(255,n*s|0)})`,l=(s,g)=>`rgba(${Math.min(255,o*s|0)},${Math.min(255,r*s|0)},${Math.min(255,n*s|0)},${g})`,d=`rgb(${255-o},${255-r},${255-n})`,p=`
    :root{--xr-chunk-face:${i(.85)};--xr-chunk-top:${i(1.25)};--xr-chunk-side:${i(.55)};--xr-chunk-edge:${d}}
    html[data-theme="dark"],body[data-theme="dark"],body.vscode-dark{
      --xr-chunk-face:${l(.85,.65)};--xr-chunk-top:${l(1.25,.65)};--xr-chunk-side:${l(.55,.65)};--xr-chunk-edge:${d}
    }
  `+jl;try{if(typeof CSSStyleSheet=="function"&&Array.isArray(document.adoptedStyleSheets)){let s=new CSSStyleSheet;s.replaceSync(p),document.adoptedStyleSheets=[...document.adoptedStyleSheets,s];return}}catch{}let u=document.createElement("style");u.setAttribute("data-xarray-repr","1"),u.textContent=p,document.head.appendChild(u)}async function Jl(e,t={}){t.injectCss!==!1&&gi({mainColor:t.mainColor});let a=await fi(e,{getAuthHeaders:t.getAuthHeaders});return mi(a)}var it,Tl,El,ve,_n,$n,Na,ei,ti,Go,Ha,Il,ai,Ce,Pa,Ra,Ua,Bl,qo,ri,Ul,li,jl,di,bi=Fr(()=>{it={ERROR:"error",READY:"ready",LOADING:"loading"},Tl={aggregate:"auto",join:null,compat:null,data_vars:null,coords:null,dim:"",group_by:"",reload:!1,access_pattern:"map",chunk_size:16,map_primary_chunksize:1,timeout:120},El="https://gridlook.pages.dev/";ve={info:'<circle cx="12" cy="12" r="9"/><path d="M12 11.5v5"/><circle cx="12" cy="7.8" r="0.6" fill="currentColor" stroke="none"/>',layers:'<path d="M12 3 3 8l9 5 9-5-9-5Z"/><path d="m3 13 9 5 9-5"/>',load:'<path d="M21 12a9 9 0 1 1-2.64-6.36"/><path d="M21 4v4h-4"/>',chevronDown:'<path d="m6 9 6 6 6-6"/>',ban:'<circle cx="12" cy="12" r="9"/><path d="m5.6 5.6 12.8 12.8"/>',link:'<path d="M9.5 13.5a4 4 0 0 0 5.7 0l2.8-2.8a4 4 0 1 0-5.7-5.7l-1 1"/><path d="M14.5 10.5a4 4 0 0 0-5.7 0l-2.8 2.8a4 4 0 1 0 5.7 5.7l1-1"/>',copy:'<rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V5a2 2 0 0 1 2-2h10"/>',check:'<path d="M20 6 9 17l-5-5"/>',database:'<ellipse cx="12" cy="5.5" rx="8" ry="3"/><path d="M4 5.5v13c0 1.66 3.58 3 8 3s8-1.34 8-3v-13"/><path d="M4 12c0 1.66 3.58 3 8 3s8-1.34 8-3"/>',cube:'<path d="M12 3 3 7.5v9L12 21l9-4.5v-9L12 3Z"/><path d="m3 7.5 9 4.5 9-4.5"/><path d="M12 12v9"/>',external:'<path d="M14 4h6v6"/><path d="M20 4 11 13"/><path d="M19 14v4a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V7a2 2 0 0 1 2-2h4"/>',refresh:'<path d="M21 12a9 9 0 1 1-2.64-6.36"/><path d="M21 4v4h-4"/>',compress:'<path d="M9 5v4H5"/><path d="m4 4 5 5"/><path d="M15 5v4h4"/><path d="m20 4-5 5"/><path d="M9 19v-4H5"/><path d="m4 20 5-5"/><path d="M15 19v-4h4"/><path d="m20 20-5-5"/>',alert:'<circle cx="12" cy="12" r="9"/><path d="M12 7.5v5.5"/><circle cx="12" cy="16.3" r="0.6" fill="currentColor" stroke="none"/>'},_n=`
data-inspector{
  --_di-bg:var(--di-bg,#fff);
  --_di-fg:var(--di-fg,#1f2937);
  --_di-muted:var(--di-muted,#6b7280);
  --_di-border:var(--di-border,#e5e7eb);
  --_di-surface:var(--di-surface,#f3f4f6);
  --_di-accent:var(--di-accent,#3b82f6);
}
@media (prefers-color-scheme:dark){
  data-inspector{
    --_di-bg:var(--di-bg,#1e293b);
    --_di-fg:var(--di-fg,#e5e7eb);
    --_di-muted:var(--di-muted,#94a3b8);
    --_di-border:var(--di-border,#475569);
    --_di-surface:var(--di-surface,#334155);
    --_di-accent:var(--di-accent,#3b82f6);
  }
}
.di-backdrop{position:fixed;inset:0;z-index:1050;display:flex;align-items:center;justify-content:center;padding:12px;background:rgba(15,23,42,.55);}
.di-modal{display:flex;flex-direction:column;width:min(1100px,96vw);max-height:95vh;overflow:hidden;background:var(--_di-bg);color:var(--_di-fg);border-radius:12px;box-shadow:0 20px 50px rgba(0,0,0,.25);font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;font-size:14px;line-height:1.5;}
.di-header{flex-shrink:0;border-bottom:1px solid var(--_di-border);padding:16px 16px 12px;}
.di-header-row{display:flex;justify-content:space-between;align-items:flex-start;gap:8px;}
.di-header-main{flex:1;min-width:0;}
.di-title{margin:0 0 12px;font-size:clamp(16px,4vw,20px);font-weight:600;display:flex;align-items:center;gap:8px;}
.di-title-ico{color:var(--_di-accent);display:inline-flex;align-items:center;font-size:16px;}
.di-close{flex-shrink:0;width:32px;height:32px;font-size:28px;line-height:1;background:transparent;border:none;border-radius:6px;color:var(--_di-muted);cursor:pointer;}
.di-close:hover{background:var(--_di-surface);color:var(--_di-fg);}
.di-muted{color:var(--_di-muted);}
.di-center{text-align:center;padding:24px 12px;}
.di-file-list{margin-bottom:12px;padding:10px;background:var(--_di-surface);border-radius:6px;max-height:120px;overflow-y:auto;}
.di-file-list-label{font-size:12px;color:var(--_di-muted);margin-bottom:6px;font-weight:500;}
.di-file-list ul{font-size:11px;margin:0;padding-left:20px;}
.di-file-list li{color:var(--_di-fg);word-break:break-all;}
.di-pathbar{margin-bottom:8px;}
.di-pathbar-label{display:block;font-size:12px;color:var(--_di-muted);margin-bottom:4px;font-weight:500;}
.di-pathbar-row{display:flex;gap:6px;flex-wrap:wrap;}
.di-input{flex:1 1 200px;min-width:0;font-size:13px;padding:6px 10px;border:1px solid var(--_di-border);border-radius:6px;color:var(--_di-fg);background:var(--_di-bg);}
.di-input:focus{outline:2px solid var(--_di-accent);outline-offset:0;border-color:var(--_di-accent);}
.di-dropdown-wrap{position:relative;flex-shrink:0;display:inline-flex;}
.di-btn{display:inline-flex;align-items:center;justify-content:center;font-size:13px;font-weight:500;border-radius:6px;border:1px solid transparent;padding:6px 12px;cursor:pointer;white-space:nowrap;background:var(--_di-bg);color:var(--_di-fg);}
.di-btn:disabled{opacity:.5;cursor:not-allowed;}
.di-btn-primary{background:var(--_di-accent);border-color:var(--_di-accent);color:#fff;}
.di-btn-primary:hover:not(:disabled){filter:brightness(.93);}
.di-btn-secondary{background:var(--_di-surface);border-color:var(--_di-surface);color:var(--_di-fg);}
.di-btn-outline{background:var(--_di-bg);border-color:var(--_di-border);color:var(--_di-fg);}
.di-btn-split{padding:6px 9px;border-top-left-radius:0;border-bottom-left-radius:0;border-left-color:rgba(255,255,255,.4);}
.di-btn-group{display:inline-flex;}
.di-btn-group>.di-btn:first-child{border-top-right-radius:0;border-bottom-right-radius:0;}
.di-menu{position:absolute;top:100%;right:0;z-index:1060;min-width:210px;margin-top:2px;padding:6px 0;background:var(--_di-bg);border:1px solid var(--_di-border);border-radius:8px;box-shadow:0 10px 25px rgba(0,0,0,.12);list-style:none;}
.di-menu-item{display:flex;align-items:center;width:100%;padding:8px 14px;font-size:13px;background:none;border:none;color:var(--_di-fg);cursor:pointer;text-align:left;}
.di-menu-item:hover{background:var(--_di-surface);}
.di-zarr-row{display:flex;align-items:center;flex-wrap:wrap;gap:6px;padding:8px 10px;background:var(--_di-surface);border-radius:6px;font-size:11px;margin-bottom:8px;}
.di-zarr-inner{display:flex;align-items:center;gap:6px;flex:1 1 100%;min-width:0;}
.di-code{flex:1;min-width:0;background:var(--_di-bg);padding:4px 8px;border-radius:4px;font-size:10px;color:var(--_di-fg);border:1px solid var(--_di-border);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;}
.di-tabs{display:flex;gap:2px;border-bottom:2px solid var(--_di-border);}
.di-tab{padding:12px 16px;font-size:14px;font-weight:500;background:transparent;border:none;border-bottom:3px solid transparent;color:var(--_di-muted);cursor:pointer;transition:all .15s ease;display:inline-flex;align-items:center;}
.di-tab:disabled{opacity:.5;cursor:not-allowed;}
.di-tab-active{font-weight:600;background:var(--_di-surface);border-bottom-color:var(--_di-accent);color:var(--_di-fg);}
.di-body{flex:1;overflow-y:auto;overflow-x:hidden;padding:16px 12px;max-height:calc(95vh - 200px);}
.di-error{border-radius:8px;background:#fee2e2;border:1px solid #fecaca;color:#991b1b;font-size:13px;padding:12px;margin-bottom:16px;}
.di-error-row{display:flex;align-items:flex-start;gap:10px;}
.di-error-ico{font-size:18px;margin-top:1px;flex-shrink:0;}
.di-error-body{flex:1;min-width:0;}
.di-error-title{display:block;margin-bottom:6px;}
.di-error-msg{word-wrap:break-word;overflow-wrap:anywhere;white-space:pre-wrap;}
.di-btn-danger{background:#dc2626;border-color:#dc2626;color:#fff;font-size:12px;padding:6px 12px;margin-top:8px;}
.di-metadata{display:flex;justify-content:flex-start;width:100%;overflow-x:auto;}
.di-metadata>*{width:100%;min-width:0;}
.di-metadata dd,.di-metadata .xr-attrs td,.di-metadata .xr-var-attrs td{overflow-wrap:anywhere;word-break:break-word;}
.di-gridlook-bar{display:flex;align-items:center;flex-wrap:wrap;gap:8px;padding:12px;background:var(--_di-surface);border-radius:8px;font-size:12px;margin-bottom:16px;border:1px solid var(--_di-border);}
.di-gridlook-inner{display:flex;align-items:center;gap:6px;flex:1 1 100%;min-width:0;}
.di-gridlook-ico{color:var(--_di-accent);flex-shrink:0;display:inline-flex;}
.di-gridlook-label{color:var(--_di-accent);font-weight:600;flex-shrink:0;}
.di-gridlook-code{flex:1;min-width:0;background:var(--_di-bg);padding:6px 10px;border-radius:4px;font-size:11px;color:var(--_di-fg);border:1px solid var(--_di-border);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;}
.di-gridlook-btn{background:var(--_di-bg);border:1px solid var(--_di-accent);color:var(--_di-accent);font-size:12px;padding:6px 12px;flex-shrink:0;}
.di-gridlook-btn-primary{background:var(--_di-accent);border-color:var(--_di-accent);color:#fff;}
.di-gridlook-frame{width:100%;height:calc(95vh - 280px);min-height:500px;background:var(--_di-surface);border-radius:8px;overflow:hidden;border:1px solid var(--_di-border);}
.di-gridlook-frame iframe{width:100%;height:100%;border:none;display:block;}
.di-agg-form{padding:16px;overflow-y:auto;}
.di-agg-form h5{margin:0 0 12px;font-size:1.05rem;font-weight:600;}
.di-agg-actions{display:flex;justify-content:flex-end;gap:8px;margin-top:12px;}
.di-empty-ico{font-size:40px;color:var(--_di-border);margin-bottom:12px;display:block;}
.di-empty-text{color:var(--_di-muted);font-size:13px;padding:0 12px;}
[hidden]{display:none!important;}
`,$n=!1;Na=class extends HTMLElement{constructor(){super(...arguments),this._pathInput="",this._copied=!1,this._gridlookCopied=!1,this._activeTab="metadata",this._dropdownOpen=!1,this._aggregationConfig={...Tl},this._output=null,this._built=!1,this._builtMode=!1,this._domOutput=void 0,this._domIframeUrl=void 0,this._domFileKey=void 0,this._copyTimer=null,this._gridlookTimer=null,this._restoreFocusTo=null,this._onOutsideClick=t=>{let a=this.querySelector("#nc-dropdown-wrap");a&&!a.contains(t.target)&&this._dropdownOpen&&(this._dropdownOpen=!1,this._syncDropdown())}}static get observedAttributes(){return["open","file","status","error","zarr-url","zarr-status-code","is-aggregation"]}get open(){return this.hasAttribute("open")}set open(t){t?this.setAttribute("open",""):this.removeAttribute("open")}get file(){let t=this.getAttribute("file");if(!t)return null;try{return JSON.parse(t)}catch{return t}}set file(t){t===null?this.removeAttribute("file"):this.setAttribute("file",Array.isArray(t)?JSON.stringify(t):t)}get status(){return this.getAttribute("status")??it.READY}set status(t){this.setAttribute("status",t)}get output(){return this._output}set output(t){this._output=t,this.isConnected&&this._render()}get error(){return this.getAttribute("error")}set error(t){t===null?this.removeAttribute("error"):this.setAttribute("error",t)}get zarrUrl(){return this.getAttribute("zarr-url")}set zarrUrl(t){t===null?this.removeAttribute("zarr-url"):this.setAttribute("zarr-url",t)}get zarrStatusCode(){let t=this.getAttribute("zarr-status-code");return t===null?null:parseInt(t,10)}set zarrStatusCode(t){t===null?this.removeAttribute("zarr-status-code"):this.setAttribute("zarr-status-code",String(t))}get isAggregation(){return this.hasAttribute("is-aggregation")}set isAggregation(t){t?this.setAttribute("is-aggregation",""):this.removeAttribute("is-aggregation")}connectedCallback(){let t=this.file;t&&!Array.isArray(t)&&(this._pathInput=t),this.open&&(this._restoreFocusTo=document.activeElement),this._render(),document.addEventListener("mousedown",this._onOutsideClick)}disconnectedCallback(){document.removeEventListener("mousedown",this._onOutsideClick),this._clearTimers()}attributeChangedCallback(t,a,o){var r,n;if(t==="open"&&o!==null){this._restoreFocusTo===null&&(this._restoreFocusTo=document.activeElement),this._activeTab="metadata",this._copied=!1,this._gridlookCopied=!1;let i=this.file;i&&this._output===null&&this.status===it.READY&&!this.isAggregation&&this._emit("inspector-submit",{file:i,aggregationConfig:null})}if(t==="file"&&(o&&!o.startsWith("[")&&(this._pathInput=o),a!==null&&a!==o&&(this._output=null,this._domOutput=void 0,this._activeTab="metadata",this._copied=!1,this._gridlookCopied=!1,this.hasAttribute("error")&&this.removeAttribute("error"),this.hasAttribute("zarr-url")&&this.removeAttribute("zarr-url"))),t==="status"&&o===it.ERROR&&(this._activeTab="metadata"),t==="zarr-status-code"&&this.isConnected&&this._built){let i=o??"3";(r=this.querySelector("#nc-pre-steps"))==null||r.setAttribute("status-code",i),(n=this.querySelector("#nc-body-steps"))==null||n.setAttribute("status-code",i);return}this.isConnected&&this._render()}_emit(t,a){this.dispatchEvent(new CustomEvent(t,{bubbles:!0,composed:!0,detail:a}))}_clearTimers(){this._copyTimer!==null&&(clearTimeout(this._copyTimer),this._copyTimer=null),this._gridlookTimer!==null&&(clearTimeout(this._gridlookTimer),this._gridlookTimer=null)}_handleInspect(){this.isAggregation?this._emit("inspector-submit",{file:this.file,aggregationConfig:this._aggregationConfig}):this._pathInput.trim()&&this._emit("inspector-submit",{file:this._pathInput.trim(),aggregationConfig:null})}_handleInspectReload(){this._dropdownOpen=!1,this._syncDropdown(),this._pathInput.trim()&&this._emit("inspector-submit",{file:this._pathInput.trim(),aggregationConfig:{reload:!0}})}_copy(t,a){Dl(t).then(()=>{a==="zarr"?this._copied=!0:this._gridlookCopied=!0,this.open&&this._update();let o=setTimeout(()=>{a==="zarr"?this._copied=!1:this._gridlookCopied=!1,this.isConnected&&this.open&&this._update()},2e3);a==="zarr"?this._copyTimer=o:this._gridlookTimer=o}).catch(()=>{})}_render(){if(!this.open){this._teardown();return}(!this._built||this._builtMode!==this.isAggregation)&&this._build(),this._update()}_teardown(){this._clearTimers(),this.innerHTML="",this._built=!1,this._domOutput=void 0,this._domIframeUrl=void 0,this._domFileKey=void 0;let t=this._restoreFocusTo;this._restoreFocusTo=null,t&&t.isConnected&&typeof t.focus=="function"&&t.focus()}_q(t){return this.querySelector(t)}_toggle(t,a){t&&(t.hidden=!a)}_setCopyBtn(t,a,o){let r=this._q(t);r&&(r.setAttribute("title",a?"Copied!":o),r.innerHTML=Se(a?ve.check:ve.copy))}_build(){var t;Ol();let a=this.isAggregation;this._builtMode=a,this._domOutput=void 0,this._domIframeUrl=void 0,this._domFileKey=void 0;let o=`
      <div class="di-header">
        <div class="di-header-row">
          <div class="di-header-main">
            <h1 class="di-title">
              <span class="di-title-ico">${Se(a?ve.layers:ve.info)}</span>
              <span id="nc-title">${a?"Aggregate Files":"File Inspector"}</span>
            </h1>
            ${a?`<div id="nc-file-list-wrap" class="di-file-list" hidden>
                     <div id="nc-file-list-label" class="di-file-list-label"></div>
                     <ul id="nc-file-list"></ul>
                   </div>`:this._pathBarHtml()}
            <div id="nc-zarr-row" class="di-zarr-row" hidden>
              <div class="di-zarr-inner">
                <span class="di-muted" style="display:inline-flex;flex-shrink:0">${Se(ve.link)}</span>
                <span class="di-muted" style="font-weight:500;flex-shrink:0">Zarr:</span>
                <code id="nc-zarr-url" class="di-code"></code>
                <button id="nc-copy-zarr" class="di-btn di-btn-outline" title="Copy Zarr URL">${Se(ve.copy)}</button>
              </div>
            </div>
          </div>
          <button id="nc-close-btn" class="di-close" aria-label="Close">&times;</button>
        </div>
      </div>`,r=a?`<div id="nc-agg-form" class="di-agg-form" hidden>
           <h5>Aggregation Configuration</h5>
           <aggregation-config id="nc-agg-config"></aggregation-config>
           <div class="di-agg-actions">
             <button id="nc-cancel-btn" class="di-btn di-btn-secondary">Cancel</button>
             <button id="nc-aggregate-btn" class="di-btn di-btn-primary">${Se(ve.compress,!0)}Aggregate Files</button>
           </div>
         </div>`:"",n=`
      <div id="nc-pre-loading" class="di-center" hidden>
        <zarr-loading-steps id="nc-pre-steps" status-code="3"${a?" is-aggregation":""}></zarr-loading-steps>
        <p id="nc-pre-loading-text" class="di-muted" style="margin-top:12px;font-size:13px;"></p>
      </div>`,i=`
      <div id="nc-tabs-wrap" hidden>
        <div class="di-tabs" role="tablist" aria-label="Inspector views">
          <button id="nc-tab-metadata" data-tab="metadata" class="nc-tab-btn di-tab" role="tab" aria-controls="nc-metadata" aria-selected="true">${Se(ve.database,!0)}Metadata</button>
          <button id="nc-tab-gridlook" data-tab="gridlook" class="nc-tab-btn di-tab" role="tab" aria-controls="nc-gridlook" aria-selected="false">${Se(ve.cube,!0)}3D Viewer</button>
        </div>
        <div id="nc-body" class="di-body">
          <div id="nc-error" class="di-error" hidden>
            <div class="di-error-row">
              <span class="di-error-ico">${Se(ve.alert)}</span>
              <div class="di-error-body">
                <strong class="di-error-title">Error loading metadata</strong>
                <div id="nc-error-msg" class="di-error-msg"></div>
                <button id="nc-retry-btn" class="di-btn di-btn-danger">${Se(ve.refresh,!0)}Retry</button>
              </div>
            </div>
          </div>

          <div id="nc-loading" class="di-center" hidden>
            <zarr-loading-steps id="nc-body-steps" status-code="3"${a?" is-aggregation":""}></zarr-loading-steps>
            <p id="nc-loading-text" class="di-muted" style="margin-top:12px;font-size:13px;"></p>
          </div>

          <div id="nc-metadata" class="di-metadata" role="tabpanel" aria-labelledby="nc-tab-metadata" tabindex="0" hidden>
            <div id="nc-metadata-inner"></div>
          </div>

          <div id="nc-gridlook" role="tabpanel" aria-labelledby="nc-tab-gridlook" tabindex="0" hidden>
            <div class="di-gridlook-bar">
              <div class="di-gridlook-inner">
                <span class="di-gridlook-ico">${Se(ve.external)}</span>
                <span class="di-gridlook-label">GridLook URL:</span>
                <code id="nc-gridlook-url" class="di-gridlook-code"></code>
                <button id="nc-copy-gridlook" class="di-btn di-gridlook-btn" title="Copy link">${Se(ve.copy)}</button>
                <button id="nc-refresh-gridlook" class="di-btn di-gridlook-btn" title="Refresh GridLook viewer">${Se(ve.refresh)}</button>
                <button id="nc-open-gridlook" class="di-btn di-gridlook-btn-primary" title="Open in new tab">${Se(ve.external,!0)}Open in New Tab</button>
              </div>
            </div>
            <div id="nc-gridlook-frame" class="di-gridlook-frame"></div>
          </div>

          <div id="nc-empty-body" class="di-center" hidden>
            <span class="di-empty-ico">${Se(ve.database)}</span>
            <p id="nc-empty-body-text" class="di-empty-text"></p>
          </div>
        </div>
      </div>`,l=`
      <div id="nc-empty-main" class="di-center" hidden>
        <span class="di-empty-ico">${Se(ve.database)}</span>
        <p id="nc-empty-main-text" class="di-empty-text"></p>
      </div>`;this.innerHTML=`
      <div id="nc-backdrop" class="di-backdrop">
        <div class="di-modal" role="dialog" aria-modal="true" aria-labelledby="nc-title" tabindex="-1">
          ${o}
          ${r}
          ${n}
          ${i}
          ${l}
        </div>
      </div>`,(t=this._q("#nc-agg-config"))==null||t.setAttribute("initial-config",JSON.stringify(this._aggregationConfig)),this._built=!0,this._attach(),this._initialFocus()}_ensureIframe(){let t=this._q("#nc-gridlook-iframe");if(t)return t;let a=this._q("#nc-gridlook-frame");if(!a)return null;let o=document.createElement("iframe");o.id="nc-gridlook-iframe",o.title="GridLook 3D Viewer",o.setAttribute("sandbox","allow-scripts allow-same-origin allow-popups allow-downloads"),o.setAttribute("referrerpolicy","no-referrer"),o.setAttribute("loading","lazy");try{a.appendChild(o)}catch{}return o}_setIframeSrc(t,a){try{t.src=a}catch{}}_pathBarHtml(){return`
      <div class="di-pathbar">
        <label class="di-pathbar-label" for="nc-path-input">File path:</label>
        <div class="di-pathbar-row">
          <input id="nc-path-input" type="text" class="di-input" placeholder="/path/to/data.nc" />
          <div id="nc-dropdown-wrap" class="di-dropdown-wrap di-btn-group">
            <button id="nc-load-btn" class="di-btn di-btn-primary">${Se(ve.load,!0)}Load</button>
            <button id="nc-load-toggle" class="di-btn di-btn-primary di-btn-split" title="More load options">${Se(ve.chevronDown)}</button>
            <ul id="nc-dropdown-menu" class="di-menu" hidden>
              <li>
                <button id="nc-reload-btn" class="di-menu-item">${Se(ve.ban,!0)}Force Reload (bypass cache)</button>
              </li>
            </ul>
          </div>
        </div>
      </div>`}_attach(){var t,a,o,r,n,i,l,d,p,u,s,g,m,A;(t=this._q("#nc-backdrop"))==null||t.addEventListener("click",b=>{b.target===b.currentTarget&&this._emit("inspector-close",null)}),(a=this._q("#nc-backdrop"))==null||a.addEventListener("keydown",b=>this._onKeydown(b)),(o=this._q("#nc-close-btn"))==null||o.addEventListener("click",()=>this._emit("inspector-close",null)),(r=this._q("#nc-cancel-btn"))==null||r.addEventListener("click",()=>this._emit("inspector-close",null)),(n=this._q("#nc-load-btn"))==null||n.addEventListener("click",()=>this._handleInspect()),(i=this._q("#nc-aggregate-btn"))==null||i.addEventListener("click",()=>this._handleInspect()),(l=this._q("#nc-retry-btn"))==null||l.addEventListener("click",()=>this._handleInspect());let v=this._q("#nc-path-input");v&&(v.value=this._pathInput,v.addEventListener("input",b=>{this._pathInput=b.target.value}),v.addEventListener("keypress",b=>{b.key==="Enter"&&this._handleInspect()})),(d=this._q("#nc-load-toggle"))==null||d.addEventListener("click",()=>{this._dropdownOpen=!this._dropdownOpen,this._syncDropdown()}),(p=this._q("#nc-reload-btn"))==null||p.addEventListener("click",()=>this._handleInspectReload()),(u=this._q("#nc-copy-zarr"))==null||u.addEventListener("click",()=>{let b=this.zarrUrl;b&&this._copy(b,"zarr")}),(s=this._q("#nc-copy-gridlook"))==null||s.addEventListener("click",()=>{let b=this.zarrUrl;b&&this._copy(Uo(b),"gridlook")}),(g=this._q("#nc-refresh-gridlook"))==null||g.addEventListener("click",()=>{var b;this.zarrUrl&&((b=this._q("#nc-gridlook-iframe"))==null||b.remove(),this._domIframeUrl=void 0,this._update())}),(m=this._q("#nc-open-gridlook"))==null||m.addEventListener("click",()=>{let b=this.zarrUrl;b&&window.open(Uo(b),"_blank","noopener,noreferrer")}),this.querySelectorAll(".nc-tab-btn").forEach(b=>{b.addEventListener("click",()=>{let k=b.dataset.tab;k&&!b.disabled&&(this._activeTab=k,this._update())})}),(A=this._q("#nc-agg-config"))==null||A.addEventListener("config-change",b=>{this._aggregationConfig=b.detail})}_update(){var t,a;let o=this.isAggregation,r=this.status,n=r===it.LOADING,i=r===it.READY,l=r===it.ERROR,d=this.zarrUrl,p=this.file,u=Array.isArray(p)?p:[],s=this._output!=null&&this._output!=="",g=String(this.zarrStatusCode??3),m=o?"Aggregating files and loading metadata...":"Loading metadata...",A=o?"Configure aggregation settings and click 'Aggregate Files' to begin":"Enter a file path and click Load to inspect metadata";if(o){let K=this._q("#nc-file-list-wrap"),ne=u.length>0;if(this._toggle(K,ne),ne){let W=this._q("#nc-file-list-label");W&&(W.textContent=`Selected files (${u.length}):`);let le=this._q("#nc-file-list"),$=u.join("\0");le&&this._domFileKey!==$&&(le.replaceChildren(...u.map(pe=>{let w=document.createElement("li");return w.textContent=pe,w})),this._domFileKey=$)}}if(!o){let K=this._q("#nc-load-btn"),ne=this._q("#nc-load-toggle");K&&(K.disabled=n),ne&&(ne.disabled=n),this._syncDropdown()}let v=!!d&&d!==(typeof p=="string"?p:null);if(this._toggle(this._q("#nc-zarr-row"),v),v&&d){let K=this._q("#nc-zarr-url");K&&(K.textContent=d)}this._setCopyBtn("#nc-copy-zarr",this._copied,"Copy Zarr URL"),o&&this._toggle(this._q("#nc-agg-form"),i&&!s),this._toggle(this._q("#nc-pre-loading"),!d&&n),this._toggle(this._q("#nc-tabs-wrap"),!!d&&(s||n||l)),this._toggle(this._q("#nc-empty-main"),!d&&!n&&i&&!s);let b=this._q("#nc-pre-loading-text");b&&(b.textContent=m),(t=this._q("#nc-pre-steps"))==null||t.setAttribute("status-code",g);let k=this._q('[data-tab="metadata"]'),M=this._q('[data-tab="gridlook"]');if(k){let K=this._activeTab==="metadata";k.classList.toggle("di-tab-active",K),k.setAttribute("aria-selected",String(K))}if(M){let K=this._activeTab==="gridlook";M.disabled=r!==it.READY||!s,M.classList.toggle("di-tab-active",K),M.setAttribute("aria-selected",String(K))}this._toggle(this._q("#nc-error"),l&&this._activeTab==="metadata");let B=this._q("#nc-error-msg");B&&(B.textContent=this.error??""),this._toggle(this._q("#nc-loading"),n);let D=this._q("#nc-loading-text");D&&(D.textContent=m),(a=this._q("#nc-body-steps"))==null||a.setAttribute("status-code",g),this._toggle(this._q("#nc-metadata"),this._activeTab==="metadata"&&s&&!n&&!l);let P=this._q("#nc-metadata-inner");P&&this._domOutput!==this._output&&(P.innerHTML=this._output??"",this._domOutput=this._output);let U=this._activeTab==="gridlook"&&!!d;if(this._toggle(this._q("#nc-gridlook"),U),d){let K=Uo(d),ne=this._q("#nc-gridlook-url");if(ne&&(ne.textContent=K),U){let W=this._ensureIframe();W&&this._domIframeUrl!==K&&(this._setIframeSrc(W,K),this._domIframeUrl=K)}}this._setCopyBtn("#nc-copy-gridlook",this._gridlookCopied,"Copy link"),this._toggle(this._q("#nc-empty-body"),!s&&!n&&!l);let q=this._q("#nc-empty-body-text");q&&(q.textContent=A);let Z=this._q("#nc-empty-main-text");Z&&(Z.textContent=A)}_syncDropdown(){let t=this._q("#nc-dropdown-menu");t&&(t.hidden=!this._dropdownOpen)}_focusables(){let t=this._q(".di-modal");return t?Array.from(t.querySelectorAll('a[href],button:not([disabled]),input:not([disabled]),select:not([disabled]),textarea:not([disabled]),[tabindex]:not([tabindex="-1"])')).filter(a=>!a.closest("[hidden]")):[]}_initialFocus(){var t;let a=this._q("#nc-path-input")??this._focusables()[0]??this._q(".di-modal");(t=a?.focus)==null||t.call(a)}_onKeydown(t){if(t.key==="Escape"){t.stopPropagation(),this._emit("inspector-close",null);return}if(t.key!=="Tab")return;let a=this._focusables();if(a.length===0){t.preventDefault();return}let o=a[0],r=a[a.length-1],n=document.activeElement;t.shiftKey&&(n===o||!this.contains(n))?(t.preventDefault(),r.focus()):!t.shiftKey&&n===r&&(t.preventDefault(),o.focus())}};customElements.define("data-inspector",Na);ei=`<svg viewBox="0 0 448 512" width="10" height="10" style="vertical-align:middle;">
  <path d="M207.029 381.476L12.686 187.132c-9.373-9.373-9.373-24.569 0-33.941l22.667-22.667
    c9.357-9.357 24.522-9.375 33.901-.04L224 284.505l154.745-154.021c9.379-9.335
    24.544-9.317 33.901.04l22.667 22.667c9.373 9.373 9.373 24.569 0 33.941
    L240.971 381.476c-9.373 9.372-24.569 9.372-33.942 0z" fill="currentColor"/>
</svg>`,ti=`<svg viewBox="0 0 256 512" width="6" height="10" style="vertical-align:middle;">
  <path d="M224.3 273l-136 136c-9.4 9.4-24.6 9.4-33.9 0l-22.6-22.6c-9.4-9.4-9.4-24.6
    0-33.9l96.4-96.4-96.4-96.4c-9.4-9.4-9.4-24.6 0-33.9L54.3 103c9.4-9.4
    24.6-9.4 33.9 0l136 136c9.5 9.4 9.5 24.6.1 34z" fill="currentColor"/>
</svg>`,Go={aggregate:"auto",join:null,compat:null,data_vars:null,coords:null,dim:"",group_by:"",reload:!1,access_pattern:"map",chunk_size:16,map_primary_chunksize:1,timeout:120},Ha=class extends HTMLElement{constructor(){super(...arguments),this._config={...Go},this._showAdvanced=!1,this._handleChange=t=>{let a=t.target,o=a.dataset.field;if(!o)return;let r;if(a.type==="checkbox")r=a.checked;else if(a.type==="number"){let n=parseFloat(a.value);r=Number.isFinite(n)?n:parseInt(a.value,10)}else r=a.value===""?null:a.value;this._config={...this._config,[o]:r},this._emitChange(),this._updateConditionals()},this._handleClick=t=>{if(t.target.closest("#nc-advanced-toggle")){this._showAdvanced=!this._showAdvanced;let a=this.querySelector(".nc-advanced"),o=this.querySelector(".nc-chevron");a&&(a.style.display=this._showAdvanced?"block":"none"),o&&(o.innerHTML=this._showAdvanced?ei:ti)}}}connectedCallback(){try{let t=JSON.parse(this.getAttribute("initial-config")??"{}");this._config={...Go,...t}}catch{this._config={...Go}}this._render(),this.addEventListener("change",this._handleChange),this.addEventListener("click",this._handleClick)}disconnectedCallback(){this.removeEventListener("change",this._handleChange),this.removeEventListener("click",this._handleClick)}_emitChange(){this.dispatchEvent(new CustomEvent("config-change",{bubbles:!0,composed:!0,detail:this._config}))}_updateConditionals(){let t=this.querySelector(".nc-dim-field"),a=this.querySelector(".nc-map-chunksize");t&&(t.style.display=this._config.aggregate==="concat"?"block":"none"),a&&(a.style.display=this._config.access_pattern==="map"?"block":"none")}_render(){let t=this._config;this.innerHTML=`
      <div class="aggregation-config">

        <!-- Aggregation Method -->
        <div class="mb-3">
          <label class="form-label fw-semibold">Aggregation Method</label>
          <select class="form-select form-select-sm" data-field="aggregate">
            <option value="auto"   ${t.aggregate==="auto"?"selected":""}>Auto (Detect automatically)</option>
            <option value="merge"  ${t.aggregate==="merge"?"selected":""}>Merge (Combine variables)</option>
            <option value="concat" ${t.aggregate==="concat"?"selected":""}>Concat (Join along dimension)</option>
          </select>
          <div class="form-text text-muted">Auto mode will automatically detect the best aggregation method</div>
        </div>

        <!-- Timeout -->
        <div class="mb-3">
          <label class="form-label fw-semibold">Timeout (seconds)</label>
          <input type="number" class="form-control form-control-sm"
            data-field="timeout" min="10" max="3600" value="${t.timeout??120}">
          <div class="form-text text-muted">
            Max wait time for the aggregation to complete (default: 120 s). Increase for large datasets.
          </div>
        </div>

        <!-- Advanced Toggle -->
        <button type="button" id="nc-advanced-toggle"
          class="btn btn-link btn-sm p-0 mb-3 text-decoration-none">
          <span class="nc-chevron">${this._showAdvanced?ei:ti}</span>
          <span class="ms-2">Advanced Options</span>
        </button>

        <!-- Advanced Options -->
        <div class="nc-advanced" style="display:${this._showAdvanced?"block":"none"};">

          <!-- Dimension (concat only) -->
          <div class="mb-3 nc-dim-field" style="display:${t.aggregate==="concat"?"block":"none"};">
            <label class="form-label">Dimension to Concatenate Along</label>
            <input type="text" class="form-control form-control-sm"
              data-field="dim" placeholder="e.g., time, ensemble" value="${t.dim??""}">
            <div class="form-text text-muted">Leave empty to create a new dimension</div>
          </div>

          <!-- Join Mode -->
          <div class="mb-3">
            <label class="form-label">Join Mode</label>
            <select class="form-select form-select-sm" data-field="join">
              <option value=""      ${t.join?"":"selected"}>Default</option>
              <option value="outer" ${t.join==="outer"?"selected":""}>Outer (Union)</option>
              <option value="inner" ${t.join==="inner"?"selected":""}>Inner (Intersection)</option>
              <option value="left"  ${t.join==="left"?"selected":""}>Left</option>
              <option value="right" ${t.join==="right"?"selected":""}>Right</option>
              <option value="exact" ${t.join==="exact"?"selected":""}>Exact (Must match)</option>
            </select>
          </div>

          <!-- Compatibility Mode -->
          <div class="mb-3">
            <label class="form-label">Compatibility Mode</label>
            <select class="form-select form-select-sm" data-field="compat">
              <option value=""              ${t.compat?"":"selected"}>Default</option>
              <option value="no_conflicts"  ${t.compat==="no_conflicts"?"selected":""}>No Conflicts</option>
              <option value="equals"        ${t.compat==="equals"?"selected":""}>Equals</option>
              <option value="override"      ${t.compat==="override"?"selected":""}>Override</option>
            </select>
          </div>

          <!-- Data Variables -->
          <div class="mb-3">
            <label class="form-label">Data Variables Handling</label>
            <select class="form-select form-select-sm" data-field="data_vars">
              <option value=""          ${t.data_vars?"":"selected"}>Default</option>
              <option value="minimal"   ${t.data_vars==="minimal"?"selected":""}>Minimal</option>
              <option value="different" ${t.data_vars==="different"?"selected":""}>Different</option>
              <option value="all"       ${t.data_vars==="all"?"selected":""}>All</option>
            </select>
          </div>

          <!-- Coordinates -->
          <div class="mb-3">
            <label class="form-label">Coordinates Handling</label>
            <select class="form-select form-select-sm" data-field="coords">
              <option value=""          ${t.coords?"":"selected"}>Default</option>
              <option value="minimal"   ${t.coords==="minimal"?"selected":""}>Minimal</option>
              <option value="different" ${t.coords==="different"?"selected":""}>Different</option>
              <option value="all"       ${t.coords==="all"?"selected":""}>All</option>
            </select>
          </div>

          <!-- Group By -->
          <div class="mb-3">
            <label class="form-label">Group By (Optional)</label>
            <input type="text" class="form-control form-control-sm"
              data-field="group_by" placeholder="e.g., ensemble, variable" value="${t.group_by??""}">
            <div class="form-text text-muted">Group files by a specific attribute</div>
          </div>

          <!-- Reload Cache -->
          <div class="mb-3">
            <div class="form-check">
              <input type="checkbox" class="form-check-input" id="aggregation-reload"
                data-field="reload" ${t.reload?"checked":""}>
              <label class="form-check-label" for="aggregation-reload">Force Reload (bypass cache)</label>
            </div>
            <div class="form-text text-muted">
              Force server to fetch fresh data instead of using cached version
            </div>
          </div>

          <!-- Access Pattern -->
          <div class="mb-3">
            <label class="form-label">Access Pattern Optimization</label>
            <select class="form-select form-select-sm" data-field="access_pattern">
              <option value="map"         ${t.access_pattern==="map"?"selected":""}>Map (spatial slices)</option>
              <option value="time_series" ${t.access_pattern==="time_series"?"selected":""}>Time Series (temporal slices)</option>
            </select>
            <div class="form-text text-muted">Optimize chunk layout for your typical data access pattern</div>
          </div>

          <!-- Chunk Size -->
          <div class="mb-3">
            <label class="form-label">Target Chunk Size (MB)</label>
            <input type="number" class="form-control form-control-sm"
              data-field="chunk_size" step="0.1" min="1" max="1000" value="${t.chunk_size??16}">
            <div class="form-text text-muted">Target size for data chunks (default: 16 MB)</div>
          </div>

          <!-- Map Primary Chunksize (map pattern only) -->
          <div class="mb-3 nc-map-chunksize" style="display:${t.access_pattern==="map"?"block":"none"};">
            <label class="form-label">Primary Dimension Chunk Size</label>
            <input type="number" class="form-control form-control-sm"
              data-field="map_primary_chunksize" min="1" value="${t.map_primary_chunksize??1}">
            <div class="form-text text-muted">Number of time steps per chunk (for map access pattern)</div>
          </div>

        </div>
      </div>`}};customElements.define("aggregation-config",Ha);Il=`
  @keyframes zarrPulseRing {
    0%   { transform: scale(0.9); opacity: 0.7; }
    60%  { transform: scale(1.6); opacity: 0;   }
    100% { transform: scale(0.9); opacity: 0;   }
  }
  @keyframes zarrSpinArc {
    from { transform: rotate(0deg);   }
    to   { transform: rotate(360deg); }
  }
  @keyframes zarrMsgIn {
    0%   { opacity: 0; transform: translateY(5px);  }
    18%  { opacity: 1; transform: translateY(0);     }
    82%  { opacity: 1; transform: translateY(0);     }
    100% { opacity: 0; transform: translateY(-5px);  }
  }
  @keyframes zarrTrackFill {
    from { width: 0%; }
    to   { width: 100%; }
  }
  .zarr-spin { animation: zarrSpinArc 1.3s linear infinite; transform-origin: center; }
  .zarr-msg  { animation: zarrMsgIn 2.8s ease-in-out forwards; }
`,ai=[{id:"submitted",label:"Submitted"},{id:"queued",label:"Queued"},{id:"converting",label:"Converting"},{id:"ready",label:"Ready"}],Ce={done:"#0d9488",active:"#14b8a6",pending:"#d1d5db",track:"#e5e7eb",textOn:"#0f766e",textOff:"#9ca3af",msg:"#6b7280",timer:"#d1d5db"},Pa=["Reading file structure\u2026","Analysing coordinate metadata\u2026","Optimising chunk layout\u2026","Building Zarr metadata store\u2026","Assembling data variables\u2026","Applying access-pattern optimisation\u2026","Finalising dataset\u2026"],Ra=["Aligning coordinate indexes\u2026","Resolving variable conflicts\u2026","Concatenating along dimension\u2026","Merging datasets\u2026","Validating compatibility\u2026","Assembling aggregated store\u2026","Almost there\u2026"];Ua=class extends HTMLElement{constructor(){super(...arguments),this._msgIdx=0,this._msgKey=0,this._elapsed=0,this._startTime=Date.now(),this._msgTimer=null,this._elapsedTimer=null}static get observedAttributes(){return["status-code","is-aggregation"]}get statusCode(){return parseInt(this.getAttribute("status-code")??"3",10)}get isAggregation(){return this.hasAttribute("is-aggregation")}connectedCallback(){Ll(),this._render(),this._startTimers()}disconnectedCallback(){this._stopTimers()}attributeChangedCallback(){this.isConnected&&this._updateStages()}_startTimers(){this._startTime=Date.now(),this._elapsedTimer=setInterval(()=>{this._elapsed=Math.floor((Date.now()-this._startTime)/1e3);let t=this.querySelector(".zarr-elapsed");t&&(t.textContent=this._formatElapsed(this._elapsed))},1e3),this._startMsgCycle()}_startMsgCycle(){if(this._msgTimer&&clearInterval(this._msgTimer),Fa(this.statusCode)!==2)return;let t=this.isAggregation?Ra:Pa;this._msgTimer=setInterval(()=>{this._msgIdx=(this._msgIdx+1)%t.length,this._msgKey+=1,this._updateMessage()},2800)}_stopTimers(){this._msgTimer&&(clearInterval(this._msgTimer),this._msgTimer=null),this._elapsedTimer&&(clearInterval(this._elapsedTimer),this._elapsedTimer=null)}_formatElapsed(t){return t<60?`${t}s`:`${Math.floor(t/60)}m ${t%60}s`}_render(){let t=Fa(this.statusCode),a=this.isAggregation?Ra:Pa;this.innerHTML=`
      <div style="padding:28px 12px 20px;display:flex;flex-direction:column;align-items:center;gap:0;">
        <div class="zarr-stages" style="display:flex;align-items:center;width:100%;max-width:360px;margin-bottom:28px;">
          ${ai.map((o,r)=>oi(r,t,o.label)).join("")}
        </div>

        <div style="height:20px;"></div>

        <div class="zarr-msg-container" style="min-height:22px;text-align:center;">
          ${this._renderMessage(t,a)}
        </div>

        <div style="margin-top:10px;">
          <span class="zarr-elapsed" style="font-size:11px;color:${Ce.timer};
            font-variant-numeric:tabular-nums;letter-spacing:0.05em;">
            ${this._formatElapsed(this._elapsed)}
          </span>
        </div>
      </div>`}_renderMessage(t,a){return t===2?`<span key="${this._msgKey}" class="zarr-msg" style="font-size:13px;color:${Ce.msg};">
                ${a[this._msgIdx]}
              </span>`:t===1?`<span style="font-size:13px;color:${Ce.msg};">Waiting for a worker to pick up the task\u2026</span>`:""}_updateStages(){let t=Fa(this.statusCode),a=this.querySelector(".zarr-stages");a&&(a.innerHTML=ai.map((r,n)=>oi(n,t,r.label)).join(""));let o=this.querySelector(".zarr-msg-container");if(o){let r=this.isAggregation?Ra:Pa;o.innerHTML=this._renderMessage(t,r)}this._startMsgCycle()}_updateMessage(){let t=Fa(this.statusCode),a=this.querySelector(".zarr-msg-container");if(a){let o=this.isAggregation?Ra:Pa;a.innerHTML=this._renderMessage(t,o)}}};customElements.define("zarr-loading-steps",Ua);Bl="freva_auth_token=";qo=class{constructor(t,a={}){this.timer=null,this.cancelled=!1,this.zarrUrl=t,this.intervalMs=a.intervalMs??2e3,this.getAuthHeaders=a.getAuthHeaders??jo,this.getStatusUrl=a.getStatusUrl??Pl,this.onStatus=a.onStatus??(()=>{}),this.onError=a.onError??(()=>{})}start(){this.cancelled=!1,this.poll(),this.timer=setInterval(()=>{this.poll()},this.intervalMs)}stop(){this.cancelled=!0,this.timer!==null&&(clearInterval(this.timer),this.timer=null)}async poll(){try{let t=this.getStatusUrl(encodeURIComponent(this.zarrUrl)),a=await fetch(t,{credentials:"same-origin",headers:this.getAuthHeaders()});if(!a.ok){this.cancelled||this.onStatus(5,null);return}let o=await a.json(),r=o.status??5,n=o.reason??null;this.cancelled||(this.onStatus(r,n),r<=2&&this.stop())}catch(t){this.cancelled||this.onError(t instanceof Error?t.message:String(t))}}},ri={isZarr:!1,version:null,consolidated:!1};Ul=0;li=`<svg class="xr-icons" aria-hidden="true"><defs>
<symbol id="icon-database" viewBox="0 0 32 32">
  <path d="M16 0c-8.837 0-16 2.239-16 5v4c0 2.761 7.163 5 16 5s16-2.239 16-5v-4c0-2.761-7.163-5-16-5z"/>
  <path d="M16 17c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"/>
  <path d="M16 26c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"/>
</symbol>
<symbol id="icon-file-text2" viewBox="0 0 32 32">
  <path d="M28.681 7.159c-0.694-0.947-1.662-2.053-2.724-3.116s-2.169-2.030-3.116-2.724c-1.612-1.182-2.393-1.319-2.841-1.319h-15.5c-1.378 0-2.5 1.121-2.5 2.5v27c0 1.378 1.122 2.5 2.5 2.5h23c1.378 0 2.5-1.122 2.5-2.5v-19.5c0-0.448-0.137-1.23-1.319-2.841zM24.543 5.457c0.959 0.959 1.712 1.825 2.268 2.543h-4.811v-4.811c0.718 0.556 1.584 1.309 2.543 2.268zM28 29.5c0 0.271-0.229 0.5-0.5 0.5h-23c-0.271 0-0.5-0.229-0.5-0.5v-27c0-0.271 0.229-0.5 0.5-0.5 0 0 15.499-0 15.5 0v7c0 0.552 0.448 1 1 1h7v19.5z"/>
</symbol>
</defs></svg>`;jl=`
:root {
  --xr-font-color0: var(--jp-content-font-color0, rgba(0,0,0,1));
  --xr-font-color2: var(--jp-content-font-color2, rgba(0,0,0,.54));
  --xr-font-color3: var(--jp-content-font-color3, rgba(0,0,0,.38));
  --xr-border-color: var(--jp-border-color2, #e0e0e0);
  --xr-disabled-color: var(--jp-layout-color3, #bdbdbd);
  --xr-background-color: var(--jp-layout-color0, white);
  --xr-background-color-row-even: var(--jp-layout-color1, white);
  --xr-background-color-row-odd: var(--jp-layout-color2, #eeeeee);
}
.xr-wrap{display:block!important;min-width:300px;max-width:700px;line-height:1.6;padding-bottom:4px}
.xr-header{padding-top:6px;padding-bottom:6px;border-bottom:solid 1px var(--xr-border-color);margin-bottom:4px}
.xr-header>div,.xr-header>ul{display:inline;margin-top:0;margin-bottom:0}
.xr-obj-type,.xr-obj-name{margin-left:2px;margin-right:10px}
.xr-obj-type{color:var(--xr-font-color2)}
.xr-sections{padding-left:0!important;display:grid;grid-template-columns:150px auto auto 1fr 0 20px 0 20px;margin-block-start:0;margin-block-end:0}
.xr-section-item{display:contents}
.xr-section-item>input,.xr-var-item>input{display:block;opacity:0;height:0;margin:0}
.xr-section-item>input+label,.xr-var-item>input+label{color:var(--xr-disabled-color)}
.xr-section-item>input:enabled+label,.xr-var-item>input:enabled+label{cursor:pointer;color:var(--xr-font-color2)}
.xr-section-item>input:enabled+label:hover,.xr-var-item>input:enabled+label:hover{color:var(--xr-font-color0)}
.xr-section-summary{grid-column:1;color:var(--xr-font-color2);font-weight:500;white-space:nowrap}
.xr-section-summary>span{display:inline-block;padding-left:.3em}
.xr-section-summary-in:disabled+label{color:var(--xr-font-color2)}
.xr-section-summary-in+label:before{display:inline-block;content:"\u25BA";font-size:11px;width:15px;text-align:center}
.xr-section-summary-in:disabled+label:before{color:var(--xr-disabled-color)}
.xr-section-summary-in:checked+label:before{content:"\u25BC"}
.xr-section-summary-in:checked+label>span{display:none}
.xr-section-summary,.xr-section-inline-details{padding-top:4px}
.xr-section-inline-details{grid-column:2/-1}
.xr-section-details{grid-column:1/-1;margin-top:4px;margin-bottom:5px}
.xr-section-summary-in~.xr-section-details{display:none}
.xr-section-summary-in:checked~.xr-section-details{display:contents}
.xr-array-wrap{grid-column:1/-1;display:grid;grid-template-columns:20px auto}
.xr-array-wrap>label{grid-column:1;vertical-align:top}
.xr-preview{color:var(--xr-font-color3)}
.xr-array-preview,.xr-array-data{padding:0 5px!important;grid-column:2}
.xr-array-data,.xr-array-in:checked~.xr-array-preview{display:none}
.xr-array-in:checked~.xr-array-data,.xr-array-preview{display:inline-block}
.xr-dim-list{display:inline-block!important;list-style:none;padding:0!important;margin:0}
.xr-dim-list li{display:inline-block;padding:0;margin:0}
.xr-dim-list:before{content:"("}
.xr-dim-list:after{content:")"}
.xr-dim-list li:not(:last-child):after{content:",";padding-right:5px}
.xr-has-index{font-weight:bold}
.xr-var-list,.xr-var-item{display:contents}
.xr-var-item>div,.xr-var-item label,.xr-var-item>.xr-var-name span{background-color:var(--xr-background-color-row-even);border-color:var(--xr-background-color-row-odd);margin-bottom:0;padding-top:2px}
.xr-var-list>li:nth-child(odd)>div,.xr-var-list>li:nth-child(odd)>label,.xr-var-list>li:nth-child(odd)>.xr-var-name span{background-color:var(--xr-background-color-row-odd);border-color:var(--xr-background-color-row-even)}
.xr-var-name{grid-column:1}
.xr-var-dims{grid-column:2}
.xr-var-dtype{grid-column:3;text-align:right;color:var(--xr-font-color2)}
.xr-var-preview{grid-column:4}
.xr-index-preview{grid-column:2/5;color:var(--xr-font-color2)}
.xr-var-name,.xr-var-dims,.xr-var-dtype,.xr-preview,.xr-attrs dt{white-space:nowrap;overflow:hidden;text-overflow:ellipsis;padding-right:10px}
.xr-var-name:hover,.xr-var-dims:hover,.xr-var-dtype:hover,.xr-attrs dt:hover{overflow:visible;width:auto;z-index:1}
.xr-var-attrs,.xr-var-data,.xr-index-data{display:none;border-top:2px dotted var(--xr-background-color);padding-bottom:20px!important;padding-top:10px!important}
.xr-var-attrs-in:checked~.xr-var-attrs,.xr-var-data-in:checked~.xr-var-data{display:block}
.xr-var-data>table{float:right}
.xr-var-data>pre,.xr-var-data>table>tbody>tr{background-color:transparent!important}
.xr-var-name span,.xr-var-data,.xr-attrs{padding-left:25px!important}
.xr-attrs,.xr-var-attrs,.xr-var-data,.xr-index-data{grid-column:1/-1}
dl.xr-attrs{padding:0;margin:0;display:grid;grid-template-columns:125px auto}
.xr-attrs dt,.xr-attrs dd{padding:0;margin:0;float:left;padding-right:10px;width:auto}
.xr-attrs dt{font-weight:normal;grid-column:1}
.xr-attrs dd{grid-column:2;white-space:pre-wrap;word-break:break-all}
.xr-icons{position:absolute;width:0;height:0;overflow:hidden}
.xr-icon-database,.xr-icon-file-text2{display:inline-block;vertical-align:middle;width:1em;height:1.5em!important;stroke-width:0;stroke:currentColor;fill:currentColor}
.xr-var-attrs-in:checked+label>.xr-icon-file-text2,.xr-var-data-in:checked+label>.xr-icon-database{color:var(--xr-font-color0);filter:drop-shadow(1px 1px 5px var(--xr-font-color2));stroke-width:.8px}
.xr-var-item>input+label{cursor:pointer;color:var(--xr-font-color2);padding:0 1px}
`,di=!1});var go="Result stream too big.";function Ao(e,t,a,o){let r=t?.[a]??a,n=e[r]??e[a],i=n?n[o]:void 0;return typeof i=="string"&&i.length>0?i:null}var Lt=["freva","cmip5","cmip6","cordex","user"],ht=["project","product","institute","model","experiment","time_frequency","realm","variable","ensemble","time_aggregation"],Bt=["cmor_table","dataset","driving_model","format","grid_label","level_type","rcm_name","rcm_version","fs_type","grid_id","user"],Vs=100,js=new Set(["variable","ensemble"]);function Gr(e){return{flavour:e.flavour,uniqKey:"file",selected:{},baseFilters:{},time:null,bbox:null,rows:[],totalCount:0,facets:[],primaryFacets:[],overviewShape:[],facetMapping:{},attributeKeys:[],flavours:[...Lt],flavourMaps:structuredClone(Xs),start:0,search:"idle",lastRequestId:0,facetsVersion:0,rowsVersion:0,rowsEpoch:0,layout:"results",theme:"night",view:"list",pickedKeys:new Set,focusKey:null,detailSource:"focus",detailsOpen:!1,details:"idle",detailsCache:new Map,terminalDraft:"",externalEdits:0,terminalFocused:!1,terminalTab:"cli",overviewFilters:{},overviewSort:{},overviewCollapsed:new Set,overviewAddOpen:!1,overviewSpan:{},overviewStacked:!1,overviewStackSeen:[],overviewSnapshot:null,overviewH:{},overviewOrder:[],overviewStale:!1,sidebarOpen:new Set,sidebarAddOpen:!1,sidebarSeeded:!1,sidebarCollapsed:!1,status:"",metadata:{},metadataVersion:0}}function vo(e){let t=e.facets.filter(o=>o.values.length);if(!t.length)return;let a=new Map(e.overviewShape.map(o=>[o.key,o]));for(let o of t)a.set(o.key,{key:o.key,label:o.label});e.overviewShape=[...a.values()]}function Qr(e){let t=new Map(e.facets.map(r=>[r.key,r])),a=[],o=new Set;for(let r of e.overviewShape){let n=t.get(r.key);a.push(n??{key:r.key,label:r.label,values:[],hasMore:!1}),o.add(r.key)}for(let r of e.facets)o.has(r.key)||a.push(r);return a}var ba="_not_";function Pe(e){return e.length>ba.length&&e.toLowerCase().endsWith(ba)?{baseKey:e.slice(0,e.length-ba.length),negated:!0}:{baseKey:e,negated:!1}}function mt(e){return Pe(e).negated?e:`${e}${ba}`}function fe(e){return Pe(e).baseKey}function jt(e,t){return t?mt(e):fe(e)}function Kt(e,t){return e.selected[fe(t)]??[]}function Ze(e,t){return e.selected[mt(t)]??[]}function rt(e,t){return Kt(e,t).length+Ze(e,t).length}function Js(e,t){delete e.selected[fe(t)],delete e.selected[mt(t)]}function Wr(e,t,a){delete e.selected[a?mt(t):fe(t)]}function wo(e,t,a,o){let{baseKey:r,negated:n}=Pe(t);return jt(Rt(e,r,a,o),n)}function gt(e,t,a){let o=fe(t);return(e.selected[o]??[]).includes(a)||Pt(e,o,a)}function wa(e,t,a){return Ze(e,t).includes(a)}function qr(e,t,a,o){let r=jt(t,!o),n=e.selected[r];if(n){let i=n.indexOf(a);i>=0&&(n.splice(i,1),n.length===0&&delete e.selected[r])}Ks(e,jt(t,o),a)}function Pt(e,t,a){let o=fe(t).toLowerCase();for(let r of Object.keys(e.baseFilters)){let{baseKey:n,negated:i}=Pe(r);if(!i&&Rt(e,n,"freva",e.flavour).toLowerCase()===o&&e.baseFilters[r].includes(a))return!0}return!1}function Qe(e,t){return nt(e).has(fe(t).toLowerCase())}function Vr(e,t){return Qe(e,t.key)?t.values.filter(a=>Pt(e,t.key,a.value)):t.values}function Zt(e){let t=[];for(let[a,o]of Object.entries(e.baseFilters)){let r=wo(e,a,"freva",e.flavour);for(let n of o)t.push([r,n])}return t}function jr(e){let t=[];for(let[a,o]of Object.entries(e.baseFilters)){let{baseKey:r,negated:n}=Pe(a);if(!n)continue;let i=Rt(e,r,"freva",e.flavour);for(let l of o)t.push([i,l])}return t}function Ks(e,t,a){let o=e.selected[t]??(e.selected[t]=[]),r=o.indexOf(a);r>=0?(o.splice(r,1),o.length===0&&delete e.selected[t]):o.push(a)}function Jr(e){e.selected={},e.time=null,e.bbox=null}function Kr(e,t){Js(e,t)}function Zr(e){return Object.keys(e.selected).length>0||!!e.time||!!e.bbox}function Yr(e){return e.replace(/_/g," ").replace(/\b\w/g,t=>t.toUpperCase())}function Yt(e,t){return e.facetMapping[t]??Yr(t)}function xo(e){let t={};for(let a of e){let o=typeof a.flavour_name=="string"?a.flavour_name:null,r=a.mapping;if(!o||!r||typeof r!="object")continue;let n={},i={};for(let[l,d]of Object.entries(r))typeof d=="string"&&(n[l]=d,i[d]=l);t[o]={forward:n,backward:i}}return t}function Rt(e,t,a,o){if(a===o)return t;let r=e.flavourMaps[a]?.backward[t]??t;return e.flavourMaps[o]?.forward[r]??r}function Xr(e,t,a,o){if(a===o)return t;let r={};for(let[n,i]of Object.entries(t))r[wo(e,n,a,o)]=i;return r}var Zs=["project","product","institute","model","experiment","time_frequency","realm","variable","ensemble","time_aggregation","cmor_table","dataset","driving_model","format","grid_id","grid_label","level_type","rcm_name","rcm_version","fs_type"],Ys={cmip5:{ensemble:"member_id",institute:"institution_id",model:"model_id"},cmip6:{experiment:"experiment_id",ensemble:"member_id",institute:"institution_id",model:"source_id",project:"mip_era",product:"activity_id",variable:"variable_id",time_frequency:"frequency",cmor_table:"table_id"},cordex:{institute:"institution",product:"domain"}},Xs=xo(["freva","cmip5","cmip6","cordex"].map(e=>{let t=Ys[e]??{},a={};for(let o of Zs)a[o]=t[o]??o;return{flavour_name:e,mapping:a}}));function At(e,t,a){return Ao(e.metadata,e.flavourMaps[e.flavour]?.backward,t,a)}function Xt(e){let t=new Set;for(let a of e.facets)t.add(a.key.toLowerCase());for(let a of e.attributeKeys)t.add(a.toLowerCase());for(let a of e.primaryFacets)t.add(a.toLowerCase());return t}function yo(e,t){let a=fe(t).toLowerCase();if((e.selected[a]??[]).length>0||(e.selected[mt(a)]??[]).length>0)return null;let o=e.facets.find(r=>r.key.toLowerCase()===a);return!o||o.hasMore||o.values.length===0?null:new Set(o.values.map(r=>r.value))}function _t(e,t){let a={},o=[],r=Xt(e),n=nt(e),i=(l,d)=>{o.push(`${l}=${It(d)}`)};for(let l of Object.keys(t)){let d=fe(l).toLowerCase();if(n.has(d)||r.size===0||!r.has(d)){for(let u of t[l])i(l,u);continue}let p=yo(e,l);for(let u of t[l]){if(p&&!p.has(u)){i(l,u);continue}(a[l]??(a[l]=[])).push(u)}}for(let l of Object.keys(a)){let{baseKey:d,negated:p}=Pe(l);if(!p)continue;let u=a[d];if(!u?.length)continue;let s=a[l].filter(g=>u.includes(g)?(i(l,g),!1):!0);s.length?a[l]=s:delete a[l]}return{accepted:a,rejected:o}}function _r(e,t){let a=Object.keys(e),o=Object.keys(t);if(a.length!==o.length)return!1;for(let r of a){let n=e[r],i=t[r];if(!i||n.length!==i.length)return!1;let l=new Set(i);for(let d of n)if(!l.has(d))return!1}return!0}function $r(e,t){return{key:t==="uri"?String(e.uri??e.file):e.file,file:e.file,fsType:e.fs_type,raw:e}}function xa(e){let t=[];for(let a=0;a+1<e.length;a+=2)t.push([String(e[a]),Number(e[a+1])]);return t}function So(e){let t=e.facets??{},a=e.facet_mapping??{},o=e.primary_facets??[],r=[],n=new Set;for(let i of o)i in t&&!n.has(i)&&(r.push(i),n.add(i));for(let i of Object.keys(t))n.has(i)||(r.push(i),n.add(i));return r.map(i=>{let d=xa(t[i]??[]).map(([u,s])=>({value:u,count:s})),p=d.length>=Vs||js.has(i);return{key:i,label:a[i]??Yr(i),values:d,hasMore:p}})}function en(e){let t={};for(let[a,o]of Object.entries(e)){let r=xa(o);if(r.length===0)continue;let n=r.map(i=>i[0]);t[a]=n.length===1?n[0]:n}return t}function ya(e){let a=(e.split("/").pop()??e).match(/_(\d{4,8})-(\d{4,8})(?:\.\w+)?$/);if(!a)return null;let o=i=>{if(i.length!==4&&i.length!==6&&i.length!==8)return null;let l=i.slice(0,4);if(i.length===4)return l;let d=Number(i.slice(4,6));if(d<1||d>12)return null;if(i.length===6)return`${l}-${i.slice(4,6)}`;let p=Number(i.slice(6,8));return p<1||p>31?null:`${l}-${i.slice(4,6)}-${i.slice(6,8)}`},r=o(a[1]),n=o(a[2]);return!r||!n||!ot(r)||!ot(n)||ft(r)>ft(n)?null:`${r} \u2192 ${n}`}var va=encodeURIComponent;function _s(e){let t=nt(e);return Sa(e).filter(([a])=>!t.has(fe(a).toLowerCase()))}function Sa(e){let t=[];for(let a of Object.keys(e.selected))for(let o of e.selected[a])t.push([a,o]);return t}function Co(e){if(!e||!e.from&&!e.to)return[];let t=e.from||"1",a=e.to||"9999";return[["time",`${t} TO ${a}`],["time_select",e.mode]]}function ko(e){return e?[["bbox",`${e.minLon},${e.maxLon},${e.minLat},${e.maxLat}`],["bbox_select",e.mode]]:[]}function Mo(e){return[...Sa(e),...Co(e.time),...ko(e.bbox)]}function tn(e){let t={};if(!e)return t;for(let[a,o]of Object.entries(e)){let r=(Array.isArray(o)?o:[o]).map(n=>String(n)).filter(n=>n.length>0);r.length&&(t[a]=r)}return t}function nt(e){let t=new Set;for(let a of Object.keys(e.baseFilters)){let{baseKey:o,negated:r}=Pe(a);r||t.add(Rt(e,o,"freva",e.flavour).toLowerCase())}return t}function an(e){let t=nt(e),a={};for(let[r,n]of Object.entries(e.baseFilters)){let i=wo(e,r,"freva",e.flavour),l=a[i]??(a[i]=[]);for(let d of n)l.includes(d)||l.push(d)}for(let[r,n]of Object.entries(e.selected)){if(t.has(fe(r).toLowerCase()))continue;let i=a[r]??(a[r]=[]);for(let l of n)i.includes(l)||i.push(l)}let o=[];for(let r of Object.keys(a))for(let n of a[r])o.push([r,n]);return o}function bt(e){return[...an(e),...Co(e.time),...ko(e.bbox)].map(([t,a])=>`${va(t)}=${va(a)}`).join("&")}function To(e){let t=new URLSearchParams(e),a=null,o={};for(let[l,d]of t.entries()){if(l==="flavour"){a=d;continue}$s.has(l)||(o[l]??(o[l]=[])).push(d)}let{time:r,bbox:n,rest:i}=ka(o);return{flavour:a,selected:i,time:r,bbox:n}}var $s=new Set(["siv","q","translate","max-results","start","fields","facets","multi-version","multi_version","uniq_key"]);function on(e,t){let a=nt(e),o=fe(t).toLowerCase();return[...an(e),...Co(e.time),...ko(e.bbox)].filter(([r])=>{let n=fe(r).toLowerCase();return n!==o||a.has(n)}).map(([r,n])=>`${va(r)}=${va(n)}`).join("&")}function It(e){return/[\s"]/.test(e)?`"${e.replace(/\\/g,"\\\\").replace(/"/g,'\\"')}"`:e}function vt(e){let t=[],a=0,o=e.length;for(;a<o;){let r=a;if(/\s/.test(e[a])){for(;a<o&&/\s/.test(e[a]);)a++;t.push({kind:"ws",raw:e.slice(r,a),value:e.slice(r,a),start:r,end:a});continue}let n="",i=!1;for(;a<o;){let l=e[a];if(!i&&/\s/.test(l))break;if(l==='"'){i=!i,a++;continue}if(i&&l==="\\"&&a+1<o&&(e[a+1]==='"'||e[a+1]==="\\")){n+=e[a+1],a+=2;continue}n+=l,a++}t.push({kind:"tok",raw:e.slice(r,a),value:n,start:r,end:a})}return t}function el(e){return vt(e).filter(t=>t.kind==="tok").map(t=>t.value)}function wt(e){let t={};for(let a of el(e)){let o=a.indexOf("=");if(o<1)continue;let r=a.slice(0,o).toLowerCase(),n=a.slice(o+1);if(!n)continue;let i=t[r]??(t[r]=[]);i.includes(n)||i.push(n)}return t}function Nr(e){return`"${e.replace(/\\/g,"\\\\").replace(/"/g,'\\"')}"`}function tl(e){let t=nt(e);return Object.keys(e.selected).filter(a=>!t.has(fe(a).toLowerCase())).map(a=>[a,e.selected[a]])}var al=/^(['"])([\s\S]*)\1$/;function bo(e){let t=e.trim(),a=t.match(al);return a?a[2].replace(/\\(["'\\])/g,"$1"):t}function Hr(e,t){let a=[],o=0,r="",n="";for(let i=0;i<e.length;i++){let l=e[i];if(r){if(l==="\\"&&i+1<e.length){n+=l+e[i+1],i++;continue}n+=l,l===r&&(r="");continue}if(l==='"'||l==="'"){r=l,n+=l;continue}l==="["||l==="{"?o++:(l==="]"||l==="}")&&(o=Math.max(0,o-1)),t.includes(l)&&o===0?(a.push(n),n=""):n+=l}return n.trim()&&a.push(n),a}function ut(e,t){return`${e}=${t.length>1?`[${t.map(Nr).join(", ")}]`:Nr(t[0])}`}function rn(e){let t=e.trim();t=t.replace(/^\bdatabrowser\s*\(/,"").replace(/^\{/,"").replace(/[})]\s*$/,"");let a=[];for(let o of Hr(t,`,
`)){let r=o.match(/^\s*(.+?)\s*[:=]\s*([\s\S]+?)\s*,?\s*$/);if(!r)continue;let n=bo(r[1]);if(!n||/^(host|flavour)$/i.test(n))continue;let i=r[2].trim(),l=i.startsWith("[")?Hr(i.replace(/^\[/,"").replace(/\]$/,""),",").map(bo).filter(Boolean):[bo(i)];for(let d of l)d&&a.push(`${n}=${It(d)}`)}return a.join(" ")}function Eo(e){let t=[];e.flavour!=="freva"&&t.push({code:ut("flavour",[e.flavour]),scope:!1});let a=new Map;for(let[o,r]of Zt(e)){let n=a.get(o)??[];n.push(r),a.set(o,n)}for(let[o,r]of a)t.push({code:ut(o,r),scope:!0});return t}function nn(e){let t=Eo(e).map(r=>`    ${r.code},`),a=Oo(e).split(`
`).filter(r=>r.trim()).map(r=>`    ${r}`),o=[...t,...a];return o.length?`from freva_client import databrowser
databrowser(
${o.join(`
`)}
)`:`from freva_client import databrowser
databrowser()`}function Do(e){if(!e)return null;let t=a=>Math.round(a*100)/100;return{...e,minLon:t(e.minLon),maxLon:t(e.maxLon),minLat:t(e.minLat),maxLat:t(e.maxLat)}}var We=new Set(["time","time_select","bbox","bbox_select"]),Jt=["flexible","strict","file"],Aa="flexible";function ol(e){let t=e.split(",").map(l=>l.trim());if(t.length!==4)return{box:null,error:"bbox needs 4 numbers: minLon,maxLon,minLat,maxLat"};let a=t.map(Number);if(a.some(l=>!Number.isFinite(l)))return{box:null,error:"bbox values must be numbers"};let[o,r,n,i]=a;return o>r?{box:null,error:"bbox: minLon must be \u2264 maxLon"}:n>i?{box:null,error:"bbox: minLat must be \u2264 maxLat"}:Math.abs(n)>90||Math.abs(i)>90?{box:null,error:"bbox: latitude must be within \xB190"}:Math.abs(o)>180||Math.abs(r)>180?{box:null,error:"bbox: longitude must be within \xB1180"}:{box:{minLon:o,maxLon:r,minLat:n,maxLat:i}}}var sn=/^(\d{4})(?:-(\d{2})(?:-(\d{2})(?:[ T](\d{2}):(\d{2})(?::(\d{2}))?)?)?)?$/;function Ca(e){let t=e.trim();return t===""||t==="*"}function ln(e,t,a,o=0,r=0,n=0){let i=new Date(0);return i.setUTCFullYear(e,t-1,a),i.setUTCHours(o,r,n,0),i}function ot(e){let t=e.trim();if(Ca(t))return!0;let a=sn.exec(t);if(!a)return!1;let[,o,r,n,i,l,d]=a,p=r?+r:1,u=n?+n:1,s=i?+i:0,g=l?+l:0,m=d?+d:0;if(p<1||p>12||s>23||g>59||m>59)return!1;if(n){let A=ln(+o,p,u);if(A.getUTCFullYear()!==+o||A.getUTCMonth()!==p-1||A.getUTCDate()!==u)return!1}return!0}function ft(e){let t=sn.exec(e.trim());if(!t)return NaN;let[,a,o,r,n,i,l]=t;return ln(+a,o?+o:1,r?+r:1,n?+n:0,i?+i:0,l?+l:0).getTime()}function rl(e){let t=e.split(/\s+TO\s+/i);if(t.length!==2)return{range:null,error:'time needs a range: time="2000 TO 2010"'};let a=n=>n.trim()==="*"?"":n.trim(),o=a(t[0]),r=a(t[1]);return!o&&!r?{range:null,error:"time range is empty"}:o&&!ot(o)?{range:null,error:`not a valid time: "${o}"`}:r&&!ot(r)?{range:null,error:`not a valid time: "${r}"`}:o&&r&&ft(o)>ft(r)?{range:null,error:"time range is reversed - the start is after the end"}:{range:{from:o,to:r}}}function cn(e){return`${e.from||"1"} TO ${e.to||"9999"}`}function Ur(e){let t=e.trim().toLowerCase();return Jt.includes(t)?t:null}function ka(e){let t={},a=[],o=null,r=null,n=d=>{let p=e[d];return p&&p.length?p[p.length-1]:null};for(let d of Object.keys(e))We.has(d)||(t[d]=e[d]);let i=n("bbox");if(i!==null){let{box:d,error:p}=ol(i);if(p&&a.push(p),d){let u=n("bbox_select"),s=u===null?Aa:Ur(u);u!==null&&s===null&&a.push(`bbox_select must be one of ${Jt.join(", ")}`),r={...d,mode:s??Aa}}}let l=n("time");if(l!==null){let{range:d,error:p}=rl(l);if(p&&a.push(p),d){let u=n("time_select"),s=u===null?Aa:Ur(u);u!==null&&s===null&&a.push(`time_select must be one of ${Jt.join(", ")}`),o={...d,mode:s??Aa}}}return{time:o,bbox:r,rest:t,errors:a}}function dn(e){let t=[],a=e.time;a&&(a.from||a.to)&&(t.push(`time=${It(cn(a))}`),t.push(`time_select=${a.mode}`));let o=e.bbox;o&&(t.push(`bbox=${o.minLon},${o.maxLon},${o.minLat},${o.maxLat}`),t.push(`bbox_select=${o.mode}`));for(let[r,n]of _s(e))t.push(`${r}=${It(n)}`);return t.join(" ")}function Oo(e){let t=[],a=e.time;a&&(a.from||a.to)&&(t.push(`${ut("time",[cn(a)])},`),t.push(`${ut("time_select",[a.mode])},`));let o=e.bbox;o&&(t.push(`${ut("bbox",[`${o.minLon},${o.maxLon},${o.minLat},${o.maxLat}`])},`),t.push(`${ut("bbox_select",[o.mode])},`));for(let[r,n]of tl(e))t.push(`${ut(r,n)},`);return t.join(`
`)}function pn(e){let t=Array.isArray(e)?e[0]:e;if(typeof t!="string")return null;let a=t.match(/ENVELOPE\s*\(([^)]*)\)/i);if(!a)return null;let o=a[1].split(",").map(p=>Number(p.trim()));if(o.length!==4||o.some(p=>!Number.isFinite(p)))return null;let[r,n,i,l]=o,d=_e({minLon:r,maxLon:n,minLat:l,maxLat:i});return{minLon:d.minLon,maxLon:d.maxLon,minLat:d.minLat,maxLat:d.maxLat}}function un(e){if(typeof e!="string")return null;let t=e.match(/^\s*[[{]?\s*(.+?)\s+TO\s+(.+?)\s*[\]}]?\s*$/i);if(!t)return null;let a=n=>n.replace(/T/," ").replace(/\s*(00:00:00|23:59:00|23:59:59)$/,"").trim(),o=a(t[1]),r=a(t[2]);return!o||!r?null:`${o} \u2192 ${r}`}function _e(e){let t=Math.min(e.minLat,e.maxLat),a=Math.max(e.minLat,e.maxLat);if(Math.abs(e.maxLon-e.minLon)>=359.5)return{minLon:-180,maxLon:180,minLat:t,maxLat:a,global:!0,wraps:!1};let o=i=>{let l=((i+180)%360+360)%360-180;return l===-180&&i>0&&(l=180),l},r=o(e.minLon),n=o(e.maxLon);return{minLon:r,maxLon:n,minLat:t,maxLat:a,global:!1,wraps:r>n}}function fn(e,t,a,o,r,n={}){let l=`${e.replace(/\/+$/,"")}/${t}/${encodeURIComponent(a)}/${o}?translate=true`;return n.maxResults!==void 0&&(l+=`&max-results=${n.maxResults}`),n.start&&(l+=`&start=${n.start}`),r&&(l+=`&${r}`),l}var xt=encodeURIComponent,De=class extends Error{constructor(t,a,o,r=!1){super(a),this.name="ApiError",this.status=t,this.detail=o,this.aborted=r}};function nl(e,t){switch(e){case 401:return"Sign in again to continue.";case 403:return"Access denied.";case 404:return"Not found - it may be temporarily unavailable.";case 413:return"Result set too large - narrow your search.";case 422:return t?`Invalid query: ${t}`:"Invalid query - check your facets.";case 429:return"Rate-limited - wait a moment and try again.";case 500:case 503:return"Service error - try again.";default:return`Request failed (${e}).`}}var Ma=class{constructor(t,a){this.channels=new Map,this.oneOffSet=new Set,this.reqCounter=0,this.cfg=t,this.base=t.apiBase.replace(/\/+$/,""),a.add(()=>{for(let o of this.channels.values())o.abort();for(let o of this.oneOffSet)o.abort()})}nextRequestId(){return++this.reqCounter}oneOffPending(){return this.oneOffSet.size}channelSignal(t){this.channels.get(t)?.abort();let a=new AbortController;return this.channels.set(t,a),a.signal}async oneOff(t){let a=new AbortController;this.oneOffSet.add(a);try{return await t(a.signal)}finally{this.oneOffSet.delete(a)}}headers(t){let a={...t};if(this.cfg.authEnabled){let r=this.cfg.getAuthToken();r&&(a.Authorization=`Bearer ${r}`)}let o=this.cfg.getCsrfToken();return o&&(a["X-CSRFToken"]=o),a}async request(t,a){let o,{headers:r,...n}=a??{};try{o=await fetch(t,{...n,credentials:"same-origin",headers:this.headers(r)})}catch(i){throw i instanceof DOMException&&i.name==="AbortError"?new De(0,"Request aborted.",void 0,!0):new De(0,"Network error - check your connection.")}if(!o.ok){let i;try{let l=await o.clone().json();typeof l.detail=="string"&&(i=l.detail)}catch{}throw new De(o.status,nl(o.status,i),i)}return o}async json(t,a){return await(await this.request(t,a)).json()}searchUrl(t,a,o,r,n){return fn(this.base,t,a,o,r,n??{})}catalogueUrl(t,a,o,r){let n=`${this.base}/${t}-catalogue/${xt(a)}/${o}?translate=true&max-results=100000`;return r&&(n+=`&${r}`),n}dataSearchUrl(t,a,o){let r=`${this.base}/data-search/${xt(t)}/${a}?translate=true&max-results=100000`;return o&&(r+=`&${o}`),r}extendedSearch(t,a,o,r){let n=this.searchUrl("extended-search",t,a,o,{maxResults:100,start:r?.start});return this.json(n,{signal:r?.signal})}metadataSearch(t,a,o,r){let n=this.searchUrl("metadata-search",t,a,o,{});return this.json(n,{signal:r})}overview(){return this.oneOff(t=>this.json(`${this.base}/overview`,{signal:t}))}filePathMetadata(t,a,o){let r=a.map(i=>`file=${xt(i)}`).join("&"),n=`${this.base}/extended-search/${xt(t)}/file?max-results=100&translate=true&fields=time&fields=bbox&${r}`;return this.json(n,{signal:o})}catalogueResponse(t,a,o,r,n){return this.request(this.catalogueUrl(t,a,o,r),{signal:n})}manifestResponse(t,a,o,r){return this.request(this.dataSearchUrl(t,a,o),{signal:r})}async dataSearchText(t,a,o){return(await this.oneOff(n=>this.request(this.dataSearchUrl(t,a,o),{signal:n}))).text()}async load(t,a){let o=`${this.base}/load/${xt(t)}?${a}`;return(await this.oneOff(n=>this.request(o,{method:"GET",signal:n}))).text()}zarrConvert(t){return this.oneOff(a=>this.json(`${this.base}/data-portal/zarr/convert`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(t),signal:a}))}zarrStatus(t){return this.oneOff(a=>this.json(`${this.base}/data-portal/zarr-utils/status?url=${xt(t)}`,{signal:a}))}shareZarr(t){return this.oneOff(a=>this.json(`${this.base}/data-portal/share-zarr`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(t),signal:a}))}zarrHtmlUrl(t){return`${this.base}/data-portal/zarr-utils/html?url=${xt(t)}`}listFlavours(){return this.oneOff(t=>this.json(`${this.base}/flavours`,{signal:t}))}};function c(e,t,a){let o=document.createElement(e),r=null;if(t)for(let[n,i]of Object.entries(t))i==null||i===!1||(n==="class"?o.className=String(i):n==="text"?o.textContent=String(i):n==="title"?(o.setAttribute("data-tip",String(i)),r=String(i)):o.setAttribute(n,String(i)));if(a)for(let n of a)n==null||n===!1||o.append(typeof n=="string"?document.createTextNode(n):n);return r&&!o.getAttribute("aria-label")&&!o.getAttribute("aria-labelledby")&&!(o.textContent&&o.textContent.trim())&&o.setAttribute("aria-label",r),o}function re(e,...t){e.textContent="";for(let a of t)a==null||a===!1||e.append(typeof a=="string"?document.createTextNode(a):a)}var il="http://www.w3.org/2000/svg";function R(e,t){let a=document.createElementNS(il,"svg"),o=t?.size??18;return a.setAttribute("viewBox",t?.viewBox??"0 0 24 24"),a.setAttribute("width",String(o)),a.setAttribute("height",String(o)),a.setAttribute("fill","none"),a.setAttribute("aria-hidden","true"),a.innerHTML=e,a}var Ta=class e{constructor(){this.items=[],this.disposed=!1,this.detach=null}get isDisposed(){return this.disposed}get size(){return this.items.length}add(t){return this.disposed?(t(),()=>{}):(this.items.push(t),()=>this.removeDisposer(t))}listen(t,a,o,r){t.addEventListener(a,o,r);let n=!0,i=()=>{},l=()=>{n&&(n=!1,t.removeEventListener(a,o,r),i())};return i=this.add(l),l}setTimeout(t,a){let o=window.setTimeout(()=>{this.removeDisposer(r),t()},a),r=()=>window.clearTimeout(o);return this.add(r),o}setInterval(t,a){let o=window.setInterval(t,a);return this.add(()=>window.clearInterval(o)),o}raf(t){let a=window.requestAnimationFrame(r=>{this.removeDisposer(o),t(r)}),o=()=>window.cancelAnimationFrame(a);return this.add(o),a}abortController(){let t=new AbortController,a=()=>t.abort();return this.add(a),t}child(){let t=new e,a=()=>t.flush();return this.add(a),t.detach=()=>this.removeDisposer(a),t}removeDisposer(t){let a=this.items.indexOf(t);a>=0&&this.items.splice(a,1)}flush(){if(!this.disposed)for(this.disposed=!0,this.detach?.(),this.detach=null;this.items.length;){let t=this.items.pop();try{t?.()}catch{}}}};function Ea(e,t,a,o,r=60){let n=0,i=()=>{let l=Math.min(a,n+r),d=document.createDocumentFragment();for(;n<l;n++)d.append(o(n));t.append(d)};if(i(),!(n>=a))if(typeof IntersectionObserver=="function"){let l=c("div",{class:"chunk-sentinel","aria-hidden":"true"});t.append(l);let d=new IntersectionObserver(p=>{p.some(u=>u.isIntersecting)&&(i(),n>=a?(d.disconnect(),l.remove()):t.append(l))});d.observe(l),e.add(()=>d.disconnect())}else{let l=c("button",{class:"chunk-more",type:"button"}),d=()=>{l.textContent=`Show ${Math.min(r,a-n)} more (${a-n} remaining)`};d(),t.append(l),e.listen(l,"click",()=>{l.remove(),i(),n<a&&(d(),t.append(l))})}}function Ft(e){let t=null,a=()=>{t!==null&&window.clearTimeout(t),t=null};return e.add(a),(o,r)=>{t!==null&&window.clearTimeout(t),t=window.setTimeout(()=>{t=null,o()},r)}}var Io="http://www.w3.org/2000/svg",sl=["netcdf","grib","zarr","stac","intake"],ll={netcdf:'<path fill-rule="evenodd" d="M134 96v47H40l2 3 3 2h89l2-3 3-2V54l-2-3-3-2zm113 0v47h-94l2 3 3 2h88l3-2 3-3v-43l-3-50c-2-1-2-1-2 46m228 0v47h-47c-48 0-48 1-46 3 1 1 2 2 47 2h46l2-3 3-2V54l-2-3-3-2zm-113 0v47h-48l-48 1 2 2c2 1 4 2 47 2h44l3-3 3-3V99c0-41 0-49-3-49zM134 209v47H87l-47 1c1 4 5 4 50 4h44l3-3 2-3v-88l-2-3-3-2zm113 0v47h-94l2 3 3 2h88l3-3 3-3v-88l-2-3-3-2zm228 0v47h-48l-47 1c1 4 5 4 50 4h44l3-3 3-3v-88l-2-3-3-2zm-113-45v92h-96l2 3 3 2h43l46-1c5-1 5-3 5-49l-1-46zM134 323v48H87l-42 3 43 1h44l3-3 4-4v-89l-3-2-2-2zm113 0v48h-47l-42 3a509 509 0 0 0 89-1l5-5v-89l-3-2-2-2zm228 0v48h-47c-47 0-47 0-45 2s4 2 46 2c43 0 43 0 47-2l4-5v-89l-3-2-2-2zm-113 0-1 47-47 1c-46 0-46 0-44 2s4 2 46 2l45-2 3-3a1011 1011 0 0 0-2-93zm-79 77-1 9v9h-4q-4 0-4 3-1 4 4 3h4v43h8l-1-15 1-15q5 0 6-5l1-2q2-1 1-3l1-4q3-4-5-6h-5v-17zm46 0-2 1-2 1-6 4-2 1-4 3-6 8-1 3-1 13c0 9 0 11 2 16l5 7 1 2 2 2 3 2 5 3 5 2c1 2 21 2 23 0l5-2c4-1 12-8 12-10l-2-2q-1-3-6 1l-5 3-1 1c0 2-10 3-17 3q-8-1-12-4-10-7-12-15l-1-4v-10l2-5q1-5 7-10 6-6 12-7 15-2 23 4 8 7 10 5 6-3-2-8l-6-5-5-2q-4-2-13-2zm48 34v33h5l6 2h7q1-2 10-2l13-2c6-4 14-14 14-20l1-2 1-8-1-9-1-4q-3-9-13-16l-10-4-7-1-13-1h-12zm65-33-1 33v33h8v-12q0-13 1-11 4 1 9-6 3-4 10-3h6v-7l-13-1h-13v-10c0-9 0-9 2-9q3 0 1 2-3 3-1 6 1 1 5-2l3-5q-2-1 8-1h8v-7l-17-1zm-57 34v26h12c10-1 12-1 17-4q6-3 8-7c3-5 3-6 3-15q1-10-2-13c-1-3-7-9-8-9l-3-2q-2-2-11-3l-7 1c1 1-5 8-7 8q-2-3 1-7 2-2-1-2c-2 0-2 0-2 27m-204-15-5 4-2-2q0-4-4-3h-2l-1 24v24h8v-11l1-12q2 0 3-3l4-6 2-4-2-2-2-2 2-1 8-2c6 0 6 0 9 3q3 3 3 7l1 4 1 15v14h6v-18c0-20 0-21-8-28-4-4-4-4-11-4-6 0-7 0-11 3m54-2-2 1-1 1c-2 0-9 7-9 9l-1 1c-1 0-3 7-3 12 0 6 2 14 3 14l1 1c0 2 6 8 11 10q3 2 11 2 10 0 17-7l6-9-4-2q-3 0-4 2-1 6-9 8l-9 1q-6-1-9-4l-3-3 5-4 4-6c-1-2 0-2 11-2s12 0 11 2q-1 2 4 2 6 0 5-4c0-5-2-13-3-13l-1-1c0-2-7-8-11-10s-14-2-20-1m1 7-5 4c-2 2-4 8-2 8l1-1h4l10 1a36 36 0 0 0 17-2l-1-3q-3-8-16-8zm189 37-3 3 4-2q5-5-1-1"/><path fill="#f9f9f9" fill-rule="evenodd" d="m373 45 1 50v46h100V43l-51-1h-50zM260 206v50h100V156H260zm113-48 1 50v48h100V156H373zM147 320v50h100V269H147zm113 0v50h100V269H260zm114-2v51l50 1h50V269H374z"/><path fill="#ececec" fill-rule="evenodd" d="M260 92v51h100V41H260zm163-50h50v99h-98V94h-2v49h50l51-1 1-51V42zh-51zM165 155l-10 1h-8v8l-1 10-1 37 2 42v3h100V156h-3l-4-1-38-1zm113 0 37 1 37-1-37-1zm113 0 38 1 38-1-38-1zm82 51h2zm-113-48h2l-1-1zm13 50h2zM32 320v50h102V269H32zm328-49 1 1 1-2h-1zm-215 54 1 37 1-37-1-38zm116 45"/><path fill="#036581" fill-rule="evenodd" d="M147 92v51h100V41H147zM32 206v50h102V156H32z"/><path fill="#264961" fill-rule="evenodd" d="m33 42-1 51v50h102V41H84zm327 52h2zm-215 3 1 38 1-37-1-37q-1-1-1 36m-94 58 38 1 37-1-37-1zm309 52h2zm0 114v48h-48l-1 2 50-1 1-50zM36 370l49 1 49-1-50-1zm114 0q-1 1 48 1l1-2zm226 0 50 1-1-2z"/>',grib:'<path fill="#404040" fill-rule="evenodd" d="m88 34-2 3v189H68c-19 0-24 1-26 4l-1 71v69l23 42 22 41v11c0 10 0 11 2 14l3 2h329l3-2c2-3 2-4 2-14v-11l15-27 22-42 8-14v-69l-1-71c-2-3-7-4-26-4h-18v-96l-50-49-49-49H91zm13 13v179h310v-85l-46-1h-45l-2-3c-2-2-2-3-2-47V46H209zm230 45v34h69l-34-34-35-35zM55 241v61l1 60h400V241l-200-1zm67 29q-11 5-17 16-7 14-1 30 6 9 16 15c4 2 6 2 14 2s10 0 15-2c10-6 18-18 18-28q0-5-3-7c-2-2-4-2-17-2-15 0-15 0-18 2q-5 5 0 10c2 2 3 2 13 2h10l-2 3-6 6q-4 3-10 3-16-1-19-16-1-21 21-22 7 1 10-2 5-5 0-9-1-4-10-3zm70 0c-2 2-2 4-2 31 0 28 0 29 2 31q4 4 10 1 2-1 2-10l1-8 12 9c14 11 17 12 21 8q4-4 1-8l-11-9-9-7 4-1 6-2q7-4 10-12 3-15-10-23c-4-2-6-2-20-2s-15 0-17 2m81 0c-3 2-2 8 0 10q2 3 9 2h7v38h-6q-9 0-11 3c-2 2-1 8 1 9 1 2 5 2 23 2 20 0 22 0 23-2q4-5 1-9-2-3-11-3h-6v-38h7q11 0 11-7c0-7-1-7-25-7-20 0-22 0-23 2m81 0c-2 2-2 3-2 29l2 33c1 2 1 2 18 2h18l4-4q8-5 8-14 0-6-2-10l-2-5 2-4c4-7 2-18-4-24-5-4-8-5-24-5-15 0-16 0-18 2m-150 17q-1 7 4 7h9q10 0 9-7 1-5-13-5h-9zm162 1v6h9l11-1 2-3v-4l-2-3-11-1h-9zm0 26v6h9l11-1q5-4 0-9-2-2-11-2h-9zM64 384l12 20 8 15 2 5v-48H60zm36 37 1 45h310v-90H100zm325-21v24l2-5 20-35 4-8h-26z"/><path fill="#e7f7dc" fill-rule="evenodd" d="m56 241-1 61 1 60h400V241l-200-1zm68 28q-11 4-16 11-9 12-7 25 6 28 33 29c7 0 10-1 18-5 9-5 17-22 15-31q-1-4-21-4c-11 0-13 0-16 2q-5 4-2 10c2 2 3 2 13 2q15-1 7 6-10 10-23 4-9-5-10-15 0-9 6-16 6-5 14-4l7-1q6-3 6-8-2-7-12-6zm68 1c-2 2-2 7-2 31 0 32 0 33 6 33q8 1 8-11l1-8 12 10c14 10 17 11 21 7q3-3 1-7l-10-10-10-7 4-1 8-3q4-1 6-7l3-8q0-14-11-19c-4-2-7-2-20-2-15 0-16 0-17 2m81-1-2 4v5q4 5 13 5l5-1v38h-7q-7 0-9 2c-2 2-3 8 0 10 2 3 7 3 27 3l19-2c4-3 2-10-2-12l-8-1h-6v-38l5 1q11 0 13-5t-1-8c-2-2-3-2-24-2zm81 1c-2 2-2 8-2 31l1 30c2 3 3 3 12 4q17 1 24-2c4-1 11-8 12-12 2-4 1-13-1-17q-3-3 0-7 4-8 1-16c-1-5-9-12-13-12l-18-1c-14 0-15 0-16 2m-150 18q-1 5 3 6h9q11 0 10-7 2-4-13-4h-9zm162 0v6l9-1h11l2-4q3-7-13-6h-9zm0 26v6h9c9 0 10 0 12-3q3-2 1-5l-4-3-10-1h-8z"/><path fill="#fff" fill-rule="evenodd" d="M100 47v90l1 89h310v-85l-46-1c-44 0-46-1-48-2l-1-47V46H208zm231 10v69h69zM61 377l3 6 20 37c2 2 2 2 2-21v-23H74zm39 44 1 45h310v-89l-156-1H100zm325-22c0 27-1 25 7 11l18-33-12-1h-13z"/>',zarr:'<defs><linearGradient id="freva-zarr-grad" x1="0" x2="0" y1="0" y2="1"><stop offset="0%" stop-color="#e57c77"/><stop offset="48%" stop-color="#e41073"/><stop offset="100%" stop-color="#bd1083"/></linearGradient></defs><path fill="url(#freva-zarr-grad)" fill-rule="evenodd" d="m247 35-11 5-1 1-2 1-3 3c-2 2-2 3-2 11l1 10 5-2 7-3 4-2 3-2 5-2q2-2 2-12V33h-3zm10 8q0 11 2 11c2 3 17 10 19 10l4 2q5 5-12-3l-12-6-1 13c0 11 0 12 2 14l4 2 11 5 10 4 1-12-1-12v-3l1-12V45l-4-2-4-2-4-2-6-4-7-2h-3zm4-8v9q0 12 4 11l4 2c2 2 12 6 13 5l1-7q1-6-2-9l-5-2-4-2-4-3-4-2zm25 22v12l12 6 13 6 1-11V59l-4-2-5-3-2-1-2-1-1-1-3-1-4-2-2-1q-3-5-3 10m-66-8-4 2-3 1-2 1-2 1-1 1-3 2-4 2-1 11 1 11 4-2 4-2 2-1 1-1 7-3 6-3V59c0-10 0-11-2-11zm68 9q0 10 2 10l2 1 1 1 3 2 3 1 6 2 4 2 1-8q0-10-3-9l-2-2-2-1-6-3-4-3-3-1q-2-1-2 8m-37 1-5 2-3 2-3 1-7 3-3 2c-2 1-2 1-2 14l1 13 1-1 3-1 3-2 4-2 2-1 1-1 7-3 5-2V70l-1-13zm-58 3-2 1-2 1-3 2-3 2-4 2-4 2c-2 1-3 2-3 13l1 11 1-1 11-5 2-1 6-4 5-2V72c0-12 0-12-5-10m68 10c0 9 0 10 2 11l16 8 2 1c2-1 2-2 2-11v-9l-11-6-11-5zm53-1c0 10 0 11 2 13l4 2q5 1 5 4l-5-2-5-1-1 12c0 13 0 14 5 15v2l-3-1-1-1-1 14c0 11 1 13 2 14q21 11 23 10l1-12-1-13-6-4-7-3-3-2-1-1 2-1 1 1 4 3 5 2 2 1 2 1 1 1 1-14V98l-3-2-6-2q-4-1-4-4l11 5h1l1-10c0-10 0-11-2-13l-9-5-8-4-4-2h-3zm2 1 1 10 3 1 2 1 7 4 7 3 1 1 1-9c0-9 0-9-2-9l-4-2-4-2-1-1-1-1-3-2-5-2c-2 0-2 0-2 8m-92 1-3 1-4 2-3 1-4 2-4 2-4 2c-2 2-2 4-2 15v12l12-5q9-6 12-6 2 2 2-13 0-18-2-13m62 13c0 13 0 13 2 13l2 1 2 1 7 3 2 1 3 1 8 4V83l-3-1-4-2-2-1-2-1-2-1-13-5zm56-1 1 13v1l-1 14v14l10 4 11 5 1 1 1 1q3 0 3-17v-8l-7-4-8-5q0-2 6 1 8 6 8 3l1-11c0-11 2-9-13-16l-8-5-3-1c-2-1-2-1-2 10m-177-9-2 1-3 2-3 2-4 2-5 3-3 1-1 10 1 11 6-2 8-3 3-2 5-2 3-1V75h-2zm123 9c0 9 0 10 2 12l5 2 3 2 4 2 3 1 3 2 2 1V97q1-12-2-12l-4-2-3-2-11-5q-3-1-2 9m58-5-1 8v9l7 3 8 3 3 2 2 2 1-10c0-8-1-8-3-10l-8-4-2-1-2-1-2-1zm-154 8-5 3-2 1-2 1-2 1-6 3-3 4v24l3-2 4-2 3-1 8-4 4-2 3-2 1-12-1-13zm59-1-3 2-3 1-4 2-3 2-4 2-4 2c-2 1-2 2-2 14v13l4-2 5-2 2-1 8-5 8-3V99l-1-12zm7 0-1 12c0 12 0 13 2 14q5 2 1 3h-3v25l13 6 13 7c0 2-5 1-6 0-1-2-17-10-19-10l-1 12 1 14 1 2-3 1-12 5-10 4-1 1-1 1-3 2-4 2-9 4-7 3-3 2-4 2-2 1q1 2-8 4l-4 2-9 5-7 4v12c0 11 0 12 2 12l4-1q6-3 7-1l-2 1-6 3-5 2v13q0 13 2 13l2-1 1-1 4-2 4-2 2-1 5-3 5-3 1-5 1-4 1 5q0 6 2 3c1-2 7-5 9-5l2-1 1-1 4-2 4-1c1 1-3 4-4 4l-2 1-1 1-8 3-6 4-1 12 1 12 6-3 2-1 2-1 1-1 3-1 3-2 5-1 2-2v-12l-1-14 5-2-1 1v12c0 12 0 12 3 12l2 1-6 5-10 5-3 2-4 2-3 1c-1 0-22 10-23 12l-4 1-2 1-12 6-12 7-2 1-26 13-5 2-6 2-3 2-4 2-2 1-17 8-3 2-3 1-3 2c-1 1-11 6-16 7l-3 3-3 1-4 2c-1 1-13 7-18 8l-2 2 7 3 3 2c2 2 12 7 15 7l4 2 3 2 4 2 4 2 2 1 9 4 6 3 6 3 9 5 5 2 2 1 2 1 2 1 13 6 3 2 4 2 2 1 15 7 3 2 3 1 1 1 18 8 3 2 3 2 2 1 11 5 2 1 4 2 3 1 1 1 3 2 11 5 4 2 5 2 2 1 1 1 4 2 4 2 3 2 4 1 2 1 2 1 2 1c0 2 8 1 10-1l7-4 5-2 22-11 2-1 2-1c1-1 15-8 18-8l3-2 25-13 19-9 1-1 2-1 2-1 7-3 3-2 7-4 5-2 11-5 5-2 5-3 1-1 2-1 2-1 11-5 5-2 4-3 2-1 15-7 2-1 1-1 3-1 7-5q6-2 7-4l-2-2c-5-1-17-7-18-8l-3-2-4-1-2-2-17-8-4-2-3-2-2-1-17-8-20-9-4-2-4-3-8-4-10-4-2-1-8-5-8-5q0-3 4-1l11 6 10 5v-3l1-12v-9l-4-1-5-3-14-7-1-1-1-1-1 9-1 9-1-10v-9l-12-6-12-7q0-2 5 1 17 9 18 8l1-27-1-28-4-1q-4-1-4-4l7 3 1 1 1-9 1-8 1 9q0 10 2 9l1 1 3 1 5 3 4 3 6 2 4 2v-2l1-13c0-9 0-10-2-11l-3-2-3-2-16-8q-2-3 12 4l11 5 1-13v-13l-7-3q-10-4-10-6 1-3 5 1l4 2 3 2q4 3 4-2l1-13c0-9 0-10-2-11l-2-1-3-1-3-1-15-8-1 14q0 15 2 13l1 1 2 1 2 1-1 1-8-4-2-1-4-2-5-2-3-2q-2-2-7-3l-7-4-6-3-3-2-2-1-3-1-4-3q-6-2-2-3l2 1 1 1 3 1 12 6c2 0 2-1 2-13v-13l-3-1-5-2-12-6-2 1 2 1 3 2 2 2 7 3 6 3v10c0 11 0 11-2 10l-3-2-4-2-5-2-2-1-2-1q-4 1-4-13l1-10v-3l-2 2v53l1 1h-4v-2l1-14c0-12 0-13-2-14l-2-1-2-1-7-4-4-3q-5-1-3-2l3 1 2 1 3 2 3 1 1 1 2 1 2 1 1 1q3 0 3-17 1-9-2-10-6-4-9-4l-3-2-3-2-2-1zm113 3-1 12-1 10 7 3 47 23c2-1 2-2 2-11 0-10 0-10-3-13l-4-2-4-2-15-7-3-2-4-2-2-1-9-5-8-4q-1-2-2 1m-239 2-6 5-2 1-6 3-1 11q0 15 3 10l1-1 3-1 4-2 7-3 2-1q4 1 4-13c0-11 0-11-2-11zm129 9v11l4 1 3 2 2 1 6 2q8 6 7-7 1-10-2-10l-10-5-2-2-7-4zm55-1 1 10 4 2 4 3 2 1 4 2q8 7 7-9 1-10-2-10l-2-1-3-1-4-3-9-4q-2-2-2 10m57 1 1 9 9 5 10 5v-9c0-9 0-10-3-11l-4-1-3-1-3-2-4-2-2-2zm-29-1 2 1q2 1 0 3l-1 10q0 11 2 11c2 2 13 8 16 8l3 1v-9l-1-11-7-3-6-4q0-2-3-2-3-1 0-2 5-1 0-2l-2-1-1-1zm-180 2-3 2-6 4c-12 6-11 4-11 17q0 13 2 13l2-1 2-1 10-6 4-1 4-2c2-2 2-4 2-14 0-12 0-12-2-12zm58 0-3 1-3 1c-1 2-13 8-15 8l-1 13c0 13 0 13 2 13l5-2 4-2 2-1 8-4c4 0 5-2 4-16q1-13-2-12zm-113 3-5 2-3 1-4 3-4 2-2 1q-4-1-4 13c0 9 0 10 2 11l2-1 2-1 2-1 1-1 5-2 5-3 4-1 3-1v-12l-1-12zm292 11 1 9 3 2 4 2 6 3c8 3 8 3 8-7 0-9 0-9-2-9l-4-2-5-2-3-1-4-2-2-1-1-1zm-31-1 1 26q3 3 0 3l-1 13v13l3 1q4 1-1 2l-2 1v12l1 13 2 1 4 2q5 1 7 4l4 2 13 6q0 3-4 0l-4-2-9-5-5-2-2-1-7-3v26l9 4 40 21c6 3 6 3 6-11v-12l-7-4-8-4-6-3-1-1v-2l9 5 10 4c3 1 3 0 3-12s0-13-2-13l-3-2-2-2-17-8v-1l6 2 7 3 1 1 2 1 2 1 2 1 3 2 1-13v-14l-7-3-10-6q1-3 5 1l4 2 3 2 3 1c2 0 2-1 2-12 0-12 0-12-2-15l-5-2-2-1-17-8-3-2-4-2-2-1-7-4-6-3-2-1-4-1zm-235 2-4 2-8 5-5 3-1 13v13l8-4 8-5 4-1c3 0 4-2 4-15v-12h-3zm52 2-7 5-3 1c-4 0-5 2-5 15l1 13 1-1 2-1 11-5 2-1 2-1 3-2 4-1v-12c0-13 0-13-2-13zm54 2c-14 7-13 6-13 20q0 16 3 11l2-1 4-1 9-5 2-1 4-2h3v-13c0-13 0-13-2-13zm186 7v12l4 3 6 3 3 1 4 2 4 2 3 1q1 3 1-11l-1-12-2-1-2-1-2-1-2-1-3-2-2-2-6-2-5-3zm-345-9-3 1-4 2-3 1-1 1-2 1-2 1-3 2-4 1v24l4-2 5-3 2-1 5-2 3-2 4-2c2-2 2-3 2-12q1-12-2-11zm234 10v11l3 1 18 8 1-10v-10l-11-6-11-5zm57 0 1 11 10 4 9 4v-10c0-10 0-11-3-12l-4-1-2-1-11-6zm-112 1 1 10 11 5 3 2 4 2q3 2 3-9 1-10-2-10l-5-4-6-3-6-2-3-1zm168-1c0 8 0 8 2 11l5 2 2 1 1 1 5 2 5 2v-18l-2-1-5-2-12-6zm-323 3-18 10-1 13 1 12 3-1 21-10 1-13-1-14zm60-2-4 2-6 4-5 2-5 3c-2 2-2 3-2 14l1 12 18-9 4-2 3-1v-12l-1-13zm52 3q-6 2-7 4l-3 1-4 2-3 2-1 12 1 13 12-5 13-6v-14l-1-13zm236 11c0 11 0 12 2 12l7 3 2 1 2 1 7 3 3 2 2 1 1-12c0-12 0-13-2-13l-12-6-11-5zM53 133l-4 2-3 1-1 1-2 1-2 1-4 3-4 2-1 11 1 11 10-5 3-1 3-1 5-3 4-2v-10l-1-11zm348 10q0 12 2 12l7 3 4 2 5 2q1 3 3 1l1-10c0-9-1-10-3-11l-4-2-4-2-3-2-4-2-3-1q-1-2-1 10m-113 1 1 11h3l4-2 4-2 4-2 4-2 2-3-1-1h-2l-1-1-3-2-14-7zm57 0c0 9 0 10 2 11l6 3 5 2 2 1 5 2 1-10c0-9-1-10-3-11l-4-2-13-7zm113 0c0 9 0 10 2 11l3 1 13 7 1 1 1-9q0-12-3-10l-2-2-2-1-3-2-3-2-5-2-2-2zm-83 0 4 2 5 4 3 1 3 1c0 1 8 6 10 6l3 1h3l-4-2-27-14zm-293 1-3 1-4 1-4 3-4 2-1 1-4 2c-2 2-2 3-2 14s1 11 3 11l3-2 4-1 4-2 4-2 2-1 2-2c2 0 3-1 3-13l-1-13zm57 0-3 1-4 2-3 1-1 1-2 1-2 1-3 2-4 1v14q0 17 3 12 3-3 7-4l4-2 3-2 4-2c3-2 3-3 3-15q0-16-2-11m53 1-6 3-7 5-6 3-1 12c0 12 0 13 2 13l2-1 8-4 4-2 3-2 5-2c2-2 2-4 2-15l-1-12zm59-1-3 1-4 3-13 6-3 1v14c0 12 0 12 3 12l2-1 4-3 6-3 4-2 5-2 3-2v-11l-1-13zm176 11v42l6 3 7 3 2 2 4 2 3 2 2 2 1-14-1-15-5-2-8-4-5-2-1-1-2-1q-3-1-1-2l3 1 2 1 4 2 7 4 5 2 1 1 1-26-3-2-4-2-3-2-6-3-6-2-3-1zm-166-8v21l3-1 6-2 5-2 3-2 3-2q3 0 1-1l-5-3-6-3-3-2-2-1zm47 2-7 4-5 3a619 619 0 0 0-66 33c-2 0-14 6-15 8l-4 1-2 1-2 1-3 1-3 1-6 4q-9 5-4 5l14 6c1 2 7 1 9-1l3-2 14-6 2-1 2-1 3-2 32-15 9-4 2-1 1-1 3-2 5-3 4-1 3-1 2-1 8-4 5-3 2-1 4-2 5-2 3-2 3-2 5-1 3-2-3-1-4-1-1-1-14-7zm66-2-1 10c0 10 0 11 3 12l4 1 7 4 4 2c2 0 2 0 2-10v-10l-3-1-5-3-10-6zm55 9v10l20 10 1-10v-10l-7-3-9-6-3-1q-2-2-2 10m-378 2-4 3-3 2-4 2-3 2h-3c-2 1-2 2-2 14s0 13 2 13l2-1 3-2 11-5 8-3v-13c0-12-1-13-2-13zm56 1-5 3-1 1-4 2-4 2-4 2c-2 3-2 4-2 14q0 12 2 11l8-3 1-1 14-7 1-13-1-13zm60-1-3 1-3 1-9 6-8 3v27l8-4 4-3 3-1 7-3c3 0 4-2 4-14l-1-13zm57 0-3 1-3 1-9 6-9 4v13l1 13 1-1 2-1 4-1c1-2 11-7 15-8 2 0 2-1 2-14zm231 0-1 13 1 12v2l-1 14 1 13q4 1 1 3c-2 0-2 1-2 12l1 13 23 12 1 1 1-13v-14l-3-1-4-2-2-1-3-2-2-2-4-1q-3-1-1-2h2l2 1 3 2 11 6 1-13c0-12 0-13-2-14q-4-3 0-3c2 0 2-1 2-13v-12l-4-2-5-3-15-7zm-124 3-6 4-4 3-11 4-9 4-2 1-2 2-3 1-9 5-32 15-2 1-4 2-16 8-2 1-5 2-7 4-4 2c-3 0-4 2-4 14l1 12 7-3 18-8 3-2 3-2 4-2 3-2 4-1 18-9 11-5 10-5 4-2 2-1 9-4 3-1h3l-2 3-2 3-2 4-3 4-2 2-3 6-5 6-3 4-1 2-3 4-4 7-4 6-4 5-2 4-2 3-2 2-3 4-3 4-1 2-2 3-1 2-1 1-3 4-6 10-2 3-2 2-2 3-2 4-1 1-1 2-1 1-2 2-3 4-3 5-2 4-3 4-2 3-2 2-3 5-4 6-2 3c-2 1-2 3-2 19l1 17 1-1 11-5 3-1 3-2 1-1 2-1 2-1 7-3 3-1 6-3 4-2 2-1 2-1 1-1 4-2 3-2 4-1 6-3 4-2 29-14 4-2 2-2 9-4 2-1 2-1 7-3c3 0 4-2 4-16v-13l-4 1-4 2-3 2-4 2-9 4-4 3-3 1-2 1-9 4-9 5-2 1-2 1-2 1-2 1-26 13-11 5-4 1q-6 1 1-5l3-5 2-4 2-3 1-2 3-3 4-6 4-6 4-5 2-3 3-4 4-7 4-5 2-3 2-3 5-8 2-3 2-2 2-3 2-3 3-4 3-4q0-3 3-6l2-3 3-3 3-4 5-9 2-3 2-2 2-3 3-4 2-3 2-3 2-3 2-3 2-2 3-4c2-2 2-2 2-20l-1-17zm70 9 1 11 10 6 10 5 1-10v-11l-6-3-7-4-2-1-4-2-2-2zm-56 2c0 9 0 10 2 11l2 1 2 1 2 1 1 1 6 3 6 2v-21l-7-3-6-4-2-1-3-1q-3-5-3 10m113-1c0 9 0 10 2 12l4 2 11 5q3 4 3-10v-10l-9-4-10-5zm-378 2-2 1-3 2-5 3q-5 0-7 4c-3 2-3 3-3 14l1 12 11-6 8-3q6-3 6-5l-1-12v-11h-2zm57 0-8 5-10 4c-2 1-2 3-2 14l1 13 7-3 6-4 2-1 3-1c4 0 5-2 5-16 0-11 0-12-2-12zm54 2-10 5q-4 1-7 4c-2 2-2 3-2 14l1 12h1l6-3 14-8 2-1c2 0 2-1 2-13l-1-13zm186-2q1 3 11 6l7 4q3 4 4 1l-3-2-3-2-1-1-3-1-3-2-4-2q-4-3-5-1m52 2v10q0 14 4 13l8 4 8 3 1-10-1-10-3-2-4-2-2-1zm-56 12c0 10 0 11 2 11l4 1 13 6 1-10-1-10-3-2-7-3-5-3-2-1c-2 0-2 1-2 11m-262-1-4 2-4 3-3 1-4 2-4 2-1 1-2 1c-2 0-2 1-2 12l1 13 4-1 6-2 4-3 5-3 3-1 3-1v-14l-1-13zm347 0 2 1 2 1 2 1 4 2 5 3q3 1 1-2l-3-2-6-3q-8-3-7-1m-406 2-2 1-4 2-3 1-4 2-4 2c-2 2-3 2-3 15l1 13 4-2 3-2 4-2 3-2 4-2 5-3c1-1 2-2 2-12v-12h-3zm111 0-3 3-3 1-4 2-5 3-4 2v12l1 12c1 1 14-6 16-8l4-1 2-1 2-1 1-12c0-13 0-13-2-13zm238 12 1 11 2 1 15 7 2 1q3-1 2-10 0-13-4-12l-9-4-4-3-4-2zm57 0c0 11 0 11 3 12l2 1 2 1 5 2 4 2 2 1 1 1 1-10v-10l-6-3-7-4-6-4zm-113-9v11l1 8 8 4 8 5 2 1q3-1 2-10 0-13-5-12l-3-2-6-3-5-3zm-9 8-1 3q-4 3 1 3 3 1 2-3l-1-4zm-252 2-3 1-4 1-4 3-4 2-2 1-2 1-2 2c-2 0-3 1-3 13v13l2-1 5-2 5-2 4-3 6-3h3v-27zm53 1-5 3-2 1-2 1-2 1-4 3-4 2-1 12v13l2-1 3-1 6-3 7-5 4-1h2v-13l-1-14zm290 10v71l5 3 7 4 10 5 1 1 1 1 1-28-1-29-15-7-4-2-3-2v-2l9 5 12 5 2 1v-25l-3-2-4-2-3-2-2-2-4-1-6-3-3-1zm-124-8-4 2-5 3-4 2-2 1-4 2-6 2-6 4c-7 3-8 5-11 11l-2 3-2 2-3 4-3 5-5 7-2 3-2 3q-3 2-3 5l-3 5q-6 6-10 14l-2 2-2 2-1 2-1 2-2 3-2 3-2 3-4 5-4 7q-3 2-5 7l-4 6-3 3-1 2-1 2-3 4-4 6 4 1 5 2 2 1 6 2q6 4 5 1l3-4 6-10 2-3 2-2 3-4 3-4 1-3 3-4 5-6 4-7 3-5 2-3 1-1 1-2 4-5 3-4 2-3 3-5 4-5 2-3 2-3 2-3 2-3 2-4 3-3 2-3 2-3 2-2 3-5 2-4 3-5 9-12 3-5 2-3 1-2 1-1-1-1zm70 10v11l6 2 5 3 2 1 3 2 3 2 1-10c0-9 0-10-2-12l-4-2-3-2-5-2-3-1-2-2zm56 0 1 11 2 1 2 1 5 2 3 2 4 2 3 2 1-10v-11l-8-4-10-5-1-1q-3-1-2 10m-103 0-3 6-3 6 2 1 4 2 5 3q6 3 5 5l-11-5-4-2-3-2-1-1-1 1c1 1 3 3 15 9q9 5 6 2v-3l1-11c0-10-1-10-3-12q-7-4-9 1m-139 10c0 11 0 12 2 14l5 2 2 1 10 5 1-12c0-12-1-13-3-14l-4-2c-1-2-10-5-12-5zm-132-9-4 1-4 3-5 3-4 2-4 2c-2 2-2 3-2 14l1 13 21-12 4-1v-13l-1-12zm53 2-3 1-1 1-2 1-3 2-3 2-5 2-3 2-1 12q0 13 2 13l2-1 4-2 5-3 7-4 6-3v-12c0-12 0-13-2-13zm58 0-3 1-2 1-8 5-8 4-1 12 1 13 4-1 4-3 4-2 3-1 1-1 2-1 2-1 2-2c2 0 3-1 3-13l-1-13zm176 11 1 12 4 2 4 3 2 1 2 1 12 6 1-14v-12l-3-2-4-1-8-4-10-5zm4-9-1 11q0 12 2 10l2 1 4 2 7 4q7 7 6-9c0-10-1-11-2-11q-7-1-12-5zm55 10v11h3l2 1 2 1 2 1 1 1 5 3 5 2q3 2 2-10 1-12-2-11c-1 0-11-5-12-7l-3-1-4-1q-1-2-1 10m57 0c0 11 0 11 2 11l4 1c1 2 12 8 14 8v-21l-9-4-10-6zm-204-1 2 5 2-5-2-1zm-172 2-2 1-2 1-1 1-3 2-4 2-4 1-3 2c-2 2-3 2-3 15v13l7-4 7-4 2-1 4-2 3-2c2-1 2-1 2-13q1-15-2-13zm57 0-4 2-5 3-4 2-3 2-4 1h-2v14l1 14 1-1 6-3 6-4 3-1 1-1 3-2 3-1v-13l-1-13zm111 0q-1 1 1 1l1-1-1-1zm120 12 1 13 3 2 4 2 3 1 6 4 5 3h-2l-3-1-4-2c-2-2-10-5-12-5l-1 12 1 13 18 10q6 2 3 3l-7-3-15-7 1 24 7 3 3 1c1 2 13 8 15 8l4 2 3 2 4 2 7 4 5 2 3 1c2 0 3-1 3-11l-1-12-3-1-11-6-2-1-4-2-5-2q-5-1-3-3c1-1 12 4 17 8l4 2 2 1 3 1h3v-25l-4-1-5-3-3-2-4-2-4-2-3-1-3-2-2-1-2-1q-3-1-1-2l3 1 1 1 19 9 5 2 3 1v-14c0-12 0-13-2-13l-2-1-2-1-3-1-17-8-3-2-7-4-8-5-6-2-2-1q-3-5-3 12m-129-8v2h1l2-1 2-1q-2-2-5 0m71 23-1 24-13-7-6-3h-5l-1 1 23 11c4 0 5-2 5-14l1-12v-2l-1-11-1-11zm4-13c0 11 0 11 2 11l6 2 13 7 1-11c0-10 0-11-2-11l-7-2-6-3-2-2-4-2zm57 0c0 10 0 11 3 12l4 1 5 3 5 2 1 1 2 1v-22l-3-1-12-6q-5-7-5 9m56-1 1 11 3 1 1 1 2 1 2 1 1 1 6 2 5 3v-11c0-10 0-11-3-12l-4-1-2-1-11-6zm-195-5-3 2c1 1 7-1 7-3q0-2-4 1m74 2-3 4-2 3 3 2q5 3 4-7-1-6-2-2m146 15v14l4 2 6 2 1 1 1 1 3 1 6 4 5 3v-14c0-14-1-16-5-16l-2-1-7-4-6-3-2-1-2-1-1-1zm-403-9-11 6-8 4v25h2l4-1c1-2 17-10 19-10l1-13c0-13 0-13-2-13zm57 0-15 7-4 3c-2 2-2 3-2 14v12l3-1 6-2 5-3 4-3 3-1 3-1c2-2 2-3 2-14 0-14 0-14-5-11m55 0-5 4-4 2-4 1-2 1-2 1c-2 1-2 2-2 14l1 13 22-11 3-1v-13c0-13 0-13-2-13zm179 11v13l8 4 10 5 5 2 3 1v-13c0-13 0-13-2-14l-2-1-2-1-11-6-7-3c-2 0-2 1-2 13m3 2c0 10 1 11 3 11l4 1 5 3 3 2 5 2v-21l-5-2-4-2-10-5zm56-1 1 10 4 3 5 3 3 1 5 2 3 2 1-11c0-10 0-11-3-12l-4-1-3-2-11-5zm57 0 1 10 5 3 4 3 4 1 3 1 2 1 1-10q0-14-4-12l-7-3-8-4zm-157-8 4 3 2 1 2 1q3-1-2-3l-4-2-1-1zm-4 6-1 3-2 3-2 3-1 1c-2 1 2 4 4 4l2 1 12 6 1-9q1-14-8-13zm-215 3-9 5-6 3-1 1-4 2c-2 2-2 4-2 14q0 16 3 11l2-1 4-1 9-6 7-3v-27zm347-1 6 4 10 5h3q1-1-5-3l-5-4-2-1-3-1zm-293 3-3 1-4 2-8 4c-3 0-4 2-4 16l1 13 2-1 3-2 3-2 8-4 7-4v-12c0-12 0-13-2-13zm55 0-5 3-3 2-6 2-5 3v12l1 12 12-6 2-1 3-2q1-2 3-1l3-2q3 0 2-13 2-16-7-9m125 2v9q0 8 2 8l9 4 5 2 5 2 1-8v-7l-5-3-6-2-3-2-4-3zm58 0-1 10v11l5 1 4 2 2 1 2 1 6 3 1-10c0-10-1-11-3-11l-2-2-2-1-4-2zm55 9v11l3 1 5 2 3 2 4 2 4 2q2 2 2-9l-1-11-2-2-2-1-2-1-4-1c-1-2-7-5-9-5zm25 12c0 12 0 13 3 15l4 2 4 2q2 3 7 4l4 2 2 1c2 0 2-1 2-13v-12l-4-2-5-3-1-1-4-2-4-2-3-1-3-2c-2 0-2 1-2 12m-404-9-5 3-2 1-2 1-4 2-5 2v13c0 12 0 13 2 13l2-1 1-1 10-5 6-3 5-3v-11c0-14-1-14-8-11m60-1-4 1-2 1q-1 2-9 6l-8 4v24h3l3-1 3-2 5-2 7-4 5-2v-13l-1-12zm58 0-3 1-4 2-9 5-8 4v11c0 10 0 11 2 11a74 74 0 0 0 22-12c1-1 2-2 2-12l-1-11zm54 3-4 3-11 5-9 4-5 3-8 4-5 2-3 1-9 5-9 5-3 1-2 1-2 1-2 1-22 11-7 3-3 2-7 4-5 2-11 5-2 1-7 4-6 3-11 5-2 1-2 1-3 2-3 1-1 1-2 1-2 1-2 1-4 2-5 3q-5 1-2 2l9 4 12 6 3 2 2 1 9 5 4 1 3 2 3 1 4 2 3 2 17 8 2 1 4 2 3 2a589 589 0 0 1 73 35l3 2 5 3 31 15 2 1 2 1 11 5 2 1 3 2 2 1 4 1q1 1 8-2l9-4 3-2 2-2 8-3 34-17 7-4 3-2 7-2 16-9 3-2 23-10 17-9 26-13 3-2 26-12 18-10q4-2 1-3l-13-6-2-1-2-1-4-2-4-3-3-1-11-5-6-3-7-4-2-1-11-5-5-2-7-4-3-2-7-3-10-5-6-3-4-2-2-1-2-1-2-1-1-1-8-3-7-4v10q1 12-2 10l-2 1-7 3-3 2-5 3-3 1-2 1-2 1-10 5-9 4-2 1-4 2-12 6-2 1-4 2-3 1-1 1-3 2-4 2-4 2-4 2-27 13-20 9-7-3-2-1-2-1-2-1q-3 0-7-3l-5-3-1-18 1-18 1-2 2-2 2-2 2-4 2-3 3-4 2-4 2-2 2-3 5-8 2-3 3-3 6-10 2-3 2-2 3-5q6-7-1-3m61 0-2 4-3 4q-3 2-5 7l-3 5-2 3-2 2-2 2q1 3 4-1l15-7 4-2 4-2 3-2 5-3 4-2-17-10zm118 9v11l4 1 3 2 1 1 4 2 4 2 3 2 2 1 1-10-1-12-6-2-4-2-1-1-3-1-4-3-3-2zm57 0 1 11 4 2 3 2 1 1 2 1 2 1 2 1 4 1q1 2 1-9v-11l-9-4-9-6-1-1zm-113-8v10l1 7 9 5 10 4v-18l-6-2-4-2-2-1-3-2q-3-3-5-1m-31 6-4 2-3 1-2 1-2 1-22 11-8 4-9 5-3 1-2 2-1 2-1 1-3 4-6 10-2 3-2 3-1 1-1 2 2-1 8-4 8-5 4-1 2-1 2-1 2-1 3-2 13-6 4-2 2-1 7-3 15-8 22-11-7-3-3-1c-1-2-9-4-11-4zm-231 3-3 2-4 2-5 3-3 2-5 2-3 1v13l1 13 6-3 1-1q5 0 8-4l5-3 4-2v-25zm344 11 2 16 3 1 12 7 7 3 1-13v-13l-8-4-10-5-3-2-3-1q-1-3-1 11m-292-8-5 3-4 2-2 1-4 2-3 1v12l1 12 1-1 1-1 10-5 9-4c2 0 2-1 2-12 0-12 0-12-2-12zm294 1 1 21 2 1 2 1 4 2 5 3 4 1 2 1 1-10c0-11 0-12-8-14l-4-2-2-1-2-1zm-56 10v9l10 4 10 5v-20l-4-1-6-2-4-2-2-1-2-1q-2-1-2 9m-321 2-8 5-5 2-5 2c-2 3-2 4-2 14 0 11 0 12 2 12l19-9 3-2c2 0 2-1 2-13l-1-13zm59-1-5 3-7 4-4 1-2 1-3 2-3 1v11c0 10 1 11 3 11l3-1 9-4 2-2 5-3 4-1v-12l-1-12zm344 0-1 13v12l3 1 5 2 1 1 1 1 4 2 4 2q4 1 4 4l-7-3-2-1-2-1-3-2-2-1-3-1h-3v11c0 9 0 10 2 11l2 1 4 2 10 6 8 4v-12l-1-13v-1l1-14c0-13 0-14-2-14l-2-1-2-1-5-2-7-4-4-2zm-54 12c0 9 1 9 4 11l4 2 4 2 4 2 3 2 2 1 1-10q1-12-2-10l-9-3-8-4-1-1-1-1zm57 2c0 10 0 11 2 11l6 3 4 2 4 2 3 2 1-11v-10l-5-3-6-2-4-2-4-3zm-374 1-3 1-3 1-9 4-2 1-1 1-4 2q-3 1-2 13l1 11 7-4 4-2 2-1 9-4c2 0 2-1 2-12zm343 9v12q-1 2 2 2l3 1 2 2 5 2 4 2 2 1 6 3 1-12v-11l-5-3-7-3-12-6zm2 2v9l2 2 5 1 2 2 3 1 5 2q4 7 4-7l-1-9-2-1-11-5-2-1-3-2q-2-3-2 8m-376 4-4 2-2 1-2 1-2 1-7 3q-4-2-4 13l1 11 3-1 3-1 3-2 3-2 2-1 1-1 6-2 4-2v-11l-1-11zm405 11q0 12 3 10l3 2 4 1 3 1 4 3 3 2v-9q0-12-3-10l-2-2-2-1-5-2-4-2-3-1q-1-3-1 8m-274 3-1 14c0 13 0 14 2 15l2 1 5 2 2 1 2 1 5 2 4 2v-32l-3-2-4-1-5-2q-7-3-9-1"/>',stac:'<path fill="#09b3ad" fill-rule="evenodd" d="m110 110-3 3v239l3 2q3 3 7 3 6 1 8-5c2-3 2-17 2-114V127h224l3-3q3-3 3-7t-3-7l-3-3H113zm201 123a97 97 0 0 0-62 41c-6 8-12 21-14 31-2 6-2 10-2 25 0 16 0 18 2 26q8 27 27 44a98 98 0 0 0 129 6q23-20 33-50l3-21a99 99 0 0 0-37-82c-9-7-25-15-35-18-11-3-34-4-44-2m9 17q-15 2-29 9l-20 15-13 19 5 1q8 0 17 10c5 5 8 11 12 26l7 16q6 6 9 0l4-17c4-22 9-31 19-36 5-3 6-3 13-3l13 2q6 1 9-3 6-5 10-17l1-6-3-3c-4-3-17-9-24-11q-14-3-30-2m70 34c-1 5-8 15-12 18q-11 10-29 7-7-3-10-1c-4 3-7 9-10 26l-5 18q-4 10-12 14c-11 4-24-3-31-17-2-2-4-10-6-16q-6-17-11-20-3-3-5 0l-5 8q-9 17 4 45a78 78 0 0 0 70 43l17-1q7-1 6-3a49 49 0 0 1 10-53q8-9 22-12h24q2 1 2-10a73 73 0 0 0-17-50zm-7 74q-17 5-18 25c0 6 1 17 3 17l18-14c9-9 19-25 17-28z"/><path fill="#144e63" fill-rule="evenodd" d="m182 182-3 3v238l3 3q5 6 11 3c6-4 6 5 6-118V199h224l3-3q3-3 3-7t-3-7l-3-3H185zm281 51-2 2-1 112v113H347l-115 3-2 7q-1 5 3 7l3 3h118l121-1c5-3 5 3 5-124 0-102 0-117-2-120q-1-6-9-5-4 0-6 3"/><path fill="#c4e2ef" fill-rule="evenodd" d="m35 35-3 4v237l3 3q5 6 12 3l4-3 1-114 1-112 112-1h113l2-3c2-2 3-9 1-12l-4-3c-3-2-18-2-121-2H39z"/>',intake:'<path fill-rule="evenodd" d="M166 100v48H63l96 97 97 96 97-96 96-97H345V51H166zM58 207l-26 28 112 114 112 111 112-111 112-113-27-28-27-27-16 15-15 17 11 12 10 11-80 80-80 80-81-80-80-81 11-11 11-12c0-1-30-31-32-31z"/>'},cl=sl.map(e=>`<symbol id="freva-brand-${e}" viewBox="0 0 512 512">${ll[e]}</symbol>`).join("");function hn(){let e=document.createElementNS(Io,"svg");return e.setAttribute("aria-hidden","true"),e.setAttribute("width","0"),e.setAttribute("height","0"),e.setAttribute("style","position:absolute;width:0;height:0;overflow:hidden"),e.innerHTML=cl,e}function qe(e,t){let a=t?.size??16,o=document.createElementNS(Io,"svg");o.setAttribute("viewBox","0 0 512 512"),o.setAttribute("width",String(a)),o.setAttribute("height",String(a)),o.setAttribute("aria-hidden","true");let r=document.createElementNS(Io,"use");return r.setAttribute("href",`#freva-brand-${e}`),o.appendChild(r),t?.chip===!1?o:c("span",{class:"brand-chip"},[o])}var mn="data:image/webp;base64,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";var z='stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"',E={search:`<circle cx="11" cy="11" r="7" ${z}/><path d="M21 21l-4.3-4.3" ${z}/>`,chevron:`<path d="M9 6l6 6-6 6" ${z}/>`,chevronDown:`<path d="M6 9l6 6 6-6" ${z}/>`,close:`<path d="M6 6l12 12M18 6L6 18" ${z}/>`,check:`<path d="M5 12l5 5 9-11" ${z}/>`,sun:`<circle cx="12" cy="12" r="4" ${z}/><path d="M12 2v2M12 20v2M2 12h2M20 12h2M5 5l1.5 1.5M17.5 17.5L19 19M19 5l-1.5 1.5M6.5 17.5L5 19" ${z}/>`,moon:`<path d="M21 12.8A8.5 8.5 0 1111.2 3a6.5 6.5 0 009.8 9.8z" ${z}/>`,notes:`<path d="M4 5h16M4 12h16M4 19h10" ${z}/>`,info:`<circle cx="12" cy="12" r="9" ${z}/><path d="M12 11v5M12 8h.01" ${z}/>`,resultsFocus:`<rect x="3" y="4" width="7" height="16" rx="1.5" ${z}/><rect x="12" y="4" width="9" height="16" rx="1.5" ${z}/>`,overview:`<path d="M4 20h16" ${z}/><rect x="5" y="10.5" width="3.4" height="6.5" rx="1" ${z}/><rect x="10.3" y="5.5" width="3.4" height="11.5" rx="1" ${z}/><rect x="15.6" y="13.5" width="3.4" height="3.5" rx="1" ${z}/>`,list:`<path d="M8 6h13M8 12h13M8 18h13M3.5 6h.01M3.5 12h.01M3.5 18h.01" ${z}/>`,grid:`<rect x="3" y="3" width="8" height="8" rx="1.5" ${z}/><rect x="13" y="3" width="8" height="8" rx="1.5" ${z}/><rect x="3" y="13" width="8" height="8" rx="1.5" ${z}/><rect x="13" y="13" width="8" height="8" rx="1.5" ${z}/>`,command:`<path d="M9 6a3 3 0 10-3 3h12a3 3 0 10-3-3v12a3 3 0 103-3H6a3 3 0 10-3 3" ${z}/>`,inspect:`<path d="M6 18h8" ${z}/><path d="M3 22h18" ${z}/><path d="M14 22a7 7 0 100-14h-1" ${z}/><path d="M9 14h2" ${z}/><path d="M8 6h6v4a2 2 0 01-2 2h-2a2 2 0 01-2-2z" ${z}/><path d="M12 6V3a1 1 0 00-1-1H9a1 1 0 00-1 1v3" ${z}/>`,download:`<path d="M12 4v11M7 11l5 5 5-5M5 20h14" ${z}/>`,caret:`<path d="M6 9l6 6 6-6" ${z}/>`,copy:`<rect x="9" y="9" width="11" height="11" rx="2" ${z}/><path d="M5 15V5a2 2 0 012-2h10" ${z}/>`,kebab:'<circle cx="12" cy="5" r="1.6" fill="currentColor"/><circle cx="12" cy="12" r="1.6" fill="currentColor"/><circle cx="12" cy="19" r="1.6" fill="currentColor"/>',clock:`<circle cx="12" cy="12" r="9" ${z}/><path d="M12 7v5l3.5 2" ${z}/>`,box:`<rect x="3" y="6" width="18" height="12" rx="1.5" ${z}/>`,expandWide:`<path d="M3 12h18M7 8l-4 4 4 4M17 8l4 4-4 4" ${z}/>`,expandTall:`<path d="M12 3v18M8 7L12 3l4 4M8 17l4 4 4-4" ${z}/>`,plus:`<path d="M12 5v14M5 12h14" ${z}/>`,x:`<path d="M6 6l12 12M18 6L6 18" ${z}/>`,retry:`<path d="M21 12a9 9 0 11-3-6.7M21 4v4h-4" ${z}/>`,uris:`<path d="M6 3h9l4 4v14H6z" ${z}/><path d="M15 3v4h4" ${z}/><circle cx="9.3" cy="9" r=".95" fill="currentColor"/><path d="M11 9h5" ${z}/><circle cx="9.3" cy="12.5" r=".95" fill="currentColor"/><path d="M11 12.5h5" ${z}/><circle cx="9.3" cy="16" r=".95" fill="currentColor"/><path d="M11 16h5" ${z}/>`,aggregate:`<rect x="3" y="4.5" width="6.5" height="5" rx="1.3" ${z}/><rect x="3" y="14.5" width="6.5" height="5" rx="1.3" ${z}/><path d="M9.5 7h3.5a2 2 0 0 1 2 2v1M9.5 17h3.5a2 2 0 0 0 2-2v-1" ${z}/><rect x="15" y="8.5" width="6" height="7" rx="1.5" ${z}/>`,shelve:`<rect x="3" y="5" width="18" height="3.4" rx="1.6" ${z}/><rect x="3" y="10.3" width="18" height="3.4" rx="1.6" ${z}/><rect x="3" y="15.6" width="18" height="3.4" rx="1.6" ${z}/>`,gear:`<circle cx="12" cy="12" r="3.2" ${z}/><path d="M12 2v3M12 19v3M4.2 4.2l2.1 2.1M17.7 17.7l2.1 2.1M2 12h3M19 12h3M4.2 19.8l2.1-2.1M17.7 6.3l2.1-2.1" ${z}/>`,reset:`<rect x="4" y="4" width="16" height="16" rx="2" ${z}/><path d="M9 9l-3-3M9 9V6M9 9H6M15 15l3 3M15 15v3M15 15h3" ${z}/>`,minimize:`<path d="M6 16h12" ${z}/><path d="M12 12l-3-3M12 12l3-3" ${z}/>`,sortCount:`<path d="M4 7h10M4 12h7M4 17h4" ${z}/><path d="M17 5v14M17 19l3-3M17 19l-3-3" ${z}/>`,sortAlpha:`<path d="M4 7h8M4 12h6M4 17h4" ${z}/><path d="M16 8l2-3 2 3M16.5 7h3M16 19l2-3 2 3M16.5 18h3" ${z}/>`,terminal:`<rect x="2.5" y="4" width="19" height="16" rx="2.5" ${z}/><path d="M7 9.5l2.8 2.5L7 14.5M12.8 15h4.4" ${z}/>`,bashTab:`<path d="M4 7l4 4-4 4M11 16h8" ${z}/>`,help:`<circle cx="12" cy="12" r="9" ${z}/><path d="M9.6 9.3a2.5 2.5 0 114 2.1c-.9.6-1.6 1-1.6 2.1M12 17h.01" ${z}/>`,pySnake:'<path fill="currentColor" d="M11.9 2c-1.6 0-3 .14-4 .5C6.6 3 6.2 3.9 6.2 5.2v1.9h5.9v.8H4.3c-1.4 0-2.6.8-3 2.4-.4 1.8-.4 2.9 0 4.8.3 1.4 1.1 2.4 2.5 2.4h1.6v-2.2c0-1.6 1.4-3 3-3h5.4c1.3 0 2.4-1.1 2.4-2.4V5.2c0-1.3-1.1-2.3-2.4-2.5-.8-.14-1.7-.2-2.5-.2zM9.2 4.2c.5 0 .9.4.9.9s-.4.9-.9.9-.9-.4-.9-.9.4-.9.9-.9z"/><path fill="currentColor" d="M18.2 7.1v2.2c0 1.7-1.4 3-3 3H9.8c-1.3 0-2.4 1.1-2.4 2.4v3.5c0 1.3 1.1 2.1 2.4 2.5 1.5.4 3 .5 4.8 0 1.2-.3 2.4-1 2.4-2.5v-1.4h-5.9v-.8h8.8c1.4 0 1.9-1 2.4-2.4.5-1.5.5-2.9 0-4.8-.3-1.4-1-2.4-2.4-2.4h-1.7zm-3.3 12c.5 0 .9.4.9.9s-.4.9-.9.9-.9-.4-.9-.9.4-.9.9-.9z"/>'};var gn={js:"https://unpkg.com/leaflet@1.9.4/dist/leaflet.js",css:"https://unpkg.com/leaflet@1.9.4/dist/leaflet.css",tileUrl:"https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",attribution:"\xA9 OpenStreetMap contributors"},Da=null,yt=null,zt=null,Lo=null,Bo="freva-leaflet-css",dl=null,An=null;function pl(){let e=document.getElementById(Bo);return!!e&&e.isConnected&&e.dataset.loaded==="true"}function ul(e,t=8e3){return new Promise((a,o)=>{let r=document.createElement("link");r.id=Bo,r.rel="stylesheet",r.href=e;let n=!1,i=d=>{n||(n=!0,window.clearTimeout(l),d?(r.dataset.loaded="true",a()):(r.remove(),o(new Error("Leaflet stylesheet failed to load"))))},l=window.setTimeout(()=>i(!1),t);r.onload=()=>i(!0),r.onerror=()=>i(!1),document.head.appendChild(r)})}var $t=!1;function fl(e){return pl()?Promise.resolve():(yt&&$t||($t=!0,yt=(An??ul)(e).then(()=>{$t=!1}).catch(t=>{throw $t=!1,yt=null,t})),yt)}function hl(e,t=8e3){return new Promise((a,o)=>{let r=window;if(r.L){a(r.L);return}let n=document.createElement("script");n.src=e,n.async=!0;let i=!1,l=window.setTimeout(()=>{i||(i=!0,n.remove(),o(new Error("Leaflet load timed out")))},t);n.onload=()=>{if(i)return;i=!0,window.clearTimeout(l);let d=window.L;d?(n.remove(),a(d)):(n.remove(),o(new Error("Leaflet did not register")))},n.onerror=()=>{i||(i=!0,window.clearTimeout(l),n.remove(),o(new Error("Leaflet failed to load")))},document.head.appendChild(n)})}function ml(){if(An)return;let e=document.createElement("div");e.className="leaflet-pane",e.style.cssText="visibility:hidden;pointer-events:none",document.body.appendChild(e);let t=getComputedStyle(e).position==="absolute";if(e.remove(),!t)throw yt=null,new Error("Leaflet stylesheet did not apply (blocked by CSP, or the wrong MIME type)")}function bn(e,t){let a=zt?Lo??e.css:e.css,o=fl(a);if(Da)zt&&e.js!==zt&&console.warn(`[freva-databrowser] Leaflet already loaded from ${zt}; ignoring a second URL (${e.js}). A page can host only one Leaflet build.`);else{zt=e.js,Lo=e.css;let n=dl??((i,l)=>hl(i.js));Da=Promise.resolve(n(e,t)).catch(i=>{throw Da=null,zt=null,Lo=null,i})}let r=Da;return Promise.all([o,r]).then(([,n])=>(ml(),n)).catch(n=>{let i=document.getElementById(Bo);throw i&&i.remove(),yt=null,$t=!1,n})}function vn(e){let t=e.getBoundingClientRect();return t.width>0&&t.height>0}function wn(e){let t=e.getBoundingClientRect(),a=window.innerWidth||t.width,o=window.innerHeight||t.height,r=Math.max(t.left,0)-t.left,n=Math.max(t.top,0)-t.top,i=Math.min(t.right,a)-t.left,l=Math.min(t.bottom,o)-t.top;return{left:r,top:n,right:i,bottom:l,width:Math.max(0,i-r),height:Math.max(0,l-n)}}function xn(e,t){let a=e.getBoundingClientRect(),o=t.getBoundingClientRect();return{top:o.top-a.top,left:o.left-a.left,right:o.right-a.left,bottom:o.bottom-a.top,width:o.width,height:o.height}}function Nt(e,t){if(!t.isConnected||!e.contains(t))return!1;if(!vn(e))return!0;let a=wn(e);if(a.width===0||a.height===0)return!1;let o=xn(e,t);return o.width===0&&o.height===0?!1:o.right>a.left&&o.left<a.right&&o.bottom>a.top&&o.top<a.bottom}function Ht(e,t,a,o={}){let r=o.margin??8,n=o.gap??6,i=o.placement??"below";if(!vn(e))return;let l=wn(e);if(l.width===0||l.height===0)return;let d=Math.max(0,l.width-r*2),p=Math.max(0,l.height-r*2),u=Math.min(d,o.maxWidth??d),s=Math.min(p,o.maxHeight??p);t.style.maxWidth=`${u}px`,t.style.maxHeight=`${s}px`,t.style.overflowY="auto",o.minWidth&&(t.style.minWidth=`${Math.min(o.minWidth,u)}px`);let g=xn(e,a),m=t.getBoundingClientRect(),A=Math.min(m.width||t.offsetWidth,u),v=Math.min(m.height||t.offsetHeight,s),b,k;i==="right"?(b=g.top,k=g.right+r,k+A>l.right-r&&(k=g.left-A-r)):(b=g.bottom+n,k=g.left,b+v>l.bottom-r&&(b=g.top-v-n)),k=Math.min(k,l.right-r-A),k=Math.max(k,l.left+r),b=Math.min(b,l.bottom-r-v),b=Math.max(b,l.top+r),t.style.left=`${Math.round(k)}px`,t.style.top=`${Math.round(b)}px`}var Oa=class{constructor(t,a){this.current=null,this.anchor=null,this.onCloseCb=null,this.reanchorCb=null,this.scrollMode="close",this.placement="below",this.root=t,a.listen(document,"mousedown",r=>{if(!this.current)return;let n=r.target;this.current.contains(n)||this.anchor&&this.anchor.contains(n)||this.close()}),a.listen(document,"keydown",r=>{if(this.current&&r.key==="Escape"){r.preventDefault();let n=this.anchor;this.close(),n?.focus()}});let o=()=>{if(!(!this.current||!this.anchor)){if(!this.anchor.isConnected&&!this.tryReanchor()){this.close();return}if(!Nt(this.root,this.anchor)&&!this.tryReanchor()){this.close();return}this.position(this.current,this.anchor,this.placement)}};a.listen(window,"resize",o),a.listen(window,"scroll",r=>{if(!this.current)return;let n=r.target;if(!(n&&typeof n.nodeType=="number"&&this.current.contains(n))){if(this.scrollMode==="close"){this.close();return}o()}},!0)}isOpen(){return this.current!==null}closeIfAnchorDetached(){this.current&&this.anchor&&!this.anchor.isConnected&&(this.tryReanchor()?this.position(this.current,this.anchor,this.placement):this.close())}tryReanchor(){let t=this.reanchorCb?.()??null;return t&&t.isConnected?(this.anchor=t,!0):!1}open(t,a,o={}){this.close();let r=c("div",{class:`pop show${o.className?" "+o.className:""}`,role:"dialog"});r.style.position="absolute";for(let n of Array.isArray(a)?a:[a])r.append(n);return this.root.append(r),this.current=r,this.anchor=t,this.onCloseCb=o.onClose??null,this.reanchorCb=o.reanchor??null,this.placement=o.placement??"below",this.scrollMode=o.scrollBehavior??(o.reanchor?"reposition":"close"),this.position(r,t,this.placement),o.autoFocus&&r.querySelector("input, button, [href], [tabindex], select, textarea")?.focus(),r}position(t,a,o){Ht(this.root,t,a,{placement:o})}close(){if(!this.current)return;let t=!!this.current.contains(this.root.ownerDocument.activeElement),a=this.anchor;this.current.remove(),this.current=null,this.anchor=null,this.reanchorCb=null;let o=this.onCloseCb;this.onCloseCb=null,t&&a&&a.isConnected&&a.focus(),o?.()}};var gl={time:(e,t)=>t(),start:()=>()=>{},getSummary:()=>({}),enabled:!1};function yn(e){if(!e)return gl;let t={},a=typeof performance<"u"&&typeof performance.now=="function",o=()=>a?performance.now():Date.now(),r=(i,l)=>{let d=t[i]??(t[i]={count:0,totalMs:0,maxMs:0,lastMs:0});if(d.count++,d.totalMs+=l,d.lastMs=l,l>d.maxMs&&(d.maxMs=l),a&&typeof performance.mark=="function"&&typeof performance.measure=="function")try{performance.measure(`fdb:${i}`,{start:o()-l,duration:l})}catch{}},n={enabled:!0,time(i,l){let d=o();try{return l()}finally{r(i,o()-d)}},start(i){let l=o();return()=>r(i,o()-l)},getSummary:()=>t};return globalThis.__frevaPerf=n,n}var bl=[...new Set([...ht,...Bt])];function kn(e){if(!e||typeof e!="object"||Array.isArray(e))return!1;for(let t of Object.values(e))if(typeof t!="string")return!1;return!0}function vl(e,t=bl){let a={};for(let o of t){let r=e[o];kn(r)&&Object.keys(r).length>0&&(a[o]={...r})}return a}function Po(e,t){let a={};for(let o of new Set([...Object.keys(e),...Object.keys(t)]))a[o]={...e[o]??{},...t[o]??{}};return a}function Ro(e){if(!e||typeof e!="object")return{};let t={};for(let[a,o]of Object.entries(e))kn(o)&&(t[a]={...o});return t}async function wl(e,t,a,o=8e3){if(typeof fetch!="function"||typeof document>"u")return{};let r=t.abortController(),n=setTimeout(()=>r.abort(),o);try{let i=await fetch(e,{signal:r.signal,credentials:"same-origin"});if(!i.ok)return{};let l=(i.headers.get("content-type")??"").toLowerCase(),d=await i.text(),p=l.includes("javascript")||l.includes("ecmascript"),u=l===""||l.includes("application/octet-stream"),s=/\.m?js(?:[?#]|$)/i.test(e),g=/^\s*</.test(d);if(!(p||u&&s&&!g))return{}}catch{return{}}finally{clearTimeout(n)}return t.isDisposed?{}:new Promise(i=>{let l=!1,d=u=>{l||(l=!0,i(u))},p;try{p=document.createElement("script")}catch{d({});return}p.src=e,p.async=!0,p.addEventListener("load",()=>{d(vl(typeof window<"u"?window:{}))}),p.addEventListener("error",()=>d({})),t.add(()=>{p.remove(),d({})}),t.setTimeout(()=>{p.remove(),d({})},o),a.appendChild(p)})}function Mn(e){return Ro(e.metadata)}async function Tn(e,t,a){let o=Ro(e.metadata),r=await xl(),n=e.metadataScriptUrl;if(!n)return Po(r,o);let i=await wl(n,t,a);return Po(Po(r,i),o)}async function xl(){try{let e=await Promise.resolve().then(()=>(Cn(),Sn));return Ro(e.BUILTIN_METADATA)}catch{return{}}}var En=`/* styles.css - tokens + component styles for both themes.
   Ported from the prototype (the binding pixel source). The generic \`.overview\` class is
   renamed to \`.overview-mode\` to avoid a class collision that blanks the page.

   The TERMINAL's styles are NOT here: they moved to @freva-org/freva-client-terminal, which injects
   its own scoped stylesheet into the terminal window's root. Rules matching \`.freva-term\` do not
   belong in this file. */

.freva-db {
  --r: 10px;
  --r-sm: 7px;
  /* Advertise the app's scheme to native controls (scrollbars, form pickers) AND to embedded
     cross-origin iframes: an iframe inherits the embedder's used color-scheme, so the GridLook 3D
     viewer picks THIS up as its prefers-color-scheme and follows the databrowser theme (it otherwise
     falls back to the OS scheme). Overridden to dark under [data-theme="night"] below. */
  color-scheme: light;
  /* The mount target must have a definite height (the demo uses 100vh); .freva-db fills it so
     .fdb-app's grid can keep header/footer fixed and give each panel its own bounded scroll. */
  position: relative;
  height: 100%;
  min-height: 0;
  --mono: "JetBrains Mono", ui-monospace, "SF Mono", Menlo, monospace;
  --ui: system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  --bg: #eef1f6;
  --surface: #fff;
  --surface-2: #f4f6fa;
  --surface-3: #e9edf4;
  --text: #0e1726;
  --dim: #475569;
  /* Raised from #8a97ac (~2.96:1 on white). --faint is used for MEANINGFUL small text - counts,
     file paths, table headers, flavour captions - so it has to clear 4.5:1 on every day surface,
     not just look quiet. #5f6b7c measures 5.41:1 on --surface and 4.61:1 on --surface-3, the
     lightest and darkest day backgrounds it lands on. */
  --faint: #5f6b7c;
  --border: #dce2ec;
  --border-2: #c7d0dd;
  --accent: #2a63e8;
  --accent-2: #1e50c8;
  --accent-soft: color-mix(in srgb, var(--accent) 10%, transparent);
  --good: #1e9e6a;
  --warn: #c7841e;
  --danger: #d8543c;
  --ocean: #d9e6f2;
  --land: #c5d2e0;
  --shadow: 0 1px 2px rgba(16, 28, 52, 0.06), 0 4px 16px rgba(16, 28, 52, 0.07);
  font-family: var(--ui);
  color: var(--text);
}
.freva-db[data-theme="night"] {
  color-scheme: dark;
  --bg: #0a1120;
  --surface: #0f1a2e;
  --surface-2: #142339;
  --surface-3: #1a2c46;
  --text: #e7edf7;
  --dim: #9dabc4;
  /* Night equivalent: #5e6e88 measured 3.06-3.36:1 across the night surfaces. #8595b0 measures
     4.64:1 against --surface-3 (#1a2c46), the LIGHTEST night surface and therefore the worst case.
     The --text / --dim / --faint hierarchy and the dark-blue identity are unchanged. */
  --faint: #8595b0;
  --border: #213352;
  --border-2: #2c4267;
  --accent: #4f8df7;
  --accent-2: #6aa0ff;
  --accent-soft: color-mix(in srgb, var(--accent) 16%, transparent);
  --good: #34c98a;
  --warn: #e6b14e;
  --danger: #f0795f;
  --ocean: #0e2138;
  --land: #1c3554;
  --shadow: 0 1px 2px rgba(0, 0, 0, 0.5), 0 6px 20px rgba(0, 0, 0, 0.4);
}
.freva-db,
.freva-db * {
  box-sizing: border-box;
}
.freva-db :focus-visible {
  outline: 2px solid var(--accent);
  outline-offset: 2px;
  border-radius: 4px;
}

/* The app is a 3-row grid (header / body / footer); header + footer never
   scroll. The body is a 3-column grid (facets / center / details); each panel owns its scroll
   and its height never depends on another panel's content. */
.fdb-app {
  height: 100%;
  min-height: 680px;
  display: grid;
  grid-template-rows: auto 1fr auto;
  /* An implicit grid column sizes to the WIDEST row's min-content, so at phone widths the top bar
     stretched the whole app past the viewport and took the body - top row included - with it.
     An explicit 0 minimum lets the column shrink; the rows clip or wrap their own content. */
  grid-template-columns: minmax(0, 1fr);
  background: var(--bg);
  color: var(--text);
  transition:
    background-color 0.35s,
    color 0.35s;
}

.top {
  display: flex;
  align-items: center;
  gap: 14px;
  height: 56px;
  flex-shrink: 0;
  min-width: 0; /* \u2026and the bar itself must be allowed to shrink rather than set the app's width */
  padding: 0 16px;
  background: var(--surface);
  border-bottom: 1px solid var(--border);
  transition:
    background-color 0.35s,
    border-color 0.35s;
}
.brand {
  display: flex;
  align-items: center;
  gap: 10px;
  font-weight: 700;
  font-size: 15px;
  white-space: nowrap;
}
.brand .mark {
  width: 28px;
  height: 28px;
  border-radius: 8px;
  display: grid;
  place-items: center;
  color: #fff;
  background: linear-gradient(135deg, var(--accent), color-mix(in srgb, var(--accent) 55%, #fff));
  font-size: 13px;
}
.brand .brand-logo {
  width: 45px;
  height: 45px;
  object-fit: contain;
  display: block;
  flex-shrink: 0;
  margin: -4px 0;
}
.lens {
  display: flex;
  align-items: center;
  gap: 8px;
  height: 36px;
  padding: 0 10px;
  border-radius: var(--r-sm);
  border: 1px solid var(--border);
  background: var(--surface-2);
  cursor: pointer;
  font-size: 13px;
  color: var(--text);
  white-space: nowrap;
  font-family: inherit;
}
.lens:hover {
  border-color: var(--border-2);
}
.lens .k {
  color: var(--faint);
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}
.lens .v {
  font-weight: 600;
}
.search {
  flex: 1;
  min-width: 0; /* a flex item's default \`min-width: auto\` would floor the bar at the input's width */
  position: relative;
}
.search input {
  width: 100%;
  height: 40px;
  border-radius: var(--r-sm);
  border: 1px solid var(--border);
  background: var(--surface-2);
  color: var(--text);
  font-size: 14px;
  font-family: inherit;
  /* Right padding clears the furniture: the spinner's reserved slot and the \u2318K hint. Reserved at
     all times, so a search that starts loading never reflows the text under the caret. */
  padding: 0 84px 0 40px;
  outline: none;
  transition:
    border-color 0.15s,
    box-shadow 0.15s,
    background-color 0.35s;
}
.search input:focus {
  border-color: var(--accent);
  box-shadow: 0 0 0 3px var(--accent-soft);
}

/* The live edge.
 *
 * A slow wash of light in the field's own border: it drifts along the edge, it rises and falls, and
 * it stops the moment somebody is typing in the field. Deliberately NOT an object travelling the
 * border - a small, high-contrast shape with an outline is something the visual system locks onto
 * and TRACKS at any speed - so this is one wide, low-contrast gradient whose brightest region
 * moves, with no edge for the eye to follow.
 *
 * The two cycles do not divide into each other (30s and 11s), so they drift in and out of phase and
 * never settle into a beat a reader can anticipate, which is how a slow loop becomes a tic.
 *
 * Drawn at \`inset: 0\`, so the ring sits exactly ON the input's own 1px border rather than beside
 * it. That is what makes the failure mode harmless: where \`mask-composite\` is unavailable the
 * pseudo-element paints nothing useful and the field still has the ordinary accent border
 * underneath it, rather than no border at all.
 */
.freva-db .search {
  /* One place to tune the whole treatment. */
  --db-breathe: 11s; /* one full rise and fall */
  --db-drift: 30s; /* one full pass of the light along the edge */
  --db-amp: 0.45; /* how far the breath dips below full */
  --db-hot: 62%; /* accent content of the light at its brightest */
}
@keyframes db-edge-drift {
  from {
    background-position: 0% 50%;
  }
  to {
    background-position: 260% 50%;
  }
}
@keyframes db-edge-breathe {
  0%,
  100% {
    opacity: calc(1 - var(--db-amp));
  }
  50% {
    opacity: 1;
  }
}
.freva-db .search::before {
  content: "";
  position: absolute;
  inset: 0;
  border-radius: var(--r-sm);
  padding: 1px;
  pointer-events: none;
  background: linear-gradient(
    100deg,
    color-mix(in srgb, var(--accent) 18%, transparent) 0%,
    color-mix(in srgb, var(--accent) var(--db-hot), transparent) 26%,
    color-mix(in srgb, var(--accent) 14%, transparent) 52%,
    color-mix(in srgb, var(--accent) var(--db-hot), transparent) 76%,
    color-mix(in srgb, var(--accent) 18%, transparent) 100%
  );
  /* Wider than the box, so the drift has somewhere to travel from and to. */
  background-size: 260% 100%;
  /* Keep only the ring: paint the border box, subtract the padding box. */
  -webkit-mask:
    linear-gradient(#000 0 0) content-box,
    linear-gradient(#000 0 0);
  -webkit-mask-composite: xor;
  mask:
    linear-gradient(#000 0 0) content-box,
    linear-gradient(#000 0 0);
  mask-composite: exclude;
  animation:
    db-edge-drift var(--db-drift) linear infinite,
    db-edge-breathe var(--db-breathe) ease-in-out infinite;
}
/*
 * In use, it stops.
 *
 * The caret is the thing to watch while you type, and a field somebody is reading their own
 * keystrokes in is the one place a moving edge would genuinely be in the way. So focus settles it
 * to an even, steady ring - still lit, no longer moving.
 */
.freva-db .search:focus-within::before {
  /*
   * \`animation: none\`, not \`animation-play-state: paused\`. A paused animation still holds the
   * property at whatever keyframe it stopped on, which would leave the ring at whatever brightness
   * the breath was passing through - a focus ring whose intensity depends on when you clicked.
   * Removing the animations entirely is what lets the declarations below take effect.
   */
  animation: none;
  opacity: 1;
  background: linear-gradient(
    90deg,
    color-mix(in srgb, var(--accent) 55%, transparent),
    color-mix(in srgb, var(--accent) 92%, transparent),
    color-mix(in srgb, var(--accent) 55%, transparent)
  );
  background-size: 100% 100%;
}
@media (prefers-reduced-motion: reduce) {
  /* The same edge, held still. The treatment is the light, not the movement of it, so the
     reduced-motion field is the finished thing minus its motion - never a bare box. */
  .freva-db .search::before {
    animation: none;
    opacity: 0.85;
    background-position: 40% 50%;
  }
}
.search .ic {
  position: absolute;
  left: 13px;
  top: 50%;
  transform: translateY(-50%);
  color: var(--faint);
  display: grid;
  place-items: center;
}
.icon-btn {
  height: 36px;
  min-width: 36px;
  padding: 0 9px;
  border-radius: var(--r-sm);
  border: 1px solid var(--border);
  background: var(--surface-2);
  color: var(--dim);
  cursor: pointer;
  display: inline-flex;
  align-items: center;
  gap: 7px;
  font-size: 13px;
  font-family: inherit;
  transition:
    background-color 0.15s,
    color 0.15s,
    border-color 0.15s;
  white-space: nowrap;
}
.icon-btn:hover {
  color: var(--text);
  border-color: var(--border-2);
}
.icon-btn.on {
  background: var(--accent-soft);
  color: var(--accent);
  border-color: transparent;
}
.theme {
  width: 60px;
  height: 34px;
  border-radius: 999px;
  border: 1px solid var(--border);
  background: var(--surface-2);
  cursor: pointer;
  position: relative;
  flex-shrink: 0;
  transition:
    background-color 0.35s,
    border-color 0.35s;
}
.theme .knob {
  position: absolute;
  top: 2px;
  left: 2px;
  width: 28px;
  height: 28px;
  border-radius: 999px;
  background: var(--surface);
  box-shadow: var(--shadow);
  display: grid;
  place-items: center;
  color: var(--accent);
  transition: transform 0.35s cubic-bezier(0.4, 0, 0.2, 1);
}
.freva-db[data-theme="night"] .theme .knob {
  transform: translateX(26px);
}

.body {
  display: grid;
  /* minmax(0, \u2026) rather than a bare 1fr: a \`1fr\` track floors at its content's min-content width,
     so at phone widths the centre column grew past the viewport and took the top row - Clear all
     and the Browse/Overview cluster included - off screen with it. The explicit 0 minimum lets the
     column actually shrink; \`.center\` already sets \`min-width: 0\` and clips its own overflow. */
  grid-template-columns: auto minmax(0, 1fr) auto;
  min-height: 0;
  position: relative;
}
/* Explicit placement: hiding .side (metaview) must NOT let .center/.details-panel auto-place into the
   wrong track (that mis-sized the center in metadata-focused view). */
.side {
  grid-column: 1;
}
.center {
  grid-column: 2;
}
.details-panel {
  grid-column: 3;
}

/* SIDEBAR */
.side {
  width: 268px;
  flex-shrink: 0;
  min-height: 0;
  border-right: 1px solid var(--border);
  background: var(--surface);
  display: flex;
  flex-direction: column;
  overflow: hidden;
  transition:
    width 0.22s ease,
    background-color 0.35s,
    border-color 0.35s;
}
.fdb-app.metaview .side {
  display: none;
}
/* Collapsible: the sidebar collapses to a slim rail with a reopen affordance; persisted. */
.fdb-app.side-collapsed .side {
  width: 44px;
}
.fdb-app.side-collapsed .side .side-scroll,
.fdb-app.side-collapsed .side .side-head .side-title {
  display: none;
}
.fdb-app.side-collapsed .side .side-filterhead {
  display: none;
}
.side-head {
  display: flex;
  align-items: center;
  justify-content: flex-end;
  gap: 6px;
  padding: 6px 8px 2px;
}
.side-head .side-title {
  padding: 2px 4px;
}
.side-collapse {
  width: 28px;
  height: 28px;
  margin-left: auto;
  border: 1px solid var(--border);
  border-radius: 7px;
  background: var(--surface-2);
  color: var(--dim);
  cursor: pointer;
  display: inline-grid;
  place-items: center;
  flex-shrink: 0;
  transition:
    color 0.12s,
    border-color 0.12s;
}
.side-collapse:hover {
  color: var(--text);
  border-color: var(--border-2);
}
.side-collapse .chev {
  transition: transform 0.22s;
  display: inline-grid;
  place-items: center;
  transform: rotate(180deg);
}
.fdb-app.side-collapsed .side-collapse {
  margin: 0 auto;
}
.fdb-app.side-collapsed .side-collapse .chev {
  transform: rotate(0deg);
}
.side-scroll {
  overflow-y: auto;
  padding: 10px 10px 16px;
  flex: 1;
}
.side-title {
  font-size: 10.5px;
  font-weight: 700;
  letter-spacing: 0.09em;
  text-transform: uppercase;
  color: var(--faint);
  padding: 8px 8px 4px;
  display: flex;
  align-items: center;
}
.facet {
  border-radius: var(--r-sm);
}
/* The per-facet sort control. Present in the markup at all times so opening a panel stays a class
   toggle rather than a re-render (see sidebar.ts), and revealed only once there is a list to sort.
   \`margin-left: auto\` is deliberately NOT used: the +N / -N clear badges already claim that slot,
   and the sort must sit between them and the chevron in both cases. */
.facet .fh-sort {
  display: none;
}
.facet.open .fh-sort {
  display: inline-flex;
  align-items: center;
  flex-shrink: 0;
  height: 21px;
  gap: 4px;
  padding: 0 6px;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--surface-2);
  color: var(--faint);
  cursor: pointer;
  font-family: inherit;
  transition:
    color 0.12s,
    border-color 0.12s;
}
.facet.open .fh-sort:hover {
  color: var(--accent);
  border-color: var(--accent);
}

.facet-head {
  position: relative;
  display: flex;
  align-items: center;
  gap: 8px;
  height: 36px;
  padding: 0 8px 0 10px;
  border-radius: var(--r-sm);
  cursor: pointer;
  font-size: 12.5px;
  color: var(--text);
  transition:
    background-color 0.12s,
    color 0.12s;
  width: 100%;
  border: none;
  background: none;
  font-family: inherit;
  text-align: left;
}
.facet-head:hover {
  background: var(--surface-2);
}
.facet-head .chev {
  color: var(--faint);
  width: 12px;
  display: inline-grid;
  place-items: center;
  transition: transform 0.2s;
}
.facet.open > .facet-head .chev {
  transform: rotate(90deg);
}
.facet-head .fh-label {
  font-weight: 600;
  letter-spacing: 0.005em;
}
.facet-head .badge {
  margin-left: auto;
  font-size: 10px;
  color: var(--faint);
  font-family: var(--mono);
  font-weight: 500;
}
/* The \`+N\` / \`-N\` clear buttons are styled with the rest of the inclusion/exclusion language at
   the end of this sheet - one filled, one dashed, and neither of them a bare accent pill. */
.facet-head .fh-count {
  margin-left: 0;
  font-family: var(--mono);
}
.facet-head.active .fh-label {
  color: var(--accent);
}
.facet-head.active .chev {
  color: var(--accent);
}
.facet-head.active::before {
  content: "";
  position: absolute;
  left: 1px;
  top: 8px;
  bottom: 8px;
  width: 3px;
  border-radius: 2px;
  background: var(--accent);
}
.facet-body {
  display: none;
  padding: 1px 0 6px 16px;
}
.facet.open > .facet-body {
  display: block;
}
.fval {
  /*
   * Positioned, because the share bar is a \`::before\` inside it. It was only the overview's cards
   * that drew one, so the context lived there; the sidebar draws the same bar now, and a row that
   * is \`static\` would hang the bar off whatever ancestor happened to be positioned - which in the
   * sidebar is the scroll container, giving one full-width bar behind the whole list.
   */
  position: relative;
  isolation: isolate;
  display: flex;
  align-items: center;
  gap: 8px;
  min-height: 27px;
  padding: 3px 8px;
  border-radius: 5px;
  font-size: 12px;
  color: var(--dim);
  cursor: pointer;
  transition:
    background-color 0.12s,
    color 0.12s;
  width: 100%;
  border: none;
  background: none;
  font-family: inherit;
  text-align: left;
}
.fval:hover {
  background: var(--surface-2);
  color: var(--text);
}
.fval.sel {
  color: var(--accent);
  font-weight: 600;
}
.fval[aria-disabled="true"] {
  opacity: 0.5;
  cursor: not-allowed;
}
.fval.locked {
  opacity: 1;
  cursor: default;
} /* base scope: active, not disabled-looking */
.fval.locked:hover {
  background: transparent;
}
.fval .nm {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.fval .n {
  margin-left: auto;
  font-family: var(--mono);
  font-size: 10px;
  color: var(--faint);
  flex-shrink: 0;
  padding-left: 6px;
}
.fval .cb {
  width: 13px;
  height: 13px;
  border-radius: 3px;
  border: 1.5px solid var(--border-2);
  flex-shrink: 0;
  display: grid;
  place-items: center;
  color: transparent;
}
.fval.sel .cb {
  background: var(--accent);
  border-color: var(--accent);
  color: #fff;
}
.fmore {
  font-size: 11px;
  color: var(--faint);
  padding: 4px 8px 2px;
  font-style: italic;
}
.special {
  display: flex;
  align-items: center;
  gap: 8px;
  min-height: 34px;
  padding: 5px 8px;
  border-radius: var(--r-sm);
  font-size: 13px;
  font-weight: 600;
  color: var(--text);
  cursor: pointer;
  transition: background-color 0.12s;
  width: 100%;
  border: none;
  background: none;
  font-family: inherit;
  text-align: left;
}
.special:hover {
  background: var(--surface-2);
}
.special.set {
  background: var(--accent-soft);
  color: var(--accent);
}
.special .val {
  margin-left: auto;
  font-size: 10px;
  color: var(--faint);
  font-family: var(--mono);
  max-width: 120px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.special.set .val {
  color: var(--accent);
}
.special .lead {
  display: inline-grid;
  place-items: center;
}
.addbtn {
  width: 100%;
  margin: 8px 0 4px;
  height: 38px;
  border-radius: var(--r-sm);
  border: 1px dashed var(--border-2);
  background: transparent;
  color: var(--dim);
  font-size: 12.5px;
  font-weight: 600;
  cursor: pointer;
  font-family: inherit;
}
.addbtn:hover {
  background: var(--surface-2);
  color: var(--text);
}

/* CENTER - toprow + res-bar are fixed chrome; only the results column scrolls. */
.center {
  display: flex;
  flex-direction: column;
  min-width: 0;
  min-height: 0;
  position: relative;
  overflow: hidden;
}
.center-fixed {
  flex-shrink: 0;
  padding: 14px 18px 0;
}
.results-scroll {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  /* NO top padding. \`.list-head\` sticks at \`top: 0\` of the PADDING box, so any top padding leaves a
     band between the scrollport edge and the pinned header - and once the header is pinned, the
     content occupying that band is the rows. That is the strip of file text that painted above the
     column header. The spacing belongs on the content that wants it, below - not here. */
  padding: 0 18px 92px;
}

/* Breathing room applied to the content rather than to the scrollport, so it scrolls away with
   that content instead of holding a gap open above the pinned header. */
.results-scroll > .overview-mode,
.results-scroll > .list-head[hidden] + .rows {
  margin-top: 8px;
}
.toprow {
  display: flex;
  align-items: flex-start;
  gap: 10px;
  margin-bottom: 12px;
}
.chips {
  display: flex;
  flex-wrap: wrap;
  gap: 7px;
  align-items: center;
  flex: 1;
  min-width: 0;
}
.chip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  height: 28px;
  padding: 0 10px;
  border-radius: 999px;
  background: var(--accent-soft);
  color: var(--accent);
  font-size: 12.5px;
  font-weight: 600;
  cursor: pointer;
  border: none;
  font-family: inherit;
}
.chip .x {
  opacity: 0.7;
  display: inline-grid;
  place-items: center;
}
.chip.geo {
  background: color-mix(in srgb, var(--good) 14%, transparent);
  color: var(--good);
}
.clear-btn {
  flex-shrink: 0;
  height: 28px;
  padding: 0 12px;
  border-radius: 999px;
  border: 1px solid color-mix(in srgb, var(--danger) 40%, transparent);
  background: transparent;
  color: var(--danger);
  font-size: 12px;
  font-weight: 600;
  cursor: pointer;
  font-family: inherit;
  display: none;
}
.clear-btn.show {
  display: inline-flex;
  align-items: center;
}
.ctrl-cluster {
  flex-shrink: 0;
  /* The mode switch belongs on the RIGHT. Relying on \`.chips\` to fill the row only worked while
     there were chips: with none (or the row hidden) the cluster fell back to the left and the
     control jumped as soon as the first filter was applied. \`margin-left: auto\` states the
     intention instead of depending on a sibling's content. */
  margin-left: auto;
  display: inline-flex;
  align-items: center;
  gap: 3px;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 999px;
  padding: 4px;
  box-shadow: var(--shadow);
}
/* The worded half of clear-all. Right-aligned so it reads as the row's action rather than as a
   third thing about the count, and quiet until it is wanted - it is an undo, not an invitation. */
.sf-clear {
  margin-left: auto;
  flex-shrink: 0;
  height: 21px;
  padding: 0 8px;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--surface-2);
  color: var(--dim);
  font-family: inherit;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.01em;
  text-transform: none;
  cursor: pointer;
  transition:
    color 0.12s,
    border-color 0.12s,
    background-color 0.12s;
}
.sf-clear:hover {
  color: var(--danger);
  border-color: color-mix(in srgb, var(--danger) 55%, transparent);
}
.ctrl {
  width: 34px;
  height: 34px;
  border-radius: 999px;
  border: none;
  background: transparent;
  color: var(--dim);
  cursor: pointer;
  display: grid;
  place-items: center;
  transition:
    background-color 0.15s,
    color 0.15s;
}
.ctrl:hover {
  background: var(--surface-2);
  color: var(--text);
}
.ctrl.on {
  background: var(--accent-soft);
  color: var(--accent);
}
.ctrl-sep {
  width: 1px;
  height: 20px;
  background: var(--border);
  margin: 0 2px;
}

.overview-mode {
  display: none;
  margin-bottom: 16px;
}
.fdb-app.metaview .overview-mode {
  display: block;
}
.overview-cap {
  font-size: 12px;
  color: var(--faint);
  margin: 0 0 10px;
}
.stale-pill {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  margin-left: 8px;
  padding: 1px 8px;
  border-radius: 999px;
  background: color-mix(in srgb, var(--warn) 16%, transparent);
  color: var(--warn);
  font-size: 10.5px;
  font-weight: 700;
}
.facet-grid {
  --block-h: 256px;
  --block-gap: 12px;
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(min(244px, 100%), 1fr));
  gap: var(--block-gap);
  /* Rows are sized by their CONTENT and each card states its own block height (below). A fixed
     \`grid-auto-rows\` could never let a minimized row close up; \`min-content\` alone let an expanded
     card grow past its box (which is what killed the value scrollbar). This does both. */
  grid-auto-rows: min-content;
  align-items: start;
}
/* "stacked": every block a full-width row. A single column forces full width regardless of each
   card's saved span, and \`1 / -1\` overrides the inline \`span N\` so no implicit tracks (= no page
   overflow) can ever be created. */
.facet-grid.stacked {
  grid-template-columns: 1fr;
}
.facet-grid.stacked .fcard,
.facet-grid.stacked .ov-addrow {
  grid-column: 1 / -1 !important;
}
.fcard {
  border: 1px solid var(--border);
  border-radius: var(--r);
  background: var(--surface);
  display: flex;
  flex-direction: column;
  overflow: hidden;
  box-shadow: var(--shadow);
  height: var(--block-h);
}
/* one block down - and no further */
.fcard[data-rows="2"] {
  height: calc(var(--block-h) * 2 + var(--block-gap));
}
.fcard.tall {
  grid-row: span 2;
}
.fcard.wide {
  grid-column: span 2;
}
.fcard-empty {
  padding: 8px 10px;
  font-size: 12px;
  color: var(--faint);
  font-style: italic;
}
/* Value lists lay out as a GRID that fits as many ~200px columns as the width allows and then
   grows DOWNWARD (vertical scroll), row-major. This replaces CSS multi-column, whose fixed-height
   column packing forced a horizontal scroll when a card was stretched. Applies uniformly: a narrow
   card gets one column, a wide/stacked/full-width card gets several - always scrolling vertically. */
.fcard-h {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 12px;
  border-bottom: 1px solid var(--border);
  font-weight: 700;
  font-size: 13px;
}
/* The header toggles collapse/expand; make that obvious (its own controls keep their cursors). */
.fcard-h.clickable {
  cursor: pointer;
}
.fcard-h.clickable:hover {
  background: var(--surface-2);
}
.fcard-h .badge {
  margin-left: auto;
  font-family: var(--mono);
  font-size: 10px;
  color: var(--faint);
}
.fcard-h .fh-count {
  margin-left: 6px;
  font-family: var(--mono);
}
/* NOTE: the hover treatment is a real element swap in a neutral colour; see the end of this sheet.
   A red \`::after\` cross drawn OVER the badge with the number merely turned transparent puts both
   on screen at once, and takes the host's danger colour - which in a red-branded deployment is
   the accent. */
.fcard-h.active {
  color: var(--accent);
}
.fcard-h .exp {
  margin-left: 4px;
  color: var(--faint);
  cursor: pointer;
  border: none;
  background: none;
  padding: 2px;
  display: inline-grid;
  place-items: center;
}
.fcard-h .exp:hover {
  color: var(--accent);
}
.fcard-h .exp.on {
  color: var(--accent);
}
.fcard .within {
  margin: 8px 10px 4px;
  height: 30px;
  border: 1px solid var(--border);
  background: var(--surface-2);
  color: var(--text);
  border-radius: 6px;
  padding: 0 9px;
  font-size: 12px;
  outline: none;
  font-family: inherit;
}
.fcard .within:focus {
  border-color: var(--accent);
  box-shadow: 0 0 0 3px var(--accent-soft);
}
.fcard-vals {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(min(190px, 100%), 1fr));
  column-gap: 10px;
  align-content: start;
  overflow-y: auto;
  overflow-x: hidden;
  padding: 2px 6px 8px;
  flex: 1 1 auto;
  min-height: 0;
}
.fcard-vals .fmore,
.fcard-vals .fcard-empty {
  grid-column: 1 / -1;
} /* notes span the whole width */
.fcard-vals .fval {
  font-size: 12px;
}
.fcard .editline {
  padding: 12px;
  display: flex;
  align-items: center;
  gap: 8px;
}
.fcard .editline .v {
  font-family: var(--mono);
  font-size: 11px;
  color: var(--accent);
  flex: 1;
  word-break: break-all;
}
.fcard .editline .v.off {
  color: var(--faint);
}

.res-bar {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px 12px;
  border-radius: var(--r);
  background: var(--surface-2);
  border: 1px solid var(--border);
  margin-bottom: 6px;
  flex-wrap: wrap;
}
.bar-div {
  width: 1px;
  height: 22px;
  background: var(--border);
  margin: 0 2px;
}
/* Select-all (in the results bar, both list + grid). Reuses the .cb checkbox box. */
.selall {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  height: 32px;
  padding: 0 9px;
  border: none;
  background: transparent;
  color: var(--dim);
  cursor: pointer;
  font-size: 12.5px;
  border-radius: 8px;
  white-space: nowrap;
}
.selall:hover:not(:disabled) {
  background: var(--surface-2);
  color: var(--text);
}
.selall:disabled {
  opacity: 0.4;
  cursor: default;
}
.selall .cb {
  color: transparent;
}
.selall .cb.on {
  background: var(--accent);
  border-color: var(--accent);
  color: #fff;
}
.selall .cb.mixed {
  background: var(--accent);
  border-color: var(--accent);
  position: relative;
}
.selall .cb.mixed::after {
  content: "";
  position: absolute;
  inset: 0;
  margin: auto;
  width: 9px;
  height: 2px;
  background: #fff;
  border-radius: 1px;
}
/* the file-panel controls fade/translate in place. The slot is RESERVED (this
   stays in flow with its width even when hidden) so nothing else in the bar moves; Export and the
   count never shift. pointer-events:none keeps the invisible controls unclickable. */
.panelctl {
  display: inline-flex;
  align-items: center;
  gap: 10px;
  opacity: 0;
  transform: translateY(-3px);
  pointer-events: none;
  transition:
    opacity 0.18s ease,
    transform 0.18s ease;
}
.panelctl.in {
  opacity: 1;
  transform: none;
  pointer-events: auto;
}
.res-bar.merged {
  border-color: color-mix(in srgb, var(--accent) 35%, var(--border));
  box-shadow: 0 2px 10px color-mix(in srgb, var(--accent) 12%, transparent);
}
@media (prefers-reduced-motion: reduce) {
  .panelctl {
    transition: none;
  }
}

.iconbtn {
  position: relative;
  width: 36px;
  height: 36px;
  border-radius: 999px;
  border: none;
  background: transparent;
  color: var(--dim);
  cursor: pointer;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  transition:
    background-color 0.15s,
    color 0.15s;
}
.iconbtn:hover {
  background: var(--surface-3);
  color: var(--text);
}
.iconbtn:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}
.iconbtn .caret {
  position: absolute;
  right: 3px;
  bottom: 4px;
  color: var(--faint);
  display: inline-grid;
  place-items: center;
}
.scope-tag {
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--faint);
  border: 1px solid var(--border-2);
  padding: 2px 7px;
  border-radius: 999px;
}
.res-count {
  font-size: 14px;
  font-weight: 700;
}
.res-count .sub {
  font-weight: 500;
  color: var(--faint);
  font-size: 12px;
  margin-left: 6px;
}
.spacer {
  flex: 1;
}
.seg {
  display: flex;
  border: 1px solid var(--border-2);
  border-radius: var(--r-sm);
  overflow: hidden;
}
.seg button {
  height: 34px;
  padding: 0 12px;
  min-width: 36px;
  display: grid;
  place-items: center;
  border: none;
  cursor: pointer;
  background: var(--surface);
  color: var(--dim);
  transition:
    background-color 0.15s,
    color 0.15s;
  font-family: inherit;
  font-size: 12px;
}
.seg button.on {
  background: var(--accent-soft);
  color: var(--accent);
}

.btn {
  height: 34px;
  padding: 0 13px;
  border-radius: var(--r-sm);
  border: 1px solid var(--border-2);
  background: var(--surface);
  color: var(--text);
  font-size: 12.5px;
  font-weight: 600;
  cursor: pointer;
  font-family: inherit;
  display: inline-flex;
  align-items: center;
  gap: 7px;
  transition:
    background-color 0.15s,
    border-color 0.15s;
}
.btn:hover {
  background: var(--surface-2);
}
.btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
.btn.primary {
  background: var(--accent);
  border-color: var(--accent);
  color: #fff;
}
.btn.primary:hover {
  background: var(--accent-2);
}
.ac {
  position: absolute;
  z-index: 70;
  background: #0e1626;
  border: 1px solid #28406a;
  border-radius: 8px;
  box-shadow: var(--shadow);
  min-width: 160px;
  max-height: 240px;
  overflow: auto;
  padding: 4px;
  display: none;
}
.ac.show {
  display: block;
}
.ac-item {
  padding: 6px 9px;
  border-radius: 5px;
  font-family: var(--mono);
  font-size: 12px;
  color: #d7e2f4;
  cursor: pointer;
  display: flex;
  gap: 8px;
  align-items: center;
}
.ac-item:hover,
.ac-item.hl {
  background: rgba(79, 141, 247, 0.2);
}
.ac-item .cnt {
  margin-left: auto;
  color: #6f7f9c;
  font-size: 10px;
}

/* results */
.rows {
  border: 1px solid var(--border);
  border-radius: var(--r);
  overflow: hidden;
  background: var(--surface);
}
.row {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 0 14px;
}
.row {
  min-height: 48px;
  padding-top: 8px;
  padding-bottom: 8px;
  border-bottom: 1px solid var(--border);
  cursor: pointer;
  transition: background-color 0.12s;
}
.row:last-child {
  border-bottom: none;
}
.row:hover {
  background: var(--surface-2);
}
.row.focus {
  background: var(--accent-soft);
  box-shadow: inset 0 0 0 1px color-mix(in srgb, var(--accent) 55%, transparent);
}
.row.picked {
  background: color-mix(in srgb, var(--accent-soft) 62%, transparent);
}
.row.focus.picked {
  background: var(--accent-soft);
  box-shadow: inset 0 0 0 1px color-mix(in srgb, var(--accent) 55%, transparent);
}
.cb {
  width: 17px;
  height: 17px;
  border-radius: 4px;
  border: 1.5px solid var(--border-2);
  flex-shrink: 0;
  display: grid;
  place-items: center;
  background: var(--surface);
  color: transparent;
  transition: background-color 0.12s;
  padding: 0;
  cursor: pointer;
}
.row.picked .cb,
.gcard.picked .cb {
  background: var(--accent);
  border-color: var(--accent);
  color: #fff;
}
.uricell {
  display: flex;
  align-items: center;
  gap: 12px;
  min-width: 0;
  flex: 1;
}
.row .ext {
  width: 30px;
  height: 30px;
  border-radius: 7px;
  flex-shrink: 0;
  display: grid;
  place-items: center;
  font-family: var(--mono);
  font-size: 9px;
  font-weight: 700;
  background: color-mix(in srgb, var(--accent) 12%, transparent);
  color: var(--accent);
}
.row .meta {
  flex: 1;
  min-width: 0;
}
/* ONE complete path per row (see results.ts \`pathEl\`). A single line with an ellipsis: the whole
   value lives in \`title\` and \`aria-label\`, so clipping loses nothing but pixels. \`direction: rtl\`
   keeps the END of a long path - the part that identifies the file - visible when it is clipped,
   while \`unicode-bidi: plaintext\` stops the text itself being reordered. */
.row .path {
  font-size: 13px;
  font-weight: 600;
  font-family: var(--mono);
  overflow: hidden;
  direction: rtl;
  text-align: left;
  text-overflow: ellipsis;
  white-space: nowrap;
  unicode-bidi: plaintext;
}
.fs {
  font-size: 11px;
  font-weight: 500;
  font-family: var(--mono);
  color: var(--dim);
  flex-shrink: 0;
  white-space: nowrap;
}
/* List-view column header (uri | fs type). Sits directly on top of the .rows box. */
.list-head {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 9px 14px;
  font-size: 10px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--dim);
  background: var(--surface-2);
  border: 1px solid var(--border);
  border-radius: var(--r) var(--r) 0 0;
}
.list-head[hidden] {
  display: none;
}
.list-head .lh-uri {
  flex: 1;
  padding-left: 42px;
}
.list-head .lh-fs {
  flex-shrink: 0;
  padding-right: 34px;
}
.list-head:not([hidden]) + .rows {
  border-top-left-radius: 0;
  border-top-right-radius: 0;
}
.grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(min(220px, 100%), 1fr));
  gap: 10px;
}
.gcard {
  border: 1px solid var(--border);
  border-radius: var(--r);
  background: var(--surface);
  padding: 12px;
  cursor: pointer;
  transition:
    border-color 0.12s,
    box-shadow 0.12s,
    transform 0.1s;
}
.gcard:hover {
  box-shadow: var(--shadow);
  border-color: var(--border-2);
  transform: translateY(-1px);
}
.gcard.focus {
  border-color: transparent;
  box-shadow:
    inset 0 0 0 1px var(--accent),
    0 0 0 3px var(--accent-soft);
}
.gcard.picked {
  background: color-mix(in srgb, var(--accent-soft) 55%, transparent);
}
.gcard .top2 {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 9px;
}
/* A card may WRAP the path across lines - it is still one complete value, never two. */
.gcard .path {
  font-size: 12px;
  font-weight: 600;
  font-family: var(--mono);
  line-height: 1.35;
  word-break: break-all;
}
.gcard .bits {
  font-size: 10.5px;
  color: var(--faint);
  margin-top: 6px;
  font-family: var(--mono);
}
.kebab {
  width: 30px;
  height: 30px;
  border-radius: 7px;
  border: none;
  background: transparent;
  color: var(--faint);
  cursor: pointer;
  display: inline-grid;
  place-items: center;
  flex-shrink: 0;
}
.kebab:hover {
  background: var(--surface-3);
  color: var(--text);
}

.load-next {
  width: 100%;
  justify-content: center;
}
.more-note {
  text-align: center;
  padding: 12px;
  color: var(--faint);
  font-size: 12.5px;
}

/* states */
.skeleton-rows {
  border: 1px solid var(--border);
  border-radius: var(--r);
  overflow: hidden;
  background: var(--surface);
}
.sk-row {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 13px 14px;
  border-bottom: 1px solid var(--border);
}
.sk-row:last-child {
  border-bottom: none;
}
.sk {
  background: linear-gradient(
    90deg,
    var(--surface-2) 25%,
    var(--surface-3) 50%,
    var(--surface-2) 75%
  );
  background-size: 400% 100%;
  animation: sk 1.3s ease infinite;
  border-radius: 6px;
}
@keyframes sk {
  from {
    background-position: 100% 0;
  }
  to {
    background-position: -100% 0;
  }
}
.fdb-app[data-reduced-motion="true"] .sk {
  animation: none;
}
.state-msg {
  text-align: center;
  padding: 44px 22px;
  color: var(--dim);
  border: 1px solid var(--border);
  border-radius: var(--r);
  background: var(--surface);
}
.state-msg .big {
  color: var(--faint);
  margin-bottom: 12px;
  display: grid;
  place-items: center;
}
.state-msg p {
  font-size: 13.5px;
  line-height: 1.5;
  margin: 0 0 12px;
}
.state-msg.err {
  color: var(--danger);
}

.pickbar {
  position: absolute;
  left: 50%;
  bottom: 16px;
  transform: translateX(-50%);
  width: min(680px, calc(100% - 36px));
  background: var(--surface);
  border: 1px solid var(--accent);
  border-radius: var(--r);
  box-shadow: var(--shadow);
  padding: 10px 14px;
  display: none;
  align-items: center;
  gap: 12px;
  z-index: 20;
}
.pickbar.show {
  display: flex;
}
.pickbar .cnt {
  font-size: 13px;
  font-weight: 600;
}
.pickbar .cnt b {
  color: var(--accent);
  font-family: var(--mono);
}
.pickbar .x {
  cursor: pointer;
  color: var(--faint);
  border: none;
  background: none;
  display: inline-grid;
  place-items: center;
}

/* RIGHT DETAILS PANEL */
.details-panel {
  width: 340px;
  flex-shrink: 0;
  border-left: 1px solid var(--border);
  background: var(--surface);
  display: flex;
  flex-direction: column;
  overflow: hidden;
  transition:
    width 0.25s,
    border-color 0.25s,
    background-color 0.35s;
}
.details-panel.collapsed {
  width: 0;
  border-left: none;
}
.info-scroll {
  overflow-y: auto;
  flex: 1;
}
.info-head {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 14px 16px;
  border-bottom: 1px solid var(--border);
}
.info-head .t {
  font-weight: 700;
  font-size: 14px;
}
.info-head .x {
  margin-left: auto;
  cursor: pointer;
  color: var(--faint);
  border: none;
  background: none;
  display: inline-grid;
  place-items: center;
}
.empty {
  padding: 40px 22px;
  text-align: center;
  color: var(--dim);
}
.empty .big {
  font-size: 30px;
  color: var(--faint);
  margin-bottom: 12px;
  display: grid;
  place-items: center;
}
.empty p {
  font-size: 13px;
  line-height: 1.5;
  margin: 0;
}
.empty code {
  font-family: var(--mono);
  color: var(--accent);
}
.info-name {
  padding: 16px 16px 2px;
  font-weight: 700;
  font-size: 13.5px;
  font-family: var(--mono);
  word-break: break-all;
}
.info-sub {
  padding: 0 16px 14px;
  font-size: 11.5px;
  color: var(--faint);
  word-break: break-all;
}
.info-sec {
  font-size: 10.5px;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--faint);
  padding: 10px 16px 6px;
  display: flex;
  align-items: center;
  gap: 6px;
}
.meta {
  padding: 0 16px;
}
.meta-row {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  padding: 6px 0;
  border-bottom: 1px solid var(--border);
  font-size: 12.5px;
}
.meta-row:last-child {
  border-bottom: none;
}
.meta-row .k {
  color: var(--dim);
}
.meta-row .v {
  font-family: var(--mono);
  color: var(--text);
  text-align: right;
  font-weight: 500;
  word-break: break-all;
}
.miniwrap {
  padding: 4px 16px 6px;
}
.minimap {
  border-radius: 8px;
  overflow: hidden;
  border: 1px solid var(--border);
  position: relative;
}
.coords {
  font-family: var(--mono);
  font-size: 10.5px;
  color: var(--dim);
  display: flex;
  justify-content: space-between;
  margin-top: 6px;
}
.na {
  font-size: 12px;
  color: var(--faint);
  padding: 2px 16px 8px;
  font-style: italic;
}
.info-actions {
  padding: 14px 16px 18px;
}
.cat-seg {
  margin: 6px 0 10px;
}
.info-actions .btn {
  width: 100%;
  justify-content: center;
  margin-bottom: 8px;
}
.scope-note {
  font-size: 11px;
  color: var(--faint);
  margin: 0 0 8px;
}
.querying {
  padding: 22px 16px;
  font-size: 12.5px;
  color: var(--faint);
  font-family: var(--mono);
}
.querying .bar {
  height: 3px;
  background: var(--surface-3);
  border-radius: 2px;
  margin-top: 10px;
  overflow: hidden;
  position: relative;
}
.querying .bar::after {
  content: "";
  position: absolute;
  left: -40%;
  top: 0;
  height: 100%;
  width: 40%;
  background: var(--accent);
  border-radius: 2px;
  animation: slide 1s infinite;
}
@keyframes slide {
  to {
    left: 100%;
  }
}
.partial-flag {
  margin: 6px 16px;
  padding: 6px 9px;
  border-radius: 6px;
  font-size: 11.5px;
  color: var(--warn);
  background: color-mix(in srgb, var(--warn) 12%, transparent);
  border: 1px solid color-mix(in srgb, var(--warn) 40%, transparent);
}

.diff-summary {
  padding: 0 16px 10px;
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}
.varchip {
  display: inline-flex;
  gap: 6px;
  align-items: center;
  background: var(--surface-2);
  border: 1px solid var(--border);
  padding: 2px 9px;
  border-radius: 999px;
  font-size: 11px;
  font-family: var(--mono);
  color: var(--accent);
}
.dscroll {
  overflow-x: auto;
  border: 1px solid var(--border);
  border-radius: 8px;
  margin: 0 16px 4px;
}
.dmatrix {
  width: 100%;
  border-collapse: collapse;
  font-size: 11.5px;
  font-family: var(--mono);
}
.dmatrix th,
.dmatrix td {
  text-align: left;
  padding: 6px 9px;
  border-bottom: 1px solid var(--border);
  white-space: nowrap;
}
.dmatrix tr:last-child td {
  border-bottom: none;
}
.dmatrix thead th {
  color: var(--faint);
  font-weight: 700;
  text-transform: uppercase;
  font-size: 9.5px;
  letter-spacing: 0.05em;
  background: var(--surface-2);
}
.dmatrix td.rownum {
  color: var(--faint);
}
.dchip {
  padding: 1px 7px;
  border-radius: 5px;
  font-weight: 600;
}
/* Enlarge control + full-screen comparison overlay (scrolls X and Y for wide/tall tables). */
.diff-tools {
  display: flex;
  justify-content: flex-end;
  margin: 0 16px 6px;
}
.diff-enlarge {
  padding: 4px 10px;
  font-size: 12px;
  display: inline-flex;
  align-items: center;
  gap: 6px;
}
/* The full-screen comparison. */
.dmm-backdrop {
  /*
   * A <dialog>, so most of this is undoing the UA's own dialog box (fit-content width, auto
   * margins, a border, a background, absolute positioning) to get a full-viewport scrim back.
   *
   * \`position: fixed\` here is against the VIEWPORT even inside a transformed host container,
   * because a modal dialog is in the top layer and the top layer's containing block is the initial
   * one. The z-index is only for the no-<dialog> fallback path, where this is an ordinary element
   * again; nothing in the top layer needs one.
   *
   * \`dvh\`, not \`vh\`, on the modal below: on a phone \`vh\` is the viewport with the browser chrome
   * RETRACTED, so a modal capped at 92vh is taller than the screen it is on until the address bar
   * scrolls away - which for a modal that owns the scroll it never does.
   */
  position: fixed;
  inset: 0;
  width: 100%;
  height: 100%;
  max-width: 100%;
  max-height: 100%;
  margin: 0;
  border: none;
  color: inherit;
  z-index: 2147483000;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 24px;
  background: rgba(8, 14, 26, 0.58);
  /* Its own stacking context, so nothing inside can be re-ordered against the page by a host
     stylesheet that happens to give something a higher z-index. */
  isolation: isolate;
  overscroll-behavior: contain;
}
/* \`display: flex\` above would otherwise defeat the UA's own \`dialog:not([open])\` rule. */
.dmm-backdrop:not([open]) {
  display: none;
}
/* The scrim is painted by the dialog itself (it has to be, for the fallback path), so the real
   backdrop pseudo-element stays out of the way rather than doubling the dimming. */
.dmm-backdrop::backdrop {
  background: transparent;
}
.dmm-modal {
  display: flex;
  flex-direction: column;
  width: min(1200px, 96vw);
  max-height: min(92vh, 92dvh);
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  box-shadow: 0 24px 60px rgba(0, 0, 0, 0.4);
  overflow: hidden;
}
.dmm-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  padding: 12px 16px;
  border-bottom: 1px solid var(--border);
  background: var(--accent);
  color: #fff;
}
.dmm-title {
  font-weight: 600;
  font-size: 14px;
}
.dmm-head .x {
  background: transparent;
  border: none;
  color: #fff;
  cursor: pointer;
  border-radius: 6px;
  padding: 4px;
  display: inline-grid;
  place-items: center;
}
.dmm-head .x:hover {
  background: rgba(255, 255, 255, 0.18);
}
.dmm-body {
  overflow: auto;
  padding: 12px;
}
.dmm-body .dmatrix {
  font-size: 12.5px;
}
.dmm-body .dmatrix th,
.dmm-body .dmatrix td {
  padding: 8px 12px;
}
.shared {
  margin: 12px 16px 4px;
  border-top: 1px solid var(--border);
  padding-top: 2px;
}
.shared-head {
  cursor: pointer;
  padding-left: 0 !important;
  display: flex;
  align-items: center;
  gap: 6px;
  border: none;
  background: none;
  width: 100%;
  font-family: inherit;
}
.shared-head .chev2 {
  color: var(--faint);
  transition: transform 0.2s;
  margin-left: 2px;
  display: inline-grid;
  place-items: center;
}
.shared:not(.open) .shared-head .chev2 {
  transform: rotate(-90deg);
}
.shared-body {
  display: none;
}
.shared.open .shared-body {
  display: block;
}
.shared-body .miniwrap,
.shared-body .meta {
  padding-left: 0;
  padding-right: 0;
}

.status {
  height: 36px;
  flex-shrink: 0;
  border-top: 1px solid var(--border);
  background: var(--surface);
  display: flex;
  align-items: center;
  padding: 0 18px;
  font-size: 12px;
  color: var(--dim);
  gap: 14px;
  transition:
    background-color 0.35s,
    border-color 0.35s;
}
.status .mono {
  font-family: var(--mono);
}

/* popovers */
.pop {
  position: absolute;
  z-index: 50;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--r);
  box-shadow: var(--shadow);
  padding: 6px;
  display: none;
}
.pop.show {
  display: block;
}
.pop-item {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 9px 10px;
  border-radius: var(--r-sm);
  cursor: pointer;
  font-size: 13px;
  color: var(--text);
  border: none;
  background: none;
  width: 100%;
  font-family: inherit;
  text-align: left;
}
.pop-item:hover {
  background: var(--surface-2);
}
.pop-item .pic {
  width: 18px;
  display: inline-grid;
  place-items: center;
  color: var(--accent);
}
.pop-item .desc {
  font-size: 11px;
  color: var(--faint);
}
/* The export menu (components/exportMenu.ts)
   ONE layout for both the whole-result Export and the pickbar's selected-files Download.

   The previous markup reused \`.desc\` - the package's faint 11px CAPTION style - for the PRIMARY
   label, and paired it with a \`.sub\` span that had no rule at all. Two inline spans with no line
   break and no hierarchy is why the menu read as
   "Intake catalogueintake-esm JSON for the whole result set". */
.xm-head {
  padding: 6px 10px 8px;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.04em;
  color: var(--faint);
  text-transform: uppercase;
  /* The scope is stated ONCE, here, instead of being repeated in all three descriptions. */
}
.xm {
  display: flex;
  flex-direction: column;
  gap: 2px;
  /* Never wider than the component it lives in: at a 320px mount the menu still fits. */
  max-width: min(340px, calc(100vw - 24px));
}
.xm-item {
  display: grid;
  /* fixed icon column | text | optional format marker */
  grid-template-columns: 22px minmax(0, 1fr) auto;
  gap: 10px;
  align-items: center;
  width: 100%;
  min-height: 44px; /* a comfortable touch target */
  padding: 7px 10px;
  border: none;
  border-radius: var(--r-sm);
  background: none;
  color: var(--text);
  font-family: inherit;
  text-align: left;
  cursor: pointer;
}
.xm-item:hover {
  background: var(--surface-2);
}
.xm-item:focus-visible {
  outline: 2px solid var(--accent);
  outline-offset: -2px;
}
.xm-ic {
  display: inline-grid;
  place-items: center;
  color: var(--accent);
}
.xm-text {
  display: grid; /* label and description on their OWN lines - the actual bug */
  gap: 1px;
  min-width: 0;
}
.xm-label {
  font-size: 13px;
  font-weight: 600;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.xm-desc {
  font-size: 11px;
  color: var(--faint);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.xm-fmt {
  padding: 1px 6px;
  border: 1px solid var(--border);
  border-radius: 999px;
  background: var(--surface-2);
  color: var(--faint);
  font-family: var(--mono);
  font-size: 9.5px;
  font-weight: 700;
  letter-spacing: 0.04em;
}

.pop-sep {
  height: 1px;
  background: var(--border);
  margin: 5px 2px;
}
.pop-item.check.on .tick {
  margin-left: auto;
  color: var(--accent);
  font-weight: 700;
  display: inline-grid;
  place-items: center;
}

/* editors */
.editor {
  width: 300px;
  padding: 12px;
}
.editor h5 {
  margin: 0 0 10px;
  font-size: 13px;
  display: flex;
  align-items: center;
  gap: 7px;
}
.editor h5 .sub {
  font-weight: 500;
  color: var(--faint);
  font-size: 11px;
}
.editor .modes {
  display: flex;
  gap: 6px;
  margin: 10px 0;
}
.editor .modes button {
  flex: 1;
  height: 30px;
  border: 1px solid var(--border-2);
  background: var(--surface);
  border-radius: 6px;
  font-size: 11.5px;
  color: var(--dim);
  cursor: pointer;
  font-family: inherit;
  font-weight: 600;
}
.editor .modes button.on {
  background: var(--accent-soft);
  border-color: var(--accent);
  color: var(--accent);
}
.editor .modes button:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}
.editor .mode-help {
  font-size: 11px;
  color: var(--faint);
  line-height: 1.45;
  min-height: 30px;
  margin-bottom: 8px;
}
.editor .daterow {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 6px;
}
.editor .daterow label {
  font-size: 11px;
  color: var(--faint);
  width: 34px;
}
.editor .daterow input {
  flex: 1;
  height: 32px;
  border: 1px solid var(--border-2);
  background: var(--surface-2);
  color: var(--text);
  border-radius: 6px;
  padding: 0 8px;
  font-family: var(--mono);
  font-size: 12px;
  outline: none;
}
.editor .daterow input:focus {
  border-color: var(--accent);
  box-shadow: 0 0 0 3px var(--accent-soft);
}
.editor .daterow input.bad {
  border-color: var(--danger);
}
.editor .actions {
  display: flex;
  gap: 8px;
  margin-top: 10px;
}
.editor .actions .btn {
  flex: 1;
  justify-content: center;
}
.editor .preview {
  font-family: var(--mono);
  font-size: 10.5px;
  color: var(--accent);
  background: var(--surface-2);
  border-radius: 6px;
  padding: 7px 9px;
  margin-top: 8px;
  word-break: break-all;
}
.editor .err-line {
  font-size: 11px;
  color: var(--danger);
  margin-top: 6px;
  min-height: 14px;
}
.draw-hint {
  font-size: 11px;
  color: var(--faint);
  text-align: center;
  margin-top: 6px;
}
.bbox-fields {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 6px;
  margin-top: 8px;
}
.bbox-fields .f {
  display: flex;
  align-items: center;
  gap: 5px;
}
.bbox-fields .f label {
  font-size: 10px;
  color: var(--faint);
  width: 42px;
}
.bbox-fields input {
  width: 100%;
  height: 28px;
  border: 1px solid var(--border-2);
  background: var(--surface-2);
  color: var(--text);
  border-radius: 5px;
  padding: 0 6px;
  font-family: var(--mono);
  font-size: 11px;
  outline: none;
}
.bbox-fields input:focus {
  border-color: var(--accent);
}
.bbox-fields input.bad {
  border-color: var(--danger);
}
.map-overlay {
  position: absolute;
  inset: 0;
  cursor: crosshair;
}

/* dev notes drawer */
.notes-drawer {
  position: fixed;
  right: 0;
  bottom: 0;
  top: 56px;
  width: 372px;
  background: var(--surface);
  border-left: 1px solid var(--border);
  box-shadow: var(--shadow);
  transform: translateX(100%);
  transition: transform 0.3s;
  z-index: 60;
  display: flex;
  flex-direction: column;
}
.notes-drawer.show {
  transform: translateX(0);
}
.notes-drawer h4 {
  margin: 0;
  padding: 16px;
  border-bottom: 1px solid var(--border);
  font-size: 14px;
  display: flex;
  align-items: center;
}
.notes-drawer h4 .x {
  margin-left: auto;
  cursor: pointer;
  border: none;
  background: none;
  color: var(--faint);
}
.notes-list {
  overflow-y: auto;
  padding: 8px 16px 20px;
}
.nl {
  padding: 12px 0;
  border-bottom: 1px solid var(--border);
}
.nl .h {
  display: flex;
  align-items: center;
  gap: 9px;
  font-weight: 700;
  font-size: 13px;
  margin-bottom: 5px;
}
.nl .h .num {
  width: 19px;
  height: 19px;
  border-radius: 999px;
  background: var(--accent);
  color: #fff;
  display: grid;
  place-items: center;
  font-size: 11px;
  font-weight: 800;
  flex-shrink: 0;
}
.nl p {
  margin: 0;
  font-size: 12.5px;
  color: var(--dim);
  line-height: 1.55;
}

.freva-db ::-webkit-scrollbar {
  width: 10px;
  height: 10px;
}
.freva-db ::-webkit-scrollbar-thumb {
  background: var(--border-2);
  border-radius: 999px;
  border: 3px solid transparent;
  background-clip: padding-box;
}
@media (max-width: 1100px) {
  .details-panel {
    position: absolute;
    right: 0;
    top: 0;
    bottom: 0;
    z-index: 30;
    box-shadow: var(--shadow);
  }
}
@media (max-width: 680px) {
  .top {
    gap: 8px;
    padding: 0 10px;
  }
  .lens .k {
    display: none;
  } /* drop the "FLAVOUR" label; keep the value */
  .side {
    width: 208px;
  } /* results sidebar shrinks so content keeps room */
  .center-fixed {
    padding: 12px 12px 0;
  }
  .results-scroll {
    padding: 0 12px 92px;
  }
  .details-panel {
    width: min(360px, calc(100vw - 24px));
  } /* the details overlay fits a phone (still 0 when collapsed) */
  /* The panel controls (Select all / View / Details) reserve an invisible slot so Export doesn't shift
     when they fade in. On a phone that reserved slot wraps to a tall blank strip inside the result bar
     (the "weird big" section in Overview). Drop the reservation here - the controls still show when
     active (file panel scrolled into view). */
  .res-bar {
    padding: 8px 10px;
    gap: 8px;
  }
  .panelctl:not(.in) {
    display: none;
  }
}
@media (max-width: 460px) {
  .brand span {
    display: none;
  } /* just the mark on very small screens */
  .top {
    gap: 6px;
    padding: 0 8px;
  }
  .center-fixed {
    padding: 10px 10px 0;
  }
  .results-scroll {
    padding: 0 10px 92px;
  }
}
@media (max-width: 560px) {
  .fdb-app:not(.side-collapsed) .side {
    position: absolute;
    left: 0;
    top: 0;
    bottom: 0;
    z-index: 25;
    box-shadow: var(--shadow);
  }
}
@media (prefers-reduced-motion: reduce) {
  .sk {
    animation: none;
  }
  .querying .bar::after {
    animation: none;
  }
}

/* Theme flip: the controller sets data-notransition around the data-theme swap so the
   variable re-resolve is ONE style pass instead of thousands of simultaneous per-node
   background/color animations (the measured cause of the toggle stutter). */
.freva-db[data-notransition],
.freva-db[data-notransition] * {
  transition: none !important;
}

/* Incremental long lists: the IO sentinel is invisible; the no-IO fallback button
   (also the deterministic path in tests) looks like the quiet inline affordances. */
.chunk-sentinel {
  height: 1px;
}
.chunk-more {
  display: block;
  width: 100%;
  padding: 7px 10px;
  margin: 2px 0;
  border: 1px dashed var(--border-2);
  border-radius: var(--r-sm);
  background: none;
  color: var(--dim);
  font: inherit;
  font-size: 12px;
  cursor: pointer;
}
.chunk-more:hover {
  background: var(--surface-2);
  color: var(--text);
}

/* Details partial-failure retry (extends the base .partial-flag rule above) */
.partial-flag {
  display: flex;
  align-items: center;
  gap: 8px;
}
.btn.sm {
  padding: 3px 9px;
  font-size: 11.5px;
}

/* Format thumbnails: the leading tile for zarr/nc/grib rows/cards. Other extensions
   keep the generic .ext text tile. The brand mark sits on a white chip so the fixed-palette logos
   (netCDF/GRIB/Intake are dark) read on light AND dark result cards. */
.ftile {
  width: 30px;
  height: 30px;
  flex-shrink: 0;
  display: grid;
  place-items: center;
  line-height: 0;
  background: #fff;
  border: 1px solid var(--border);
  border-radius: 7px;
}
.gcard .ftile {
  width: 30px;
  height: 30px;
}
/* Small white chip for brand logos shown inline in menus/buttons (Export \u25BE, Details downloads). */
.brand-chip {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
  background: #fff;
  border: 1px solid var(--border);
  border-radius: 5px;
  padding: 2px;
  line-height: 0;
}

/* Manual load-next with a proportion bar (no scroll auto-load: cheaper on Solr, no
   jank at thousands of rendered rows). */
.more-loader {
  margin: 14px 0 0;
  padding: 12px 14px;
  border: 1px solid var(--border);
  border-radius: var(--r);
  background: var(--surface);
}
.more-info {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  font-size: 12.5px;
  color: var(--dim);
}
.more-pct {
  font-family: var(--mono);
  font-size: 11px;
  color: var(--faint);
}
.more-bar {
  height: 4px;
  background: var(--surface-3);
  border-radius: 2px;
  overflow: hidden;
  margin: 8px 0 10px;
}
.more-bar-fill {
  height: 100%;
  background: var(--accent);
  border-radius: 2px;
  transition: width 0.25s ease;
}
.more-loader .load-next {
  margin: 0;
}

/* One loading language: the shared inline spinner primitive. */
.spin {
  width: 14px;
  height: 14px;
  border: 2px solid var(--border-2);
  border-top-color: var(--accent);
  border-radius: 999px;
  display: inline-block;
  vertical-align: -2px;
  animation: fdb-spin 0.7s linear infinite;
}
@keyframes fdb-spin {
  to {
    transform: rotate(360deg);
  }
}
.fdb-app[data-reduced-motion="true"] .spin {
  animation: none;
}

/* Flavour (naming) change: a clean spinner veil over the sidebar while labels/counts re-fetch. */
.side {
  position: relative;
}
.side-flavour-veil {
  position: absolute;
  inset: 0;
  display: none;
  place-items: center;
  z-index: 5;
  background: color-mix(in srgb, var(--surface) 45%, transparent);
}
.side-flavour-veil .spin {
  width: 22px;
  height: 22px;
  border-width: 2.5px;
}
.fdb-app.flavour-loading .side-flavour-veil {
  display: grid;
}
.fdb-app.side-collapsed .side-flavour-veil {
  display: none;
}

/* Chip/mode/diff tags replace the removed \`\xB7\` separators with quiet grouping. */
.chip-tag {
  font-size: 10px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  opacity: 0.72;
  margin-left: 2px;
}
.mode-tag {
  font-size: 10px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: var(--faint);
  margin-left: 8px;
  font-family: var(--mono);
}
.varchip .vc-n {
  color: var(--accent);
  font-weight: 700;
  margin-left: 6px;
}

/* Config: brand description in the results scope line */
.scope-desc {
  font-size: 11.5px;
  color: var(--faint);
  margin-left: 2px;
}

/* Value-first main search dropdown */
.vsearch-pop {
  display: none;
  z-index: 60;
  background: var(--surface);
  border: 1px solid var(--border-2);
  border-radius: var(--r);
  box-shadow: var(--shadow-lg, 0 12px 32px rgba(0, 0, 0, 0.4));
  max-height: 340px;
  overflow-y: auto;
  padding: 5px;
}
.vsearch-pop.show {
  display: block;
}
.vs-item {
  display: flex;
  align-items: center;
  gap: 9px;
  padding: 7px 9px;
  border-radius: 7px;
  cursor: pointer;
}
.vs-item.hl,
.vs-item:hover {
  background: var(--accent-soft);
}
.vs-badge {
  font-size: 10px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.03em;
  color: var(--accent);
  background: var(--accent-soft);
  padding: 2px 7px;
  border-radius: 999px;
  flex-shrink: 0;
}
.vs-val {
  font-family: var(--mono);
  font-size: 12.5px;
  font-weight: 600;
  color: var(--text);
  flex-shrink: 0;
}
.vs-desc {
  font-size: 11.5px;
  color: var(--dim);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.vs-cnt {
  margin-left: auto;
  font-family: var(--mono);
  font-size: 10.5px;
  color: var(--faint);
  flex-shrink: 0;
}
.vs-empty {
  padding: 10px 12px;
  font-size: 12.5px;
  color: var(--faint);
}

/* Footer console + toasts */
.status {
  display: flex;
  align-items: center;
  gap: 9px;
  padding: 0 14px;
  user-select: none;
}
.status-dot {
  width: 8px;
  height: 8px;
  border-radius: 999px;
  flex-shrink: 0;
  background: var(--faint);
}
.status-dot.info {
  background: var(--accent);
}
.status-dot.success {
  background: var(--good);
}
.status-dot.warn {
  background: var(--warn);
}
.status-dot.error {
  background: var(--danger);
}
/* The footer message itself is coloured by severity (green routine/ok, yellow warning, red
   error), so activity reads at a glance without an event-log panel. */
.status-msg.info,
.status-msg.success {
  color: var(--good);
}
.status-msg.warn {
  color: var(--warn);
}
.status-msg.error {
  color: var(--danger);
}
.status .spacer {
  flex: 1;
}
.log-toggle {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  height: 22px;
  padding: 0 8px;
  border-radius: 6px;
  border: 1px solid var(--border);
  background: var(--surface-2);
  color: var(--dim);
  font-size: 11px;
  cursor: pointer;
}
.log-toggle:hover,
.log-toggle.on {
  color: var(--text);
  border-color: var(--border-2);
}
.log-count {
  font-family: var(--mono);
  font-size: 10.5px;
}

.console-panel {
  position: absolute;
  left: 0;
  right: 0;
  bottom: 36px;
  z-index: 55;
  display: none;
  max-height: 42%;
  background: var(--surface);
  border-top: 1px solid var(--border-2);
  box-shadow: 0 -10px 30px rgba(0, 0, 0, 0.35);
  flex-direction: column;
}
.console-panel.show {
  display: flex;
}
.console-head {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 8px 14px;
  border-bottom: 1px solid var(--border);
  flex-shrink: 0;
}
.console-title {
  font-weight: 700;
  font-size: 12.5px;
}
.console-cap {
  font-family: var(--mono);
  font-size: 10.5px;
  color: var(--faint);
}
.console-clear {
  margin-left: auto;
  height: 24px;
  padding: 0 10px;
  border-radius: 6px;
  border: 1px solid var(--border-2);
  background: var(--surface-2);
  color: var(--dim);
  font-size: 11.5px;
  cursor: pointer;
}
.console-clear:hover {
  color: var(--text);
}
.console-list {
  overflow-y: auto;
  padding: 4px 0;
}
.log-row {
  display: flex;
  align-items: baseline;
  gap: 10px;
  padding: 3px 14px;
  font-size: 12px;
}
.log-row:hover {
  background: var(--surface-2);
}
.log-time {
  font-family: var(--mono);
  font-size: 10.5px;
  color: var(--faint);
  flex-shrink: 0;
}
.log-sev {
  font-family: var(--mono);
  font-size: 9.5px;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  flex-shrink: 0;
  width: 54px;
}
.log-sev.info {
  color: var(--accent);
}
.log-sev.success {
  color: var(--good);
}
.log-sev.warn {
  color: var(--warn);
}
.log-sev.error {
  color: var(--danger);
}
.log-msg {
  color: var(--dim);
}
.log-row.error .log-msg {
  color: var(--text);
}
.log-empty {
  padding: 14px;
  color: var(--faint);
  font-size: 12.5px;
}

/* Toasts live TOP-RIGHT (out of the way of the results/terminal, which own the lower half) and
   slide in from the right rather than popping up from the bottom. */
.toast-host {
  position: absolute;
  right: 16px;
  top: 60px;
  z-index: 140;
  display: flex;
  flex-direction: column;
  gap: 8px;
  pointer-events: none;
}
/* Immediate, styled tooltip (replaces the slow native \`title\` popup - see components/tooltip.ts).
   Fixed-position so it is never clipped by a scroll container; flips/clamps to stay on screen. */
.fdb-tip {
  position: fixed;
  left: 0;
  top: 0;
  z-index: 1500;
  pointer-events: none;
  max-width: 280px;
  padding: 5px 9px;
  border-radius: 7px;
  font-size: 12px;
  line-height: 1.45;
  font-weight: 500;
  background: var(--surface-2);
  color: var(--text);
  border: 1px solid var(--border);
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
  white-space: normal;
  opacity: 0;
  transition: opacity 0.1s ease;
}
.fdb-tip.show {
  opacity: 1;
}
.toast {
  display: flex;
  align-items: flex-start;
  gap: 9px;
  width: 320px;
  max-width: calc(100vw - 32px);
  padding: 11px 12px;
  border-radius: 12px;
  background: var(--surface);
  border: 1px solid var(--border-2);
  box-shadow:
    0 10px 30px rgba(0, 0, 0, 0.18),
    0 1px 0 rgba(255, 255, 255, 0.03) inset;
  color: var(--text);
  font-size: 12.5px;
  line-height: 1.45;
  pointer-events: auto;
  cursor: pointer;
  opacity: 0;
  transform: translateX(12px);
  transition:
    opacity 0.18s ease,
    transform 0.18s ease;
}
.toast.in {
  opacity: 1;
  transform: translateX(0);
}
/* a status dot instead of a left bar - reads faster and keeps the card shape clean */
.toast::before {
  content: "";
  flex: 0 0 auto;
  width: 8px;
  height: 8px;
  margin-top: 5px;
  border-radius: 50%;
  background: var(--accent);
}
.toast.success::before {
  background: var(--good);
}
.toast.warn::before {
  background: var(--warn);
}
.toast.error::before {
  background: var(--danger);
}
.toast.info::before {
  background: var(--accent);
}
.toast-msg {
  flex: 1;
}
.fdb-app[data-reduced-motion="true"] ~ .toast-host .toast,
.freva-db[data-reduced-motion="true"] .toast {
  transition: none;
}

/* Metadata-focused block controls: sort, collapse, additional */
.fcard-h .drag-grip {
  cursor: grab;
  color: var(--dim);
  font-size: 13px;
  margin-right: 2px;
  user-select: none;
  padding: 0 2px;
  opacity: 0.9;
  appearance: none;
  background: none;
  border: 0;
  font-family: inherit;
  line-height: 1;
}
.fcard-h .drag-grip:hover {
  color: var(--text);
  opacity: 1;
}
.fcard-h .drag-grip-fixed {
  cursor: default;
  opacity: 0.4;
}
.fcard-h .drag-grip-fixed:hover {
  color: var(--dim);
  opacity: 0.4;
}
.fcard-h button.drag-grip:focus-visible {
  outline: 2px solid var(--accent);
  outline-offset: 1px;
  border-radius: 3px;
  color: var(--text);
}
.fcard.dragging {
  opacity: 0.55;
  outline: 2px solid var(--accent);
  outline-offset: -2px;
}
.fcard.resizing {
  outline: 1px dashed var(--accent);
  outline-offset: -1px;
}
body.fdb-dragging {
  cursor: grabbing;
  user-select: none;
}
body.fdb-dragging * {
  user-select: none !important;
}
.fcard.collapsed .fcard-vals,
.fcard.collapsed .within {
  display: none;
}
.fcard-h .fh-label {
  font-weight: 600;
}
.ov-addrow {
  grid-column: 1 / -1;
}
.ov-addbtn {
  width: 100%;
  height: 40px;
  border-radius: var(--r-sm);
  border: 1px dashed var(--border-2);
  background: transparent;
  color: var(--dim);
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
}
.ov-addbtn:hover {
  color: var(--text);
  border-color: var(--accent);
}

/* Overview card drag-resize handle */
.fcard {
  position: relative;
}
.fcard-resize {
  position: absolute;
  right: 3px;
  bottom: 3px;
  width: 14px;
  height: 14px;
  cursor: ew-resize;
  opacity: 0;
  z-index: 2;
  appearance: none;
  border: 0;
  padding: 0;
  background: linear-gradient(
    135deg,
    transparent 55%,
    var(--border-2) 55%,
    var(--border-2) 66%,
    transparent 66%,
    transparent 78%,
    var(--border-2) 78%,
    var(--border-2) 88%,
    transparent 88%
  );
  transition: opacity 0.12s;
}
.fcard:hover .fcard-resize {
  opacity: 1;
}
.fcard-resize:focus-visible {
  opacity: 1;
  outline: 2px solid var(--accent);
  outline-offset: 1px;
}
.fcard.collapsed .fcard-resize {
  display: none;
}

/* Spinners: search-in-flight + export-in-progress */
.res-spin {
  display: none;
  align-items: center;
  margin-left: 2px;
}
.res-spin.show {
  display: inline-flex;
}
.iconbtn.busy {
  position: relative;
  color: transparent;
}
.iconbtn.busy svg {
  visibility: hidden;
}
.iconbtn.busy::after {
  content: "";
  position: absolute;
  inset: 0;
  margin: auto;
  width: 15px;
  height: 15px;
  border: 2px solid var(--border-2);
  border-top-color: var(--accent);
  border-radius: 999px;
  animation: fdb-spin 0.7s linear infinite;
}
.freva-db[data-reduced-motion="true"] .iconbtn.busy::after {
  animation: none;
}

/* read-only time/bbox prefix (always first) */
/* Read-only (time/bbox/flavour) tokens are deliberately NOT blue/amber - those colours mean
   "you typed this, you can edit it". They're also NOT boxed: a bordered chip read as an
   autocomplete row. They're plain, dimmed and italic - quietly present, clearly not editable. */
.tf-tok {
  white-space: nowrap;
  font-style: italic;
  opacity: 0.72;
}
.tf-k,
.tf-eq,
.tf-v {
  color: #7f8da3;
}

/* app-level Help panel (top bar) */
.help-pop {
  display: none;
  position: fixed;
  right: 18px;
  top: 62px;
  z-index: 130;
  width: min(400px, calc(100vw - 36px));
  padding: 16px;
  border-radius: var(--r);
  border: 1px solid var(--border-2);
  background: var(--surface);
  box-shadow: 0 24px 60px rgba(0, 0, 0, 0.34);
}
.help-pop.show {
  display: block;
}
.help-head {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 10px;
  color: var(--text);
}
.help-head .t {
  font-weight: 700;
  font-size: 14px;
  flex: 1;
}
.help-x {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 26px;
  height: 26px;
  border: none;
  background: none;
  color: var(--dim);
  border-radius: 6px;
  cursor: pointer;
}
.help-x:hover {
  background: var(--surface-2);
  color: var(--text);
}
.help-pop p {
  margin: 7px 0;
  font-size: 12.5px;
  line-height: 1.55;
  color: var(--dim);
}
.help-h2 {
  margin-top: 12px;
  font-weight: 700;
  font-size: 12.5px;
  color: var(--text);
}
.help-code {
  margin: 6px 0;
  padding: 8px 10px;
  border-radius: 7px;
  background: var(--surface-2);
  border: 1px solid var(--border-2);
  color: var(--text);
  font-family: var(--mono);
  font-size: 12px;
  overflow-x: auto;
}
.help-dim {
  color: var(--dim);
  font-size: 11.5px;
}
.help-link {
  display: inline-block;
  margin-top: 8px;
  color: var(--accent);
  font-size: 12.5px;
  text-decoration: none;
}
.help-link:hover {
  text-decoration: underline;
}
.spacer {
  flex: 1 1 0;
  min-width: 0;
} /* above the editable kwargs */

/* Sidebar - one "Filter" header, sections named by what they are, each with its
   own search + capped scroll area. */

/* Pinned, with its rule.
 *
 * "Filter" and its count are the sidebar's own header - what the panel below IS, and the control
 * that clears it - so they stay put while the facets scroll under them. Scrolling a header out of
 * a panel that has its own scrollbar loses the one clear-everything affordance exactly when a long
 * filtered list is the reason you want it.
 *
 * The negative margins and the matching \`top\` pull the sticky edge out to \`.side-scroll\`'s border
 * box, past its 10px padding: without them the head would stop 10px down and content would keep
 * scrolling visibly through the gap above it. The background is opaque for the same reason - a
 * transparent sticky header is a sticky header with the list running through it. */
.side-filterhead {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 12px 2px 10px;
  margin: -10px -10px 4px;
  padding-left: 12px;
  padding-right: 12px;
  border-bottom: 1px solid var(--border-2);
  position: sticky;
  top: -10px;
  z-index: 3;
  background: var(--surface);
}
.sf-title {
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--text);
}
.sf-badge {
  display: inline-grid;
  place-items: center;
  min-width: 20px;
  height: 20px;
  padding: 0 6px;
  border-radius: 4px;
  background: var(--accent);
  color: #fff;
  font-size: 11px;
  font-weight: 700;
  font-family: var(--mono);
  border: none;
  cursor: pointer;
  position: relative;
  transition: background-color 0.12s;
}
/* The global Filter total's hover treatment lives at the end of this sheet: a real element swap in
   a NEUTRAL colour. A red \`::after\` cross over a merely-transparent number puts two crosses on
   screen, and takes its red from \`--danger\`, which in a red-branded deployment is the accent. */

/* sections: a hairline rule between them, chevron on the RIGHT (the e-commerce convention) */
.facet {
  border-bottom: 1px solid var(--border-2);
  border-radius: 0;
}
.facet-head {
  height: auto;
  min-height: 44px;
  padding: 10px 10px;
  gap: 10px;
  border-radius: 6px;
}
.facet-head:focus-visible {
  outline-offset: -2px;
} /* inset, so it never lands on the text */
.special:focus-visible {
  outline-offset: -2px;
}
.facet-head:hover {
  background: none;
}
.facet-head:hover .fh-label {
  color: var(--accent);
}
.fh-text {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 1px;
  min-width: 0;
  flex: 1;
}
/* the selected values, readable WITHOUT expanding the section */
.fh-sel {
  font-size: 11px;
  color: var(--dim);
  max-width: 100%;
  white-space: normal;
  overflow-wrap: anywhere;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}
.facet.open .fh-sel {
  display: none;
} /* redundant once the values are visible below */
.facet-head .chev {
  margin-left: 0;
  transition: transform 0.2s;
}
.facet.open .facet-head .chev {
  transform: rotate(90deg);
} /* right -> down, not left */
.facet-head .badge,
.facet-head .fh-count {
  margin-left: 0;
}

/* per-facet search + a capped, scrollable value list (a facet may hold thousands of values) */
.fval-search {
  width: 100%;
  margin: 2px 0 8px;
  padding: 7px 9px;
  border-radius: var(--r-sm);
  border: 1px solid var(--border-2);
  background: var(--surface-2);
  color: var(--text);
  font-size: 12px;
}
.fval-search:focus {
  outline: none;
  border-color: var(--accent);
}
.fval-list {
  max-height: 240px;
  overflow-y: auto;
  overscroll-behavior: contain;
  display: flex;
  flex-direction: column;
  gap: 1px;
  padding-right: 2px;
}
.fval-list::-webkit-scrollbar {
  width: 8px;
}
.fval-list::-webkit-scrollbar-thumb {
  background: var(--border-2);
  border-radius: 4px;
}

/* interactive map: an on-demand upgrade over the instant SVG */
.map-slot {
  position: relative;
}
.map-zoom {
  position: absolute;
  right: 8px;
  top: 8px;
  z-index: 2;
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 4px 8px;
  border-radius: 999px;
  border: 1px solid var(--border-2);
  background: var(--surface);
  color: var(--dim);
  font-size: 11px;
  cursor: pointer;
  box-shadow: var(--shadow-sm);
}
.map-zoom:hover {
  color: var(--text);
  border-color: var(--accent);
}
.map-zoom:disabled {
  cursor: default;
  opacity: 0.7;
}
/* once Leaflet is mounted the SVG underneath is redundant */
.minimap.has-leaflet > svg {
  display: none;
}
.lmap {
  width: 100%;
  height: 220px;
  border-radius: var(--r-sm);
  overflow: hidden;
}
.miniwrap .lmap {
  height: 180px;
}
.leaflet-container {
  background: var(--surface-2);
  font: inherit;
}

/* Metadata view (overview) */
.fcard.collapsed {
  height: auto !important;
  align-self: start;
  min-height: 0;
}
.fcard.collapsed .fcard-vals,
.fcard.collapsed .within,
.fcard.collapsed .fcard-special-body {
  display: none;
}
/* a minimized card must not keep its 2-block height */
.fcard.collapsed[data-rows="2"] {
  height: auto !important;
}

/* (a wider card gets more columns automatically from the auto-fill grid on .fcard-vals) */

/* resize grip: now a 2-D handle (sideways AND up), so say so */
.fcard-resize {
  cursor: nwse-resize;
}

/* the sort control shows its mode, not just an icon */
.sortbtn {
  width: auto;
  gap: 4px;
  padding: 0 6px;
}
.sortlbl {
  font-family: var(--mono);
  font-size: 10px;
  letter-spacing: 0.02em;
}

/* time / bbox cards wear the same chrome as the facet cards, and their editor is always visible */
.fcard.fcard-sp .badge.on {
  background: var(--accent);
  color: #fff;
}
.fcard-special-body {
  flex: 1;
  min-height: 0;
  overflow: auto;
  padding: 6px 8px 8px;
}
/* Centred only while the content FITS. \`justify-content: center\` on a scroll container pushes
   overflow off both ends, and the start-side overflow is unreachable - which is how the From row
   ended up hidden beneath the card header. \`safe center\` falls back to start-alignment the moment
   the content is taller than the body. */
.fcard-special-body.time-body {
  display: flex;
  flex-direction: column;
  justify-content: safe center;
}
/* The inline editor is stretched to height:100% (so the bbox map can fill its card), which left
   nothing for the body's justify-content to centre. Centre the time picker's rows WITHIN that
   full-height editor instead. Only time - the bbox editor wants its map to fill. */
.fcard-special-body.time-body .editor.inline {
  justify-content: safe center;
}
/* the embedded editors size to the CARD instead of overflowing it */
.fcard-special-body .editor {
  border: none;
  box-shadow: none;
  padding: 0;
  background: none;
  width: auto;
}
.fcard-special-body .editor h5 {
  display: none;
} /* the card header already says what this is */
.fcard-special-body .minimap {
  width: 100%;
}
.fcard-special-body .minimap > svg {
  width: 100%;
  height: auto;
  display: block;
}

/* Leaflet attribution: required for OSM, but it was dominating a small card. Keep it, shrink it. */
.leaflet-control-attribution {
  font-size: 9px !important;
  padding: 0 4px !important;
  line-height: 1.4;
  background: rgba(255, 255, 255, 0.72) !important;
}
.leaflet-control-attribution a {
  color: var(--dim) !important;
  text-decoration: none;
}
.leaflet-control-zoom {
  margin: 6px !important;
}
.leaflet-control-zoom a {
  width: 22px !important;
  height: 22px !important;
  line-height: 22px !important;
  font-size: 14px !important;
}

/* overview: minimized rows must close up, not leave a hole */
.fcard.collapsed {
  height: auto !important;
  align-self: start;
  min-height: 0;
}

/* inline (in-card) editors: everything visible, nothing clipped */
.fcard-special-body {
  padding: 0 10px 10px;
}
.editor.inline {
  width: auto;
  padding: 0;
  border: none;
  box-shadow: none;
  background: none;
  gap: 6px;
}
.editor.inline h5 {
  display: none;
} /* the card header already names it */
.editor.inline .preview {
  display: none;
} /* the terminal shows the query; a card has no room */
.editor.inline .mode-help {
  font-size: 10.5px;
  line-height: 1.35;
}
.editor.inline .daterow {
  gap: 6px;
}
.editor.inline .daterow input {
  min-width: 0; /* a grid item's default \`min-width: auto\` would let the field push the row wider */
}
.editor.inline .modes {
  gap: 4px;
}
.editor.inline .modes .btn {
  padding: 3px 8px;
  font-size: 11px;
}
.editor.inline .bbox-fields {
  gap: 5px;
}
.editor.inline .bbox-fields input {
  min-width: 0;
}
.editor.inline .draw-hint {
  font-size: 10.5px;
}
/* the map scales to the card instead of overflowing it */

/* time / bbox blocks: same family, and legible inside one block */
.fcard.fcard-sp .fcard-special-body {
  display: flex;
  flex-direction: column;
  min-height: 0;
  overflow: auto;
}
.editor.inline {
  display: flex;
  flex-direction: column;
  gap: 7px;
  height: 100%;
  min-height: 0;
}

/* the map takes the room that's left, so it fills the block instead of being a squashed strip */
.editor.inline .map-slot {
  flex: 1 1 auto;
  min-height: 130px;
  display: flex;
}
.editor.inline .minimap {
  flex: 1;
  display: flex;
  min-height: 130px;
}
.editor.inline .lmap {
  flex: 1 1 auto;
  height: auto;
  min-height: 130px;
}
.editor.inline .minimap > svg {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

/* bounds/dates: a tight 2-up grid rather than four stacked rows that overflow the card */
.editor.inline .bbox-fields {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 5px;
}
.editor.inline .bbox-fields .f {
  display: flex;
  align-items: center;
  gap: 4px;
}
.editor.inline .bbox-fields label {
  font-size: 10px;
  color: var(--faint);
  min-width: 34px;
}
.editor.inline .bbox-fields input {
  width: 100%;
  padding: 4px 6px;
  font-size: 11px;
}
/* THREE children, THREE columns: label, text field, calendar button.
   Declaring only \`34px 1fr\` puts the calendar button on an implicit second grid row, making each
   date row two lines tall. Two of those plus the mode buttons overflow the card body, and because
   the body centres its content the overflow goes off BOTH ends, clipping the From row under the
   card header. The third column is what holds it, not a taller card. */
.editor.inline .daterow {
  display: grid;
  grid-template-columns: 34px minmax(0, 1fr) auto;
  align-items: center;
}
.editor.inline .daterow label {
  font-size: 10px;
  color: var(--faint);
}
.editor.inline .daterow input {
  padding: 5px 7px;
  font-size: 11.5px;
}
/* the mode help is one line in a card - the full text lives in the popover editor */
.editor.inline .mode-help {
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
  font-size: 10px;
  color: var(--faint);
}
.editor.inline .draw-hint {
  display: none;
} /* the map itself says "drag" well enough in a card */

/* Leaflet: a container with a REAL height (never a % inside a flex chain), and the Draw/Pan
       toggle. A 0-height container is what makes Leaflet mis-tile and mis-map its coordinates. */
.lmap {
  position: relative;
  min-height: 130px;
  background: var(--surface-2);
  isolation: isolate; /* contain Leaflet's high pane z-indexes so the map never paints over the terminal (z-index 80) */
}
.lmap.drawing,
.lmap.drawing .leaflet-grab {
  cursor: crosshair;
}
/* While a rectangle is being dragged the pointer sweeps across the +/- controls and the attribution,
   and the browser treats that as a text selection - the controls light up with the selection
   highlight mid-gesture. Suppressing selection for the duration of Draw mode, on this map only,
   removes the highlight without touching hit-testing: the controls are still clickable, still
   focusable, still keyboard-activatable, and Pan mode selects text normally again. */
.lmap.drawing,
.lmap.drawing * {
  user-select: none;
  -webkit-user-select: none;
}
.lmap-mode {
  position: absolute;
  right: 8px;
  top: 8px;
  z-index: 500; /* above Leaflet panes */
  padding: 4px 9px;
  border-radius: 999px;
  border: 1px solid var(--border-2);
  background: var(--surface);
  color: var(--dim);
  font-size: 11px;
  font-weight: 600;
  cursor: pointer;
  box-shadow: var(--shadow-sm);
}
.lmap-mode.on {
  border-color: var(--accent);
  color: var(--accent);
}
.lmap-mode:hover {
  color: var(--text);
}

/* Leaflet hygiene 
   Leaflet sizes and positions its tiles in JS assuming ITS OWN css. Application-wide resets that
   reach inside \`.leaflet-container\` are the classic cause of a mosaic-looking map, so we explicitly
   keep our resets out of it. */
.freva-db .leaflet-container,
.freva-db .leaflet-container * {
  box-sizing: content-box;
}
.freva-db .leaflet-container img {
  max-width: none !important;
  max-height: none !important;
}
.freva-db .leaflet-pane,
.freva-db .leaflet-tile,
.freva-db .leaflet-marker-icon {
  position: absolute;
}
.freva-db .leaflet-tile {
  padding: 0;
  border: 0;
}

/* Controls: two labelled TASK modes on top; view/details/export in the result bar.
   (Four icon-only buttons in a row answered three different questions at once.) */
.ctrl-cluster .ctrl {
  width: auto;
  height: 30px;
  padding: 0 11px;
  gap: 6px;
  border-radius: 8px;
  display: inline-flex;
  align-items: center;
}
.ctrl-lbl {
  font-size: 12.5px;
  font-weight: 650;
  letter-spacing: 0.005em;
}

/* status text, not a control - a pill shape here reads as clickable */
.scope-lbl {
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--faint);
}
.view-lbl {
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.07em;
  text-transform: uppercase;
  color: var(--faint);
}

/* labelled buttons in the result bar */
.tbtn {
  width: auto;
  height: 30px;
  padding: 0 10px;
  gap: 6px;
  border-radius: 8px;
  display: inline-flex;
  align-items: center;
  border: 1px solid transparent;
}
.tbtn:hover {
  background: var(--surface-2);
  color: var(--accent);
}
/* Export locked past the 100k ceiling: greyed out, not-allowed cursor, no hover lift. The click
   handler already refuses to export; this makes it LOOK unavailable. Hover still shows the tooltip. */
.tbtn.is-disabled {
  opacity: 0.4;
  cursor: not-allowed;
  color: var(--dim);
}
.tbtn.is-disabled:hover {
  background: transparent;
  color: var(--dim);
}
.tbtn[hidden] {
  display: none;
}
.tbtn.on {
  color: var(--accent);
  background: var(--surface-2);
  border-color: var(--border);
}
.tbtn-lbl {
  font-size: 12px;
  font-weight: 600;
}
@media (max-width: 760px) {
  .tbtn-lbl,
  .view-lbl,
  .scope-lbl {
    display: none;
  }
}

/* overview: share-of-result-set bar per value
   The bar is the value's share of the WHOLE result set (count / totalCount), so a bar means the
   same thing in every card and cards can be compared with each other. A per-card scale makes every
   card's top value look "full", so \`historical\` (17%) and \`cmip6\` (56%) read alike.

   Drawn as a tinted fill BEHIND the row via ::before - no extra DOM node, and it reads as a bar
   chart rather than an underline. \`--pct\` is set per row, by BOTH panels: the overview's cards and
   the sidebar's filter lists draw the same bar off the same helper, so one value is one length
   wherever a reader meets it. The positioning context is on \`.fval\` itself (see the base rule),
   not on \`.fcard .fval\`, so both panels can carry a bar. */
.fcard .fval {
  border-radius: 6px;
}
.fval.has-bar::before {
  content: "";
  position: absolute;
  z-index: -1;
  left: 0;
  top: 2px;
  bottom: 2px;
  width: var(--pct, 0%);
  min-width: 2px;
  border-radius: 5px;
  /* derived from the accent so re-theming (any --accent) recolours every bar in one place */
  background: linear-gradient(
    90deg,
    color-mix(in srgb, var(--accent) 18%, transparent),
    color-mix(in srgb, var(--accent) 6%, transparent)
  );
  transition: width 0.18s ease;
}
.fval.has-bar:hover::before {
  background: linear-gradient(
    90deg,
    color-mix(in srgb, var(--accent) 26%, transparent),
    color-mix(in srgb, var(--accent) 9%, transparent)
  );
}
.fval.sel.has-bar::before {
  background: linear-gradient(
    90deg,
    color-mix(in srgb, var(--accent) 36%, transparent),
    color-mix(in srgb, var(--accent) 13%, transparent)
  );
}
/* Dark mode: the same accent needs more alpha to read on the dark surface. Still derived from
   --accent, so a custom accent recolours the dark bars too. */
.freva-db[data-theme="night"] .fval.has-bar::before {
  background: linear-gradient(
    90deg,
    color-mix(in srgb, var(--accent) 36%, transparent),
    color-mix(in srgb, var(--accent) 15%, transparent)
  );
}
.freva-db[data-theme="night"] .fval.has-bar:hover::before {
  background: linear-gradient(
    90deg,
    color-mix(in srgb, var(--accent) 50%, transparent),
    color-mix(in srgb, var(--accent) 21%, transparent)
  );
}
.freva-db[data-theme="night"] .fval.sel.has-bar::before {
  background: linear-gradient(
    90deg,
    color-mix(in srgb, var(--accent) 64%, transparent),
    color-mix(in srgb, var(--accent) 27%, transparent)
  );
}
/* the count must stay readable where the bar runs under it */
.fcard .fval .n {
  position: relative;
  z-index: 1;
}

/* data-inspector: themed to the databrowser (blue), auto-following light/night
   The <data-inspector> (imported lazily) renders inside .freva-db, so our design tokens are in
   scope. We map them onto the component's public knobs (--di-*), so the modal follows the app theme
   instead of prefers-color-scheme. On top of that:
     \u2022 a solid blue header - there is no --di-header-bg knob, so we colour .di-header directly and
       flip its text to white (this mirrors what other embedders like grid-doctor do);
     \u2022 the Load button is inverted (white-on-blue) so it doesn't vanish into the header;
     \u2022 the xarray repr caps itself at max-width:700px, which reads as "left-aligned" in the wide
       modal - we lift the cap so metadata fills the width, like grid-doctor's docs.
   These header/xr overrides target the package's internal classes and are therefore version-coupled;
   they degrade gracefully (a class rename just falls back to the component's own defaults). */
.freva-db data-inspector {
  --di-bg: var(--surface);
  --di-fg: var(--text);
  --di-muted: var(--dim);
  --di-border: var(--border);
  --di-surface: var(--surface-2);
  --di-accent: var(--accent);
  /* The xarray repr colours default to a fixed LIGHT palette (--jp-* and white fallbacks), so
     the metadata table ignored the app theme. Map them onto our tokens so the table follows light/night. */
  --xr-font-color0: var(--text);
  --xr-font-color2: var(--dim);
  --xr-font-color3: var(--dim);
  --xr-border-color: var(--border);
  --xr-disabled-color: var(--dim);
  --xr-background-color: var(--surface);
  --xr-background-color-row-even: var(--surface);
  --xr-background-color-row-odd: var(--surface-2);
  /* The chunk-cube diagram (shown on an expanded variable) defaults to Freva's brown
     (#9b7a52). Shade it from our accent so it reads as the dominant blue, in both themes. */
  --xr-chunk-face: var(--accent);
  --xr-chunk-top: color-mix(in srgb, var(--accent) 68%, #fff);
  --xr-chunk-side: color-mix(in srgb, var(--accent) 80%, #000);
  --xr-chunk-edge: color-mix(in srgb, var(--accent) 38%, #fff);
}
/* For a direct zarr store the resolved URL IS the file path, so the component's "Zarr:" row
   just duplicates the path bar above it. Our inspector only ever loads zarr stores, so suppress it.
   (If server-side non-zarr->zarr conversion is ever wired, revisit - then the URLs genuinely differ.) */
.freva-db data-inspector .di-zarr-row {
  display: none !important;
}
.freva-db data-inspector .di-header {
  background: var(--accent);
  border-bottom: none;
}
.freva-db data-inspector .di-header .di-title,
.freva-db data-inspector .di-header .di-title-ico,
.freva-db data-inspector .di-header .di-pathbar-label,
.freva-db data-inspector .di-header .di-muted,
.freva-db data-inspector .di-header .di-close {
  color: #fff;
}
.freva-db data-inspector .di-header .di-close:hover {
  background: rgba(255, 255, 255, 0.18);
  color: #fff;
}
.freva-db data-inspector .di-header .di-btn-primary {
  background: #fff;
  border-color: #fff;
  color: var(--accent);
}
.freva-db data-inspector .di-header .di-btn-primary:hover:not(:disabled) {
  filter: none;
  background: rgba(255, 255, 255, 0.88);
}
.freva-db data-inspector .di-header .di-btn-split {
  border-left-color: rgba(0, 0, 0, 0.12);
}
/* Let the metadata fill the modal instead of the built-in 700px cap. */
.freva-db data-inspector .xr-wrap {
  max-width: none;
}
/* Production hardening. Each block names the defect it removes. */

/* Screen-reader-only utilities
   Used by the in-field search status and, when \`features.footer:false\`, by the status region.
   \`display:none\` / \`visibility:hidden\` would take the node OUT of the accessibility tree and
   silence the live region - which is the whole reason this class exists. */
.freva-db .sr-only,
.freva-db .sr-status {
  position: absolute;
  width: 1px;
  height: 1px;
  margin: -1px;
  padding: 0;
  overflow: hidden;
  clip: rect(0 0 0 0);
  clip-path: inset(50%);
  white-space: nowrap;
  border: 0;
}

/* Footer as an independent option
   The grid drops to two rows, so the strip consumes NO height (rather than being painted and then
   hidden, which still reserves its track). */
.freva-db .fdb-app.no-footer {
  grid-template-rows: auto 1fr;
}

/* Overlays are COMPONENT-scoped, not viewport-scoped
   All three overlay owners now append to \`.freva-db\` and are positioned absolutely in its
   coordinate space by anchor.ts. \`position: fixed\` here would re-introduce exactly the embedded-host
   bug that fix exists to remove (a transformed/contained ancestor changes what \`fixed\` resolves
   against, and an \`overflow:hidden\` mount clips whatever lands outside it). */
.freva-db .fdb-tip {
  position: absolute;
  /* A very long unbroken label - a deep path, an ensemble id - must not push the bubble past a
     viewport edge, and must not render as one unwrappable line. */
  max-inline-size: min(280px, calc(100vw - 16px));
  overflow-wrap: anywhere;
  word-break: normal;
}

/* Chips: one long value cannot own the row
   Without a bounded, ellipsised label an unbroken value stretched its chip past the available
   width, pushing Clear all and the Browse/Overview cluster off a phone screen. The full value stays
   available as the tooltip and the accessible name. */
.freva-db .chip {
  min-width: 0;
  max-width: 100%;
}
.freva-db .chip-label {
  min-width: 0;
  max-inline-size: min(22ch, 60vw);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
/* The NEGATIVE chip is neutral - dotted, hatched, struck on the value only. \`--danger\` red says
   nothing in a red-branded deployment, in greyscale or in forced colours. Its rules live at the end of this sheet with the rest of the language. */
/* An IMMUTABLE base-scope indicator. Not a button: there is nothing to click, and "Clear all"
   deliberately does not touch it. */
.freva-db .chip.scope {
  background: var(--surface-3);
  color: var(--dim);
  border: 1px solid var(--border);
  cursor: default;
  font-weight: 600;
}

/* The top row wraps DETERMINISTICALLY at phone widths
   Chips take a full-width row of their own; Clear all and the Browse/Overview cluster share the
   next one. Nothing is absolutely positioned, so they cannot overlap however long the labels get.
   An empty chip row collapses instead of leaving a blank strip. */
.freva-db .chips.empty {
  display: none;
}
@media (max-width: 430px) {
  .freva-db .toprow {
    flex-wrap: wrap;
    align-items: center;
    row-gap: 8px;
  }
  .freva-db .toprow > .chips {
    flex: 1 0 100%; /* own row */
    order: 1;
  }
  .freva-db .toprow > .clear-btn {
    order: 2;
  }
  .freva-db .toprow > .ctrl-cluster {
    order: 3;
    margin-left: auto; /* the following row, right-aligned - never overlapping the chips */
  }
  .freva-db .chip-label {
    max-inline-size: min(18ch, 52vw);
  }
}

/* The list header stays visible
   It sticks INSIDE \`.results-scroll\` (it deliberately stays out of \`.rows\`, whose children are
   counted by the incremental append). Overview/terminal content above it scrolls away first,
   because sticking only begins once the header reaches the top of the scroller. */
.freva-db .results-scroll .list-head {
  position: sticky;
  top: 0;
  z-index: 6; /* above rows, below the pickbar/popovers/toasts */
  /* Opaque: rows scrolling underneath a translucent header is unreadable. A z-index alone would
     only put the strip BEHIND the header. What holds is that there is no band for rows to occupy
     (see \`.results-scroll\`'s padding); this keeps whatever does pass under it hidden. */
  background: var(--surface-2);
  box-shadow: 0 1px 0 var(--border);
  /* SQUARE top corners. The header's background is opaque, but a rounded corner is not part of the
     background - it is a hole, and a row passing underneath shows through the two little curved
     wedges at the top left and top right. There is no honest way to round the corner of something
     other content slides beneath, so the corner goes rather than the opacity. \`.rows\` already
     squares its own top corners when the header is present, so the two still read as one table. */
  border-top-left-radius: 0;
  border-top-right-radius: 0;
}

/* The comparison matrix has a bounded height
   Growing with the number of differing fields would push the rest of the Details panel out of
   reach. A SHORT comparison still uses only the height it needs (max-height, not height). */
.freva-db .dscroll {
  max-height: clamp(180px, 38vh, 360px);
  overflow: auto; /* both axes */
}
.freva-db .dmatrix thead th {
  position: sticky;
  top: 0;
  z-index: 1;
  background: var(--surface-2); /* opaque, or the scrolled rows show through the header */
}

/* Include / exclude, side by side
   TWO sibling controls. Nesting a button inside the button-like value row would be invalid HTML and
   unreliable for AT and touch. */
.freva-db .fval-row {
  display: flex;
  align-items: center;
  gap: 2px;
  min-width: 0;
}
.freva-db .fval-row > .fval {
  flex: 1;
  min-width: 0;
}
.freva-db .fval-ex {
  flex-shrink: 0;
  width: 22px;
  height: 22px;
  border-radius: 5px;
  border: 1px solid transparent;
  background: none;
  color: var(--faint);
  font-family: inherit;
  font-size: 13px;
  line-height: 1;
  cursor: pointer;
  /* Compact on pointer devices: revealed on hover/focus-within, so the row stays calm. */
  opacity: 0;
  transition: opacity 0.12s;
}
.freva-db .fval-row:hover .fval-ex,
.freva-db .fval-row:focus-within .fval-ex,
.freva-db .fval-ex:focus-visible,
.freva-db .fval-ex.on {
  opacity: 1;
}
.freva-db .fval-ex:hover {
  background: color-mix(in srgb, var(--danger) 12%, transparent);
  color: var(--danger);
}
.freva-db .fval-ex.on {
  color: var(--danger);
  border-color: color-mix(in srgb, var(--danger) 45%, transparent);
  background: color-mix(in srgb, var(--danger) 12%, transparent);
}
/* An EXCLUDED value in a FACET LIST: struck through, \`!=\`-marked and latched with a dashed control.
   The strike belongs here - a value list is a set of things you are choosing between, and the line
   is what shows at a glance which ones are out. The top-level CHIPS are the opposite case: there
   the value IS the label you have to read, so those are left unstruck. */
.freva-db .fval.excl .nm {
  text-decoration: line-through;
  text-decoration-thickness: 1px;
}
.freva-db .fval.excl {
  color: var(--danger);
}
/* On a touch layout there is no hover, so the control must be permanently discoverable. */
@media (hover: none), (pointer: coarse) {
  .freva-db .fval-ex {
    opacity: 1;
  }
}

/* Selection cap */
.freva-db .cb.capped {
  opacity: 0.4;
  cursor: not-allowed;
}
.freva-db .pickbar .cnt.at-cap b {
  color: var(--warn);
}

/* Remote source-file list */
.freva-db .dl-pop {
  width: min(460px, calc(100% - 24px));
}
.freva-db .dl-head {
  font-weight: 700;
  font-size: 13px;
  padding: 4px 8px 2px;
}
.freva-db .dl-note {
  font-size: 11.5px;
  color: var(--dim);
  padding: 0 8px 8px;
  line-height: 1.5;
}
.freva-db .dl-list {
  max-height: 300px;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 2px;
}
.freva-db .dl-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 7px 8px;
  border-radius: 7px;
  color: var(--text);
  text-decoration: none;
  font-size: 12.5px;
  min-width: 0;
}
.freva-db .dl-item:hover {
  background: var(--surface-2);
}
.freva-db .dl-name {
  font-weight: 600;
  flex-shrink: 0;
  max-width: 45%;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.freva-db .dl-path {
  color: var(--faint);
  font-family: var(--mono);
  font-size: 10.5px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  direction: rtl; /* keep the informative TAIL of a long path visible */
  text-align: left;
}

/* Native date picker beside the text field
   The text input remains the source of truth (it is the only one that can express YYYY, YYYY-MM
   and open bounds); the native input exists solely to raise the platform calendar. */
.freva-db .date-pickwrap {
  position: relative;
  display: inline-flex;
  flex-shrink: 0;
}
.freva-db .date-pick {
  width: 30px;
  height: 30px;
  display: inline-grid;
  place-items: center;
  border-radius: var(--r-sm);
  border: 1px solid var(--border);
  background: var(--surface-2);
  color: var(--dim);
  cursor: pointer;
}
.freva-db .date-pick:hover {
  border-color: var(--border-2);
  color: var(--text);
}
/* Present for showPicker()/focus(), but never a second visible field or a tab stop. */
.freva-db .date-native {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  opacity: 0;
  pointer-events: none;
  border: 0;
  padding: 0;
}

/* Flavour control: a real control, not a caption
   Layout only - the switching logic was already correct. */
.freva-db .lens {
  height: 43px;
  min-width: 160px;
  padding: 0 12px;
  gap: 10px;
  background: var(--surface); /* opaque - it sits over the top bar, not in it */
  border-color: var(--border-2);
  box-shadow: 0 1px 2px rgba(16, 28, 52, 0.06);
}
.freva-db .lens .v {
  flex: 1;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  text-align: left;
}
.freva-db .lens svg {
  flex-shrink: 0;
  margin-left: auto; /* the caret stays pinned at the far edge as the value flexes */
}
/* The menu is never narrower than the control it drops from. */
.freva-db .lens-pop {
  min-width: 170px;
}
.freva-db .lens-pop .pop-item {
  padding: 9px 11px;
}
.freva-db .lens-pop .pop-item.on {
  background: var(--accent-soft);
  color: var(--accent);
  font-weight: 600;
}
@media (max-width: 680px) {
  /* Dropping the FLAVOUR caption is fine; a usable value width and touch target are not optional. */
  .freva-db .lens {
    min-width: 108px;
    height: 42px;
  }
}

/* Search is the primary control */
.freva-db .search input {
  height: 45px;
  font-size: 14.5px;
  background: var(--surface); /* opaque, so it reads as raised rather than as part of the bar */
  border-color: color-mix(in srgb, var(--accent) 35%, var(--border));
  padding-right: 38px; /* the reserved spinner slot */
}
.freva-db .search .ic {
  color: var(--accent);
}
.freva-db .search input::placeholder {
  /* Explicit colour AND opacity:1 - the UA default is a low-opacity render of the text colour,
     which lands well under 4.5:1 and differs between engines. */
  color: var(--faint);
  opacity: 1;
}
.freva-db .search input:focus {
  border-color: var(--accent);
  box-shadow: 0 0 0 3px var(--accent-soft);
}
/* The slot is ALWAYS reserved; only visibility changes, so showing the spinner shifts nothing. */
/* The field's right-hand furniture: the spinner's reserved slot and the keyboard hint, in one
   row. One element, so the input's right padding is a single number rather than a running sum. */
.freva-db .search-aside {
  position: absolute;
  right: 10px;
  top: 50%;
  transform: translateY(-50%);
  display: flex;
  align-items: center;
  gap: 8px;
  pointer-events: none; /* the field underneath stays clickable across its whole width */
}
.freva-db .search-spin {
  display: grid;
  place-items: center;
  visibility: hidden;
  color: var(--accent);
}
/* The shortcut the field actually answers to. Quiet by design - it is a reminder for the second
   visit, not a control - and it steps back further once the field is in use, where the caret is
   the thing to look at. */
.freva-db .search-kbd {
  display: flex;
  align-items: center;
  gap: 3px;
  transition: opacity 0.18s;
}
.freva-db .search-kbd kbd {
  font: inherit;
  font-size: 10.5px;
  line-height: 1;
  padding: 3px 5px;
  min-width: 17px;
  text-align: center;
  border-radius: 4px;
  border: 1px solid var(--border-2);
  background: var(--surface);
  color: var(--faint);
}
.freva-db .search:focus-within .search-kbd {
  opacity: 0.35;
}
.freva-db .search-spin.show {
  visibility: visible;
}
@media (prefers-reduced-motion: reduce) {
  /* Still visibly BUSY, just not spinning: a static ring rather than nothing at all. */
  .freva-db .search-spin .spin {
    animation: none;
    border-top-color: currentColor;
    opacity: 0.85;
  }
}

/* Small-text legibility
   Meaningful 9.5-10px labels move to 11px where the space exists. Hierarchy comes from weight,
   spacing and grouping - not from making text too faint to read. */
.freva-db .dmatrix thead th,
.freva-db .list-head,
.freva-db .dl-path,
.freva-db .fval .n {
  font-size: 11px;
}
.freva-db .scope-note,
.freva-db .more-info {
  font-size: 11.5px;
}
.freva-db input::placeholder,
.freva-db textarea::placeholder {
  color: var(--faint);
  opacity: 1;
}
/* Disabled text still has to be READ to be understood. 0.5 alpha on --dim does not clear 4.5:1. */
.freva-db .btn:disabled,
.freva-db [aria-disabled="true"] {
  opacity: 0.72;
}

/* 1,000-row interaction cost
   \`content-visibility\` lets the engine skip layout and paint for rows that are off screen.
   \`contain-intrinsic-size\` supplies a placeholder box so the scrollbar stays honest and the
   scroll position does not jump. Checked in Chromium against focus, keyboard navigation, the
   sticky list header and scrolling. Engines that ignore these properties simply render every row,
   exactly as before - the fallback is doing nothing. */
.freva-db .rows > .row {
  content-visibility: auto;
  contain-intrinsic-size: auto 48px;
}
.freva-db .grid > .gcard {
  content-visibility: auto;
  contain-intrinsic-size: auto 132px;
}
/* At or above the documented threshold (MANY_RESULTS_THRESHOLD = 500 loaded rows) the side panels
   stop ANIMATING their width. A width transition on a panel re-lays-out the centre column on every
   animation frame; with 1,000 rows in it, that is the measured cost of opening Details or
   collapsing the sidebar. The panels still change state instantly - only the tween is dropped. */
.freva-db.many-results .side,
.freva-db.many-results .details-panel {
  transition: none;
}
@media (max-width: 1100px) {
  /* Where the details panel is already an OVERLAY it does not reflow the grid, so its motion is
     compositor-only and can stay. */
  .freva-db.many-results .details-panel {
    transition: transform 0.18s ease;
  }
}

/* The top bar fits a phone
   At 320px the brand, the flavour control, the search field and four icon buttons cannot all keep
   their comfortable sizes. Rather than let the bar set a min-content width that pushes the entire
   app off screen, the negotiable parts give way in a defined order: the brand mark stays, the
   flavour control shrinks to a usable-but-tight touch target, and the search field keeps the rest. */
@media (max-width: 430px) {
  .freva-db .top {
    gap: 6px;
    padding: 0 8px;
  }
  .freva-db .lens {
    min-width: 84px;
    padding: 0 8px;
    gap: 4px;
  }
  .freva-db .search input {
    padding-left: 34px;
    padding-right: 32px;
  }
  .freva-db .search .ic {
    left: 10px;
  }
}

/* INCLUSION vs EXCLUSION - told apart by CHARACTER and SHAPE, not by colour

   The host's accent is configurable and may itself be red, so "accent = kept, red = removed" is not
   a distinction at all in some deployments - and it is none whatsoever in greyscale, in
   \`forced-colors\`, or to a colour-blind reader. Every surface carries the meaning twice:

     +N   kept      a FILLED badge, and the control that clears ONLY the kept values
     -N   removed   a DASHED outlined badge, and the control that clears ONLY the removed ones
     !=   an excluded VALUE, struck through, on a dotted hatched chip

   Colour still reinforces all of it; it is simply never the only carrier. */

/* The per-facet +N / -N clear buttons */
.freva-db .fh-count {
  position: relative;
  display: inline-grid;
  place-items: center;
  /* Sized by the COUNT either way, so swapping in the cross cannot make the header jump.
     \`--fb-ch\` is the count's character length, set when the button is built. */
  min-width: calc(var(--fb-ch, 2) * 1ch + 16px);
  height: 18px;
  padding: 0 6px;
  border-radius: 999px;
  border: 1px solid transparent;
  font-size: 10.5px;
  font-weight: 800;
  line-height: 16px;
  font-variant-numeric: tabular-nums;
  cursor: pointer;
  flex: none;
  transition: background-color 0.12s;
}
.freva-db .fh-count.fb-inc {
  background: var(--accent);
  color: #fff;
}
.freva-db .fh-count.fb-exc {
  background: transparent;
  /* \`currentColor\`, not the danger colour: in forced-colors the dash survives and the hue need not.
     The dashes ARE the signal. */
  border: 1px dashed currentColor;
  color: var(--text);
  font-weight: 700;
}
/* The count and the cross occupy the SAME grid cell and exactly one is rendered. Fading a cross in
   on top of a number leaves both readable at once, which is what "no overlap" rules out. */
.freva-db .fh-count > .fb-n,
.freva-db .fh-count > .fb-x {
  grid-area: 1 / 1;
}
.freva-db .fh-count > .fb-x {
  display: none;
  font-size: 13px;
  font-weight: 800;
  line-height: 1;
  /* NEUTRAL - the button's own text colour, never \`--danger\` and never the accent, both of which
     can be the same red in a themed deployment. */
  color: currentColor;
}
.freva-db .fh-count:hover > .fb-n,
.freva-db .fh-count:focus-visible > .fb-n {
  display: none;
}
.freva-db .fh-count:hover > .fb-x,
.freva-db .fh-count:focus-visible > .fb-x {
  display: block;
}
/* Hover changes what is WRITTEN, not what shape the badge is: solid stays solid, dashed dashed. */
.freva-db .fh-count.fb-inc:hover,
.freva-db .fh-count.fb-inc:focus-visible {
  background: color-mix(in srgb, var(--accent) 78%, var(--text));
  outline: none;
}
.freva-db .fh-count.fb-exc:hover,
.freva-db .fh-count.fb-exc:focus-visible {
  background: color-mix(in srgb, currentColor 10%, transparent);
  outline: none;
}

/* The facet header is a ROW of siblings, never nested buttons */
.freva-db .facet-head {
  display: flex;
  align-items: center;
  gap: 6px;
}
.freva-db .facet-head > .fh-toggle {
  flex: 1;
  min-width: 0;
  display: flex;
  align-items: center;
  gap: 8px;
  background: none;
  border: 0;
  padding: 0;
  margin: 0;
  font: inherit;
  color: inherit;
  text-align: left;
  cursor: pointer;
}
.freva-db .facet-head > .fh-toggle:focus-visible {
  outline: 2px solid var(--accent);
  outline-offset: 2px;
  border-radius: 6px;
}

/* The GLOBAL Filter total: ONE number, swapped for ONE cross */
.freva-db .sf-badge {
  position: relative;
  display: inline-grid;
  place-items: center;
  min-width: calc(var(--fb-ch, 1) * 1ch + 14px);
  height: 20px;
  padding: 0 6px;
  border-radius: 4px;
  background: var(--accent);
  color: #fff;
  font-size: 11px;
  font-weight: 800;
  font-variant-numeric: tabular-nums;
  cursor: pointer;
}
.freva-db .sf-badge > .sf-n,
.freva-db .sf-badge > .sf-x {
  grid-area: 1 / 1;
}
.freva-db .sf-badge > .sf-x {
  display: none;
  font-size: 14px;
  line-height: 1;
  color: currentColor;
}
.freva-db .sf-badge:hover > .sf-n,
.freva-db .sf-badge:focus-visible > .sf-n {
  display: none;
}
.freva-db .sf-badge:hover > .sf-x,
.freva-db .sf-badge:focus-visible > .sf-x {
  display: block;
}

/* An excluded VALUE, in the lists */
.freva-db .fval-row.excl .nm,
.freva-db .fval.excl .nm,
.freva-db .fval.excl .fv-t {
  text-decoration: line-through;
  text-decoration-thickness: 1px;
}
.freva-db .fval-row.excl .nm::before,
.freva-db .fval.excl .nm::before,
.freva-db .fval.excl .fv-t::before {
  content: "\\2260\\00a0";
  text-decoration: none;
  display: inline-block;
  font-weight: 700;
}
.freva-db .fval-row.excl .fval-ex,
.freva-db .fval-row .fval-ex[aria-pressed="true"] {
  border: 1px dashed currentColor;
  border-radius: 5px;
}

/* Negative top-level chips: neutral, dotted, hatched */
/* No \`--danger\`, no red, no accent. The chip reads as "removed" from its dotted outline, its \`NOT\`
   tag, its \`!=\` operator and a subtle static hatch - all theme-neutral, so the same treatment works
   in a dark theme, a light one and a red-branded deployment alike. */
.freva-db .chip.neg {
  border: 1px dotted var(--border-2);
  color: var(--text);
  background-color: var(--surface-2);
  /* Built from the TEXT colour at low alpha, so it follows the theme rather than carrying one of
     its own, and it is static - a moving pattern behind a label is unreadable. */
  background-image: repeating-linear-gradient(
    -45deg,
    color-mix(in srgb, var(--text) 9%, transparent) 0 1px,
    transparent 1px 6px
  );
}
.freva-db .chip.neg .chip-label {
  display: inline-flex;
  align-items: baseline;
  min-width: 0;
  overflow: hidden;
}
.freva-db .chip.neg .chip-k,
.freva-db .chip.neg .chip-op,
.freva-db .chip.neg .chip-tag,
.freva-db .chip.neg .x {
  color: var(--text);
  text-decoration: none;
}
/* The value is NOT struck through: the dotted outline, the hatch, \`NOT\` and \`!=\` carry the meaning,
   and an unstruck value stays legible - which matters most for the long ones. */
.freva-db .chip.neg .chip-v {
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.freva-db .chip.neg .chip-tag {
  background: none;
  border: 1px dotted currentColor;
  border-radius: 3px;
  padding: 0 3px;
  font-size: 9px;
  font-weight: 800;
  letter-spacing: 0.04em;
  opacity: 0.85;
}
.freva-db .chip.neg .x {
  opacity: 0.9;
}

@media (forced-colors: active) {
  /* No hatch survives forced colours, and none is needed: the dotted border, \`NOT\` and \`!=\` carry
     it on their own. */
  .freva-db .chip.neg {
    background-image: none;
    border-style: dotted;
  }
  .freva-db .fh-count.fb-inc {
    border: 1px solid CanvasText;
  }
  .freva-db .fh-count.fb-exc {
    border: 1px dashed CanvasText;
  }
  .freva-db .fval-row.excl .fval-ex {
    border-style: dashed;
  }
}

/* The File Inspector, modal. */

/*
 * The Inspector is a \`<dialog>\` opened with \`showModal()\`, so the browser puts
 * it in the top layer: laid out against the viewport, painted over everything,
 * with the rest of the document inert. What it needs from CSS is only that the
 * element inside stops trying to be the modal itself.
 *
 * The packaged element renders its own full-screen dimmer and centres a panel
 * inside it, with both the panel and its body sized in \`vh\`. Inside a dialog
 * that is a second modal inside the first, and the viewport units are measured
 * against the whole screen while the box they live in is not, which pushes the
 * header off the top and the actions off the bottom. So the dimmer becomes an
 * ordinary flex box, the dialog carries the geometry, and the body is the only
 * thing that scrolls.
 */
.freva-db.fdb-inspector-dialog[open] {
  display: flex;
  flex-direction: column;
  /* The panel sits in the middle of the frame; the frame is the whole band. */
  justify-content: center;
}
.freva-db.fdb-inspector-dialog {
  /* The component root's own \`position: relative; height: 100%\` are for a widget
     filling a mount, and this is a dialog sizing itself to its content. */
  position: fixed;
  height: auto;
  width: min(1100px, 96vw);
  max-width: 96vw;
  /*
   * A *definite* height, not a cap - and this is the load-bearing line.
   *
   * With \`max-height\` alone the dialog's height stays "auto, then clamped", and
   * an auto-height flex container does not make its items shrink: they lay out
   * at their content size and overflow the clamp. Everything below inherits
   * that indefiniteness, which is why the panel kept growing out of the box no
   * matter how many \`min-height: 0\`s went into the chain.
   *
   * So the dialog is a full-height transparent frame with the panel centred in
   * it, and every child below has something definite to resolve against.
   * Dynamic units, so a mobile browser's collapsing toolbars cannot hide a row.
   */
  height: 92dvh;
  max-height: 92dvh;
  padding: 0;
  border: 0;
  background: none;
  /* The dialog is the boundary; nothing inside it may paint past it. */
  overflow: hidden;
  color: inherit;
}
.freva-db.fdb-inspector-dialog::backdrop {
  /* The dim, drawn here rather than by the element. From the top layer it covers
     the whole viewport, which is the point: the header and the footer dim too. */
  background: rgba(15, 23, 42, 0.55);
}
/*
 * The custom element itself, which is a box like any other.
 *
 * \`<data-inspector>\` sits between the dialog and everything the element draws.
 * Left unstyled it is an ordinary auto-height block, and it is the first and
 * quietest break in the chain from the dialog's height down to the scrolling
 * body - a chain only as definite as its least definite link.
 */
.freva-db.fdb-inspector-dialog > data-inspector {
  display: flex;
  flex-direction: column;
  flex: 1 1 auto;
  min-height: 0;
}

/*
 * The element's dimmer becomes an ordinary box inside the dialog.
 *
 * It keeps its node - its key handler is bound to it - and loses everything
 * that would make it a modal in its own right: the fixed positioning, the
 * full-screen inset, the dim (the dialog's \`::backdrop\` draws that), the
 * padding, and \`align-items: center\`, which centres a panel taller than the box
 * and lets it overflow off the top of the screen.
 *
 * It is a *shrinkable* flex column, which is the part that matters: it is the
 * first link in the chain from the dialog's height down to the scrolling body,
 * and a link that cannot shrink breaks the whole chain.
 */
.freva-db.fdb-inspector-dialog .di-backdrop {
  position: static;
  inset: auto;
  z-index: auto;
  padding: 0;
  background: none;
  display: flex;
  flex-direction: column;
  align-items: stretch;
  justify-content: center;
  flex: 1 1 auto;
  min-height: 0;
}
/*
 * The panel is the dialog's one flex item and shrinks with it.
 *
 * Not \`max-height: 100%\`: the dialog's height is \`auto\` capped by \`max-height\`,
 * and a percentage against an indefinite height resolves to nothing at all.
 * \`flex: 1 1 auto\` with \`min-height: 0\` is what actually makes a flex item
 * smaller than its content - the automatic minimum size is the reason panels
 * like this one grow through the bottom of the box that holds them.
 */
.freva-db.fdb-inspector-dialog .di-modal {
  width: 100%;
  /* Grows to its content and stops at the frame - now that the frame is
     definite, \`100%\` is a real number and the panel can be smaller than it. */
  flex: 0 1 auto;
  min-height: 0;
  max-height: 100%;
}
/*
 * The one unclassed box in the chain.
 *
 * Between the panel and the scrolling body sits \`#nc-tabs-wrap\`, an ordinary
 * block. A block does not shrink, so the flex chain stopped there and the body
 * grew to its content with the panel following it out of the dialog. Making it
 * a shrinkable flex column is what connects the dialog's height to the body's.
 */
.freva-db.fdb-inspector-dialog #nc-tabs-wrap {
  display: flex;
  flex-direction: column;
  flex: 1 1 auto;
  min-height: 0;
}
.freva-db.fdb-inspector-dialog #nc-tabs-wrap[hidden] {
  display: none;
}

/*
 * The header and the action row stay; only the body scrolls. \`min-height: 0\` is
 * the load-bearing part - without it the flex item refuses to shrink below its
 * content and the panel grows past the dialog again.
 */
.freva-db.fdb-inspector-dialog .di-body {
  flex: 1 1 auto;
  min-height: 0;
  max-height: none;
}
`;var Dn="freva.db.theme",On="freva.db.layout",In="freva.db.view",Ln="freva.db.sidebar";function St(e){try{return window.localStorage.getItem(e)}catch{return null}}function Ct(e,t){try{window.localStorage.setItem(e,t)}catch{}}function Bn(){return St(Dn)==="day"?"day":"night"}function Pn(e){Ct(Dn,e)}function Rn(e="results"){let t=St(On);return t==="overview"?"overview":t==="results"?"results":e}function Fn(e){Ct(On,e)}function zn(){return St(In)==="grid"?"grid":"list"}function Nn(e){Ct(In,e)}function Hn(e=!1){let t=St(Ln);return t==="collapsed"?!0:t==="open"?!1:e}function Un(e){Ct(Ln,e?"collapsed":"open")}var Gn="freva.db.overview";function Qn(){let e=St(Gn);if(!e)return null;let t=o=>{let r={};if(o&&typeof o=="object")for(let[n,i]of Object.entries(o)){let l=typeof i=="number"?i:Number(i);Number.isFinite(l)&&(r[n]=l)}return r},a=o=>Array.isArray(o)?o.filter(r=>typeof r=="string"):[];try{let o=JSON.parse(e),r=o.snapshot;return{sort:o.sort&&typeof o.sort=="object"?o.sort:{},collapsed:a(o.collapsed),h:t(o.h),span:t(o.span),order:a(o.order),addOpen:o.addOpen===!0,stacked:o.stacked===!0,stackSeen:a(o.stackSeen),snapshot:r&&Array.isArray(r.collapsed)&&r.span&&typeof r.span=="object"&&r.h&&typeof r.h=="object"?{collapsed:a(r.collapsed),span:t(r.span),h:t(r.h)}:null}}catch{return null}}function Wn(e){try{Ct(Gn,JSON.stringify(e))}catch{}}var qn="freva.db.term.bg",yl=new Set(["black","ink","graphite","midnight","forest","plum","paper"]);function Vn(){let e=St(qn);return e&&yl.has(e)?e:null}function jn(e){Ct(qn,e)}var Jn="freva.db.term.alpha";function Kn(){let e=Number(St(Jn));return Number.isFinite(e)&&e>=.5&&e<=1?e:null}function Zn(e){Ct(Jn,String(e))}function Fo(e,t,a,o={}){let r=o.parts?[c("span",{class:"chip-label"},[c("span",{class:"chip-k",text:o.parts.key}),c("span",{class:"chip-op",text:` ${o.parts.op} `}),c("span",{class:"chip-v",text:o.parts.value,title:o.title??t})])]:[c("span",{class:"chip-label",text:t,title:o.title??t})];o.tag&&r.push(c("span",{class:"chip-tag",text:o.tag})),r.push(c("span",{class:"x"},[R(E.x,{size:12})]));let n=c("button",{class:`chip${o.geo?" geo":""}${o.negative?" neg":""}`,type:"button","aria-label":`Remove ${o.title??t}`},r);return e.listen(n,"click",a),n}function Sl(e,t){return c("span",{class:"chip scope",role:"note",text:e,title:t,"aria-label":t})}function Yn(e){let t=e.region("chips"),a=e.roots.chips,o=[];for(let[i,l]of jr(e.state))o.push(Sl(`Scope: ${i} \u2260 ${l}`,`This instance always excludes ${i} = ${l}. It cannot be removed.`));for(let i of Object.keys(e.state.selected)){if(Qe(e.state,i))continue;let{baseKey:l,negated:d}=Pe(i);for(let p of e.state.selected[i]){let u=d?`${l} \u2260 ${p}`:`${i}=${p}`;o.push(Fo(t,u,()=>d?e.excludeFacet(l,p):e.toggleFacet(i,p),{negative:d,tag:d?"NOT":void 0,...d?{parts:{key:l,op:"\u2260",value:p}}:{}}))}}let r=e.state.time;r&&o.push(Fo(t,`time ${r.from} \u2192 ${r.to}`,()=>e.setTime(null),{geo:!0,tag:r.mode}));let n=e.state.bbox;if(n){let i=`bbox ${n.minLon},${n.maxLon},${n.minLat},${n.maxLat}`;o.push(Fo(t,i,()=>e.setBbox(null),{geo:!0,tag:n.mode}))}re(a,...o),a.classList.toggle("empty",o.length===0),e.roots.clearAllBtn.classList.toggle("show",Zr(e.state))}var Ia="http://www.w3.org/2000/svg",ea=(e,t)=>(e+180)/360*t,kt=(e,t)=>(90-e)/180*t,No=(e,t)=>Math.max(-180,Math.min(180,e/t*360-180)),Ho=(e,t)=>Math.max(-90,Math.min(90,90-e/t*180)),Cl=[[[-168,66],[-166,60],[-158,57],[-152,59],[-138,58],[-130,53],[-124,47],[-124,40],[-117,32],[-110,23],[-105,20],[-97,16],[-88,15],[-83,9],[-81,13],[-84,22],[-80,25],[-81,29],[-76,35],[-70,42],[-66,45],[-60,47],[-56,51],[-64,53],[-78,52],[-80,60],[-94,58],[-95,68],[-110,68],[-124,70],[-140,70],[-156,71],[-168,66]],[[-46,60],[-43,64],[-40,66],[-30,68],[-22,70],[-20,76],[-33,80],[-45,82],[-58,82],[-62,78],[-53,70],[-50,64],[-46,60]],[[-81,8],[-77,8],[-70,12],[-62,10],[-52,5],[-50,0],[-44,-2],[-40,-6],[-35,-8],[-38,-13],[-42,-23],[-48,-28],[-54,-34],[-58,-39],[-64,-42],[-66,-45],[-68,-50],[-66,-55],[-72,-54],[-74,-45],[-73,-37],[-71,-30],[-70,-20],[-76,-14],[-81,-6],[-80,0],[-78,4],[-81,8]],[[-16,15],[-16,21],[-10,28],[-6,36],[10,37],[20,33],[25,32],[32,31],[34,28],[43,12],[51,12],[48,5],[41,-2],[40,-10],[35,-18],[32,-26],[26,-34],[18,-35],[14,-23],[9,-1],[8,4],[-4,5],[-8,4],[-13,8],[-16,15]],[[-9,44],[-9,39],[-6,36],[3,42],[8,44],[12,45],[18,40],[16,45],[13,45],[13,54],[8,58],[5,61],[10,64],[15,68],[25,71],[30,68],[28,60],[38,60],[40,50],[30,46],[28,41],[22,40],[15,40],[8,44],[-9,44]],[[26,40],[36,36],[36,30],[43,39],[50,44],[48,30],[57,25],[60,25],[66,25],[68,20],[73,18],[77,8],[80,13],[80,22],[88,22],[90,16],[98,10],[104,9],[106,17],[109,22],[112,22],[121,31],[122,40],[128,42],[130,35],[128,45],[135,48],[142,54],[135,58],[150,60],[160,60],[170,66],[180,68],[178,72],[160,71],[140,73],[120,74],[100,77],[80,74],[68,77],[55,73],[50,69],[60,66],[68,66],[62,58],[52,52],[48,46],[40,46],[26,40]],[[114,-22],[113,-28],[116,-34],[123,-34],[131,-32],[138,-35],[141,-38],[147,-38],[150,-37],[153,-28],[153,-25],[146,-19],[142,-11],[136,-12],[130,-13],[124,-16],[122,-18],[114,-22]],[[-180,-72],[-140,-74],[-100,-74],[-60,-70],[-20,-72],[20,-70],[70,-68],[110,-66],[150,-70],[180,-72],[180,-84],[-180,-84],[-180,-72]]];function kl(e,t,a){let o=document.createElementNS(Ia,"polygon");return o.setAttribute("points",e.map(([r,n])=>`${ea(r,t).toFixed(1)},${kt(n,a).toFixed(1)}`).join(" ")),o.setAttribute("fill","var(--land)"),o.setAttribute("stroke","color-mix(in srgb, var(--land) 60%, #000)"),o.setAttribute("stroke-width","0.6"),o.setAttribute("opacity","0.92"),o}function zo(e,t,a,o,r,n){let i=document.createElementNS(Ia,"line");return i.setAttribute("x1",String(e)),i.setAttribute("y1",String(t)),i.setAttribute("x2",String(a)),i.setAttribute("y2",String(o)),i.setAttribute("stroke","var(--border-2)"),i.setAttribute("stroke-width",String(r)),i.setAttribute("opacity",String(n)),i}function La(e,t){let a=document.createElementNS(Ia,"svg");a.setAttribute("width",String(e)),a.setAttribute("height",String(t)),a.setAttribute("viewBox",`0 0 ${e} ${t}`),a.style.display="block",a.style.background="var(--ocean)";for(let o of Cl)a.appendChild(kl(o,e,t));for(let o=-120;o<=120;o+=60)a.appendChild(zo(ea(o,e),0,ea(o,e),t,.5,.4));for(let o=-60;o<=60;o+=30)a.appendChild(zo(0,kt(o,t),e,kt(o,t),.5,.4));a.appendChild(zo(0,kt(0,t),e,kt(0,t),.8,.85));for(let o of["selrect","selrect2"]){let r=document.createElementNS(Ia,"rect");r.setAttribute("class",o),r.setAttribute("x","0"),r.setAttribute("y","0"),r.setAttribute("width","0"),r.setAttribute("height","0"),r.setAttribute("fill","var(--accent)"),r.setAttribute("fill-opacity","0.22"),r.setAttribute("stroke","var(--accent)"),r.setAttribute("stroke-width","1.4"),a.appendChild(r)}return a}function ta(e,t,a,o){let r=e.querySelector(".selrect"),n=e.querySelector(".selrect2");if(!r)return;let i=s=>{s?.setAttribute("width","0"),s?.setAttribute("height","0")};if(!t){i(r),i(n);return}let l=_e(t),d=kt(l.maxLat,o),p=kt(l.minLat,o)-d,u=(s,g,m)=>{if(!s)return;let A=ea(g,a);s.setAttribute("x",String(A)),s.setAttribute("y",String(d)),s.setAttribute("width",String(Math.max(0,ea(m,a)-A))),s.setAttribute("height",String(p))};if(l.wraps){u(r,l.minLon,180),u(n,-180,l.maxLon);return}u(r,l.minLon,l.maxLon),i(n)}var Xn=.1;function Ml(e,t,a=3e3){let o=()=>e.offsetWidth>0&&e.offsetHeight>0;return o()?Promise.resolve(!0):new Promise(r=>{let n=!1,i=p=>{n||(n=!0,d?.disconnect(),window.clearTimeout(l),r(p))},l=window.setTimeout(()=>i(o()),a),d=typeof ResizeObserver=="function"?new ResizeObserver(()=>{o()&&i(!0)}):null;if(d?.observe(e),!d){let p=window.requestAnimationFrame??(u=>window.setTimeout(u,16));p(()=>p(()=>i(o())))}t.add(()=>i(!1))})}async function Ba(e,t,a,o){let r;try{r=await bn(e.cfg.map,t)}catch{return null}if(t.isDisposed||!a.isConnected)return null;let n=c("div",{class:"lmap"});if(a.append(n),!await Ml(n,t)||t.isDisposed||!n.isConnected)return n.remove(),null;let l=r.map(n,{worldCopyJump:!1,zoomControl:!0,attributionControl:!0,dragging:!o.editable,boxZoom:!1});l.attributionControl.setPrefix(!1),r.tileLayer(e.cfg.map.tileUrl,{attribution:e.cfg.map.attribution,maxZoom:12,noWrap:!0}).addTo(l);let d=null,p=m=>{if(d&&(d.remove(),d=null),!m)return;let A=_e(m),v=A.wraps?[[[A.minLat,A.minLon],[A.maxLat,180]],[[A.minLat,-180],[A.maxLat,A.maxLon]]]:[[[A.minLat,A.minLon],[A.maxLat,A.maxLon]]];d=r.layerGroup(v.map(b=>r.rectangle(b,{color:"#4f7cff",weight:1.5,fillOpacity:.18}))).addTo(l)};if(o.bbox){let m=_e(o.bbox);p(o.bbox);let A=Math.max(0,(2-(m.maxLat-m.minLat))/2),v=Math.max(0,(2-(m.maxLon-m.minLon))/2);l.fitBounds([[m.minLat-A,m.minLon-v],[m.maxLat+A,m.maxLon+v]],{padding:[14,14],maxZoom:5})}else l.setView([20,0],1);if(o.editable&&o.onChange){let m=!0,A=c("button",{class:"lmap-mode",type:"button","aria-pressed":"true",title:"Draw a box (click to switch to panning)","aria-label":"Draw a box",text:"\u25AD Draw"}),v=()=>{A.setAttribute("aria-pressed",m?"true":"false"),A.classList.toggle("on",m),A.textContent=m?"\u25AD Draw":"\u270B Pan",A.setAttribute("data-tip",m?"Drawing a box (click to pan instead)":"Panning (click to draw a box)"),n.classList.toggle("drawing",m),m?l.dragging.disable():l.dragging.enable()};t.listen(A,"click",P=>{P.preventDefault(),P.stopPropagation(),m=!m,v()}),r.DomEvent.disableClickPropagation(A),n.append(A),v();let b=null,k=P=>!!P.originalEvent?.target?.closest?.(".leaflet-control, .leaflet-bar, .leaflet-control-attribution"),M=P=>{if(!m||k(P))return;let U=P.originalEvent;U&&U.button===0&&U.preventDefault(),b=P.latlng},B=P=>{b&&p({minLon:Math.min(b.lng,P.latlng.lng),maxLon:Math.max(b.lng,P.latlng.lng),minLat:Math.min(b.lat,P.latlng.lat),maxLat:Math.max(b.lat,P.latlng.lat),mode:"flexible"})},D=P=>{if(!b)return;let U=b;b=null;let q={minLon:Math.min(U.lng,P.latlng.lng),maxLon:Math.max(U.lng,P.latlng.lng),minLat:Math.min(U.lat,P.latlng.lat),maxLat:Math.max(U.lat,P.latlng.lat)};if(Math.abs(q.maxLon-q.minLon)<Xn||Math.abs(q.maxLat-q.minLat)<Xn){p(o.bbox);return}o.onChange?.(q)};l.on("mousedown",M),l.on("mousemove",B),l.on("mouseup",D),t.add(()=>{l.off("mousedown",M),l.off("mousemove",B),l.off("mouseup",D)})}let u=!1,s=()=>{if(!(u||!n.isConnected))try{l.invalidateSize()}catch{}};if(t.setTimeout(s,0),typeof ResizeObserver=="function"){let m=new ResizeObserver(()=>s());m.observe(n),t.add(()=>m.disconnect())}if(typeof IntersectionObserver=="function"){let m=new IntersectionObserver(A=>{A.some(v=>v.isIntersecting)&&s()});m.observe(n),t.add(()=>m.disconnect())}t.listen(window,"resize",s);let g={destroy(){if(!u){u=!0;try{l.remove()}catch{}n.remove()}}};return t.add(()=>g.destroy()),g}var aa=null,Ga=null,Kl=e=>e?import(e):Promise.resolve().then(()=>(bi(),Ai)),Zl=Kl;async function Yl(e){let t=e||null;return aa&&t!==Ga&&console.warn(`[freva-databrowser] data-inspector already loaded from ${Ga??"the packaged dependency"}; ignoring a second source (${t??"the packaged dependency"}). The custom element can only be registered once per page.`),aa||(Ga=t,aa=Zl(t??void 0).then(a=>(a?.DataInspectorElement&&!customElements.get("data-inspector")&&customElements.define("data-inspector",a.DataInspectorElement),a)).catch(a=>{throw aa=null,Ga=null,a})),aa}function Jo(e){return e.cfg.features.inspect}function Xl(e){return e.cfg.features.inspect?"":"Inspect is disabled for this deployment"}function _l(e){return e.cfg.authEnabled&&e.cfg.enableHeavyOps}function $l(e){return e.cfg.authEnabled?"This file isn\u2019t a zarr store - inspecting it needs the data-portal":"This file isn\u2019t a zarr store - inspecting it needs sign-in"}function vi(e){let t=e.dis;async function a(o){if(!Jo(e)){e.toast("warn",Xl(e));return}e.log("info",o?`Inspecting ${o.split("/").pop()??o}\u2026`:"Opening the inspector\u2026");let r;try{r=await Yl(e.cfg.inspectorUrl)}catch{e.toast("error","Inspector unavailable \u2014 the data-inspector module could not be loaded.");return}if(t.isDisposed)return;let n=document.createElement("data-inspector"),i=document.createElement("dialog");i.className="freva-db fdb-inspector-dialog";let l=e.roots.app.getAttribute("data-theme");l&&i.setAttribute("data-theme",l),i.append(n);let d=t.child(),p=!1,u=0,s=document.activeElement;d.add(()=>{try{i.open&&i.close(),i.remove()}catch{}s?.isConnected&&typeof s.focus=="function"&&s.focus()});let g=()=>{p||(p=!0,d.flush())};n.addEventListener("inspector-close",g),i.addEventListener("cancel",b=>{b.preventDefault(),g()}),i.addEventListener("close",g);let m=!1;i.addEventListener("pointerdown",b=>{m=b.target===i}),i.addEventListener("click",b=>{m&&b.target===i&&g(),m=!1});let A={getAuthHeaders:()=>({})},v=async b=>{let k=++u;n.setAttribute("zarr-url",b),n.setAttribute("status","loading"),n.error=null;try{if(typeof r.loadZarrMetadataHtml!="function")throw new Error("inspector build lacks loadZarrMetadataHtml");let M=await r.loadZarrMetadataHtml(b,A);if(p||k!==u)return;n.output=typeof M=="string"?M:M?.html??"",n.setAttribute("status","ready")}catch(M){if(p||k!==u)return;let B=M instanceof Error?M.message:String(M);n.error=_l(e)?`Could not read this as a zarr store (${B}). Server-side inspection isn\u2019t wired in this build.`:$l(e),n.setAttribute("status","error")}};n.addEventListener("inspector-submit",b=>{let M=b.detail?.file??o;M&&v(M)}),o&&(n.file=o),e.roots.overlay.appendChild(i),o?v(o):n.setAttribute("status","ready"),n.setAttribute("open",""),typeof i.showModal=="function"?i.showModal():i.setAttribute("open","")}return{open:o=>a(o),openEmpty:()=>a(null)}}var wi=new WeakMap;function Va(e){let t=wi.get(e);return t||(t={sig:null,sigSeen:null,reqId:0,attempted:new Set,failed:new Set,errorMsg:""},wi.set(e,t)),t}function Si(e,t){return e.map(a=>a.key).join("|")+"#"+t}var xi=["#4F8DF7","#34C98A","#E6B14E","#C79BF0","#F0795F","#3FB6D8","#E0608A"];function ec(e,t){let a=parseInt(e.slice(1),16);return`rgba(${a>>16&255},${a>>8&255},${a&255},${t})`}function Qa(e){return Array.isArray(e)?e.join(", "):String(e)}function Xo(e){let t=e.lastIndexOf("/");return t<0?e:e.slice(t+1)}function tc(e){let t=e.lastIndexOf("/");return t<0?"":e.slice(0,t+1)}function ac(e){let{detailSource:t,focusKey:a,pickedKeys:o}=e.state;return t==="focus"&&a?[a]:o.size>0?[...o]:a?[a]:[]}function Zo(e,t,a=!1){return c("div",{class:"meta-row"},[c("span",{class:"k",text:e}),c("span",{class:`v${a?" na":""}`,text:t})])}function ra(e,t){let a=c("div",{class:"info-sec"});return e&&a.append(R(E[e],{size:13})),a.append(c("span",{text:t})),a}function Ci(e,t,a){let o=a.files,r=a.count>1,n,i=null;if(r){let u=a.count,s=u>10,g=!e.cfg.authEnabled||!e.cfg.enableHeavyOps||s,m=s?`Aggregation handles up to ${10} files - deselect ${u-10} to enable it`:e.cfg.authEnabled?e.cfg.enableHeavyOps?"Aggregation isn\u2019t wired up in this build yet":"Aggregate - data-portal not enabled":"Aggregate - needs sign-in",A=c("button",{class:`btn primary${s?" locked":""}`,type:"button",disabled:g?"true":null,title:m},[R(E.aggregate,{size:15}),c("span",{text:"Aggregate"})]);g||t.listen(A,"click",()=>e.toast("warn","Aggregation isn\u2019t wired up in this build yet.")),n=A,g&&(i=c("p",{class:"scope-note",text:m}))}else{let u=Jo(e),s=c("button",{class:"btn primary",type:"button",disabled:u?null:"true",title:u?"Inspect data (zarr stores render without sign-in)":"Inspect is disabled for this deployment"},[R(E.inspect,{size:15}),c("span",{text:"Inspect data"})]);u&&t.listen(s,"click",()=>{e.openInspect(o[0])}),n=s}let l=u=>{let s=o.map(g=>`file=${encodeURIComponent(g)}`).join("&");e.exportCatalogue(u,s,"file")},d=c("button",{class:"btn",type:"button"},[qe("intake",{size:16}),c("span",{text:"Intake catalogue (.json)"})]),p=c("button",{class:"btn",type:"button"},[qe("stac",{size:16}),c("span",{text:"STAC catalogue (.zip)"})]);return t.listen(d,"click",()=>l("intake")),t.listen(p,"click",()=>l("stac")),c("div",{class:"info-actions"},[n,i,ra("download","Download as catalog"),c("p",{class:"scope-note",text:r?`scoped to your ${a.count} picks - file= constraint`:"scoped to this file - file= constraint"}),d,p])}function ki(e,t,a){if(!a)return null;let o=c("button",{class:"btn sm",type:"button",text:"Retry failed"});return t.listen(o,"click",()=>{let r=Va(e);r.attempted.clear(),r.failed.clear(),r.sig=null,Re(e)}),c("div",{class:"partial-flag"},[c("span",{text:`${a} file(s) could not be loaded. `}),o])}function oc(e,t,a,o,r=0){let n=e.roots.infoScroll,i=[c("div",{class:"info-name",text:Xo(a.file)}),c("div",{class:"info-sub",text:tc(a.file)})],l=ki(e,t,r);if(l&&i.push(l),i.push(ra("box","Bounding box")),a.bbox){let A=La(290,150),v=_e(a.bbox),b=v.global?"lon global (\u2212180 \u2192 180)":`lon ${v.minLon} \u2192 ${v.maxLon}${v.wraps?" (crosses the antimeridian)":""}`,k=c("div",{class:"minimap"},[A]),M=c("button",{class:"map-zoom",type:"button",title:"Zoomable map","aria-label":"Switch to the zoomable map"},[R(E.search,{size:13}),c("span",{text:"Zoom"})]);t.listen(M,"click",()=>{M.disabled=!0,M.textContent="Loading map\u2026",Ba(e,t,k,{editable:!1,bbox:a.bbox??null}).then(D=>{if(!D){M.disabled=!1,M.textContent="Zoom unavailable";return}A.remove(),k.classList.add("has-leaflet"),M.remove()})});let B=c("div",{class:"miniwrap"},[c("div",{class:"map-slot"},[k,M]),c("div",{class:"coords"},[c("span",{text:b}),c("span",{text:`lat ${v.minLat} \u2192 ${v.maxLat}`})])]);i.push(B),ta(A,a.bbox,290,150)}else i.push(c("div",{class:"meta"},[c("div",{class:"na",text:"Spatial extent: not available yet."})]));i.push(ra("clock","Time"));let d=a.timeRange,p=d?null:ya(a.file),u=d??p,s=p!=null||a.timeRangeInferred===!0;i.push(c("div",{class:"meta"},[u?Zo(s?"time range (from filename)":"time range",u):c("div",{class:"na",text:"Time range: not available yet."})])),i.push(ra(null,"Facets (from ?file=)"));let g=c("div",{class:"meta"}),m=Object.entries(a.meta??{});if(m.length)for(let[A,v]of m)g.append(Zo(Yt(e.state,A),Qa(v)));else g.append(c("div",{class:"na",text:"No per-file metadata returned."}));i.push(g),i.push(Ci(e,t,o)),re(n,...i)}var Yo=25;function rc(e,t,a,o){let r=t.child(),n=document.activeElement,i=()=>{r.flush()},l=null,d=c("button",{class:"x",type:"button","aria-label":"Close comparison"},[R(E.close,{size:18})]),p=c("div",{class:"dmm-head"},[c("span",{class:"dmm-title",text:o}),d]),u=c("div",{class:"dmm-body"},[a.cloneNode(!0)]),s=c("div",{class:"dmm-modal",tabindex:"-1"},[p,u]),g=c("dialog",{class:"dmm-backdrop","aria-label":o},[s]);r.listen(d,"click",i),r.listen(g,"click",v=>{v.target===g&&i()});let m=()=>Array.from(s.querySelectorAll('button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])')).filter(v=>!v.hasAttribute("disabled"));r.listen(document,"keydown",v=>{let b=v;if(b.key==="Escape"){i();return}if(b.key!=="Tab")return;let k=m();if(k.length===0){b.preventDefault(),s.focus();return}let M=k[0],B=k[k.length-1],D=document.activeElement;s.contains(D)?b.shiftKey&&D===M?(b.preventDefault(),B.focus()):!b.shiftKey&&D===B&&(b.preventDefault(),M.focus()):(b.preventDefault(),M.focus())}),r.add(()=>{try{l?.open&&l.close()}catch{}try{g.remove()}catch{}queueMicrotask(()=>{let v=n&&n.isConnected?n:e.roots.infoBtn;if(v&&v.isConnected)try{v.focus()}catch{}})}),e.roots.app.appendChild(g);let A=g;if(typeof A.showModal=="function"){l=A;try{A.showModal()}catch{l=null,A.setAttribute("open","")}r.listen(A,"cancel",v=>{v.preventDefault(),i()})}else A.setAttribute("open","");(m()[0]??s).focus()}var Wa="__time",qa="__bbox";function nc(e){let t=e.timeRange,a=t?null:ya(e.file),o=t??a;return o?a||e.timeRangeInferred===!0?`${o} (from filename)`:o:null}function ic(e){if(!e.bbox)return null;let t=_e(e.bbox);return`${t.global?"lon \u2212180 \u2192 180":`lon ${t.minLon} \u2192 ${t.maxLon}${t.wraps?" (crosses the antimeridian)":""}`}, lat ${e.bbox.minLat} \u2192 ${e.bbox.maxLat}`}function Ko(e,t){return t===Wa?"Time range":t===qa?"Spatial extent":Yt(e.state,t)}function sc(e,t,a,o,r,n=0){let i=e.roots.infoScroll,l=a.map(B=>{let D=nc(B),P=ic(B),U={...B.meta??{}};return D&&(U[Wa]=D),P&&(U[qa]=P),U}),d=[...new Set(l.flatMap(B=>Object.keys(B)))].filter(B=>B!==Wa&&B!==qa),u=[...[Wa,qa].filter(B=>l.some(D=>B in D)),...d],s=B=>[...new Set(l.map(D=>B in D?Qa(D[B]):"-"))],g=u.filter(B=>s(B).length>1),m=u.filter(B=>!g.includes(B)),A=[c("div",{class:"info-name",text:`${o.count} files selected`}),c("div",{class:"info-sub",text:`comparing - ${g.length} field${g.length===1?"":"s"} differ`})];if(r){let B=ki(e,t,r);B&&A.push(B)}if(A.push(ra(null,"Differences")),g.length){let B=c("div",{class:"diff-summary"});for(let K of g)B.append(c("span",{class:"varchip"},[c("span",{class:"vc-k",text:Ko(e,K)}),c("span",{class:"vc-n",text:String(s(K).length)})]));A.push(B);let D={};for(let K of g)D[K]=s(K);let P=c("tr",{},[c("th",{text:"#"}),...g.map(K=>c("th",{text:Ko(e,K)}))]),U=a.map((K,ne)=>{let W=l[ne],le=g.map($=>{let pe=$ in W?Qa(W[$]):"-",w=xi[D[$].indexOf(pe)%xi.length];return c("td",{},[c("span",{class:"dchip",style:`background:${ec(w,.16)};color:${w}`,text:pe})])});return c("tr",{},[c("td",{class:"rownum",title:Xo(K.file),text:String(ne+1)}),...le])}),q=c("table",{class:"dmatrix"},[c("thead",{},[P]),c("tbody",{},U)]),Z=c("button",{class:"btn diff-enlarge",type:"button",title:"Open the comparison full-screen"},[R(E.expandWide,{size:14}),c("span",{text:"Enlarge"})]);t.listen(Z,"click",()=>rc(e,t,q,`Comparing ${a.length} file${a.length===1?"":"s"}`)),A.push(c("div",{class:"diff-tools"},[Z])),A.push(c("div",{class:"dscroll",role:"region","aria-label":"Per-file differences"},[q])),n>0&&A.push(c("p",{class:"scope-note",style:"margin:6px 16px 0",text:`Comparing the first ${Yo} files - ${n} more selected. Deselect some to compare the rest.`}))}else A.push(c("p",{class:"scope-note",style:"margin:0 16px",text:u.length===0?`No per-file metadata was returned to compare across these ${a.length} files.`:`Every field is identical across the ${a.length} files.`}));let v=c("div",{class:"shared-body"},[c("div",{class:"meta"},m.map(B=>Zo(Ko(e,B),Qa(l[0][B]))))]),b=c("div",{class:"info-sec shared-head",role:"button",tabindex:"0"},[R(E.check,{size:13}),c("span",{text:`Shared by all ${a.length}`}),c("span",{class:"chev2"},[R(E.caret,{size:12})])]),k=c("div",{class:"shared open"},[b,v]),M=()=>{k.classList.toggle("open")};t.listen(b,"click",M),t.listen(b,"keydown",B=>{let D=B.key;(D==="Enter"||D===" ")&&(B.preventDefault(),M())}),A.push(k),A.push(Ci(e,t,o)),re(i,...A)}function yi(e,t){re(e.roots.infoScroll,c("div",{class:"empty"},[c("div",{class:"big"},[R(E.info,{size:30})]),c("p",{},[document.createTextNode(t??"Details is on. Click a file to query "),t?null:c("code",{text:"?file=<name>"}),document.createTextNode(t?"":" - nothing is fetched until you do.")])]))}function lc(e,t){re(e.roots.infoScroll,c("div",{class:"querying"},[c("span",{text:`querying ?file= for ${t}\u2026`}),c("div",{class:"bar"})]))}function cc(e,t,a){let o=c("button",{class:"btn",type:"button"},[R(E.retry,{size:15}),c("span",{text:"Retry"})]);t.listen(o,"click",()=>{let r=Va(e);r.attempted.clear(),r.failed.clear(),Re(e)}),re(e.roots.infoScroll,c("div",{class:"state-msg err"},[c("p",{text:a}),o]))}async function dc(e,t,a){let o=new Array(e.length),r=0,n=async()=>{for(;r<e.length;){let i=r++;o[i]=await a(e[i])}};return await Promise.all(Array.from({length:Math.min(t,e.length)},()=>n())),o}var pc=6,uc=500;function fc(e,t,a){let o=Va(e),r=e.state.flavour,n=Si(t,r);if(o.sig===n)return;o.sig=n;let i=e.api.nextRequestId();o.reqId=i,e.state.details="loading";let l=e.api.channelSignal("details");dc(a,pc,d=>e.api.filePathMetadata(r,[d.file],l).then(p=>({ok:!0,r:d,res:p})).catch(p=>({ok:!1,r:d,err:p}))).then(d=>{if(o.reqId!==i||e.state.flavour!==r)return;let p=!1;for(let u of d)if(o.attempted.add(u.r.key),u.ok){let s=(u.res.search_results??[]).find(b=>(b.file??b.uri)===u.r.file)??(u.res.search_results??[])[0],g=pn(s?.bbox),m=un(s?.time),A=m?null:ya(u.r.file),v=e.state.detailsCache;for(v.set(oa(r,u.r.key),{...u.r,meta:en(u.res.facets??{}),bbox:g?{...g,mode:"flexible"}:u.r.bbox,timeRange:m??A??void 0,timeRangeInferred:!m&&!!A});v.size>uc;){let b=v.keys().next().value;if(b===void 0)break;v.delete(b)}o.failed.delete(u.r.key)}else u.err instanceof De&&u.err.aborted?p=!0:(o.failed.add(u.r.key),o.errorMsg=u.err instanceof De?u.err.message:"Details request failed.");p||(o.sig=null,Re(e))}).catch(()=>{if(!(o.reqId!==i||e.state.flavour!==r)){for(let d of a)o.attempted.add(d.key),o.failed.add(d.key);o.errorMsg="Details response could not be read.",o.sig=null,Re(e)}})}var oa=(e,t)=>`${e}::${t}`;function Re(e){let t=e.region("details"),a=e.roots.info;if(!e.state.detailsOpen){a.classList.add("collapsed");return}a.classList.remove("collapsed");let o=Va(e),r=ac(e);if(r.length===0){e.state.details="idle",yi(e);return}let n=new Map(e.state.rows.map(v=>[v.key,v])),i=r.map(v=>e.state.detailsCache.get(oa(e.state.flavour,v))??n.get(v)).filter(v=>!!v),l=Math.max(0,i.length-Yo),d=l>0?i.slice(0,Yo):i;if(d.length===0){e.state.details="empty",yi(e,"The selected file is no longer in the results.");return}let p=Si(d,e.state.flavour);o.sigSeen!==p&&(o.attempted.clear(),o.failed.clear(),o.sigSeen=p);let u=d.filter(v=>!e.state.detailsCache.has(oa(e.state.flavour,v.key))&&!o.attempted.has(v.key));if(u.length){fc(e,d,u),lc(e,d.length>1?`${d.length} files`:Xo(d[0].file));return}let s=d.filter(v=>e.state.detailsCache.has(oa(e.state.flavour,v.key))),g=d.length-s.length;if(s.length===0){e.state.details="error",cc(e,t,o.errorMsg||"No metadata returned.");return}e.state.details="loaded";let m=s.map(v=>e.state.detailsCache.get(oa(e.state.flavour,v.key))),A={files:d.map(v=>v.file),count:d.length};m.length>1?sc(e,t,m,A,g,l):oc(e,t,m[0],A,g)}var hc="\u2212",Mi="\u2260";function na(e,t){let a=Kt(e,t).length,o=Ze(e,t).length;return{included:a,excluded:o,total:a+o}}function ia(e){let t=[];return e.included&&t.push({negative:!1,count:e.included}),e.excluded&&t.push({negative:!0,count:e.excluded}),t}function sa(e,t,a,o){let r=e.negative?"excluded":"included",n=`${e.negative?hc:"+"}${e.count}`,i=`Clear ${e.count} ${r} ${t} value${e.count===1?"":"s"}`,l=c("button",{class:`fh-count fb ${e.negative?"fb-exc":"fb-inc"}`,type:"button","data-mode":e.negative?"exclude":"include",title:i,"aria-label":i},[c("span",{class:"fb-n",text:n}),c("span",{class:"fb-x","aria-hidden":"true",text:"\xD7"})]);return l.style.setProperty("--fb-ch",String(n.length)),o(l,"click",d=>{d.stopPropagation(),a()}),l}function la(e,t){let a=e.state.totalCount;return!a||a<=0?null:Math.min(100,t/a*100)}function ja(e,t,a,o){let r=la(e,a),n=r===null?"":` - ${a.toLocaleString("en-US")} (${r<.1?"<0.1":r.toFixed(1)}% of results)`;return o?`${t} - ${o}${n}`:`${t}${n}`}function _o(e){return ot(e)}function mc(e,t){return!ot(e)||!ot(t)?!1:Ca(e)||Ca(t)?!0:ft(e)<=ft(t)}var gc={flexible:"Any overlap between your range and the file\u2019s period (intersects).",strict:"Containment match - sent to the backend as time_select=strict.",file:"File-relative containment - sent as time_select=file."};function $o(e,t,a,o=!1){let r=e.state.time,n=r?.mode??"flexible",i=c("input",{class:"date-text",type:"text",inputmode:"numeric",placeholder:"YYYY or YYYY-MM-DD",value:r?.from??"","aria-label":"From"}),l=c("input",{class:"date-text",type:"text",inputmode:"numeric",placeholder:"YYYY or YYYY-MM-DD",value:r?.to??"","aria-label":"To"}),d=c("div",{class:"err-line"}),p=(b,k)=>{let M=c("input",{type:"date",class:"date-native",tabindex:"-1","aria-hidden":"true"}),B=c("button",{class:"date-pick",type:"button","aria-label":`Choose a ${k.toLowerCase()} date from a calendar`,title:`Pick a ${k.toLowerCase()} date`},[R(E.clock,{size:14})]),D=()=>{let P=b.value.trim();M.value=/^\d{4}-\d{2}-\d{2}$/.test(P)&&_o(P)?P:""};return t.listen(B,"click",()=>{D();let P=M;if(typeof P.showPicker=="function")try{P.showPicker();return}catch{}M.focus(),M.click()}),t.listen(M,"change",()=>{M.value&&(b.value=M.value,g(),m())}),t.listen(b,"change",D),D(),c("span",{class:"date-pickwrap"},[B,M])},u=["flexible","strict","file"].map(b=>{let k=c("button",{type:"button",class:b===n?"on":"",title:gc[b],text:b});return t.listen(k,"click",()=>{n=b;for(let M of u)M.classList.toggle("on",M===k);g(),m()}),k}),s=()=>mc(i.value.trim(),l.value.trim());function g(){let b=_o(i.value),k=_o(l.value);i.classList.toggle("bad",!b),l.classList.toggle("bad",!k);let M=b&&k&&s();d.textContent=!b||!k?"Use YYYY, YYYY-MM, YYYY-MM-DD or a datetime.":M?"":"The start is after the end."}t.listen(i,"input",g),t.listen(l,"input",g);let m=()=>{let b=i.value.trim(),k=l.value.trim();if(!b&&!k){e.setTime(null);return}s()&&e.setTime({from:b,to:k,mode:n})};t.listen(i,"change",m),t.listen(l,"change",m);let A=c("button",{class:"btn",type:"button",text:"Clear"});t.listen(A,"click",()=>{a(),e.setTime(null)});let v=c("div",{class:`editor${o?" inline":""}`},[c("h5",{},[R(E.clock,{size:16}),c("span",{text:"Time range"}),c("span",{class:"sub",text:"time_select"})]),c("div",{class:"daterow"},[c("label",{text:"From"}),i,p(i,"From")]),c("div",{class:"daterow"},[c("label",{text:"To"}),l,p(l,"To")]),c("div",{class:"modes"},u),d,...o?[]:[c("div",{class:"actions"},[A])]]);return g(),v}function Ti(e,t){let a=e.region("popover"),o=$o(e,a,()=>e.popover.close());e.popover.open(t,o,{placement:"right",className:"editor-pop",autoFocus:!0,reanchor:()=>e.roots.facetList.querySelector('.special[aria-label="Edit time range"]')})}var Ja=e=>Math.round(e*100)/100,Mt=276,Tt=150,Ac={flexible:"Any overlap between your box and the file (intersects).",strict:"Containment match - sent to the backend as bbox_select=strict.",file:"File-relative containment - sent as bbox_select=file."};function Ei(e){return Number.isFinite(e.minLon)&&Number.isFinite(e.maxLon)&&Number.isFinite(e.minLat)&&Number.isFinite(e.maxLat)&&e.minLon>=-180&&e.maxLon<=180&&e.minLat>=-90&&e.maxLat<=90&&e.minLon<e.maxLon&&e.minLat<e.maxLat}function er(e,t,a,o={}){let r=o.autoMap??!1,n=o.inline??!1,i=e.state.bbox,l=i?{minLon:i.minLon,maxLon:i.maxLon,minLat:i.minLat,maxLat:i.maxLat}:null,d=i?.mode??"flexible",p=La(Mt,Tt),u=c("div",{class:"map-overlay"}),s=c("div",{class:"minimap"},[p,u]),g=(w,y)=>{let S=c("input",{type:"text",inputmode:"decimal","aria-label":w,value:l?String(l[y]):""});return{wrap:c("div",{class:"f"},[c("label",{text:w}),S]),input:S}},m=g("minLon","minLon"),A=g("maxLon","maxLon"),v=g("minLat","minLat"),b=g("maxLat","maxLat");function k(){l&&(m.input.value=l.minLon.toFixed(1),A.input.value=l.maxLon.toFixed(1),v.input.value=l.minLat.toFixed(1),b.input.value=l.maxLat.toFixed(1))}function M(){ta(p,l?{...l,mode:d}:null,Mt,Tt);let w=l!==null&&Ei(l);for(let y of[m,A,v,b])y.input.classList.toggle("bad",l!==null&&!w)}let B=()=>{l={minLon:parseFloat(m.input.value),maxLon:parseFloat(A.input.value),minLat:parseFloat(v.input.value),maxLat:parseFloat(b.input.value)},M()};for(let w of[m,A,v,b])t.listen(w.input,"input",B);let D=null,P=!1,U=w=>{let y=u.getBoundingClientRect(),S=y.width?Mt/y.width:1,O=y.height?Tt/y.height:1;return{x:Math.max(0,Math.min(Mt,(w.clientX-y.left)*S)),y:Math.max(0,Math.min(Tt,(w.clientY-y.top)*O))}};t.listen(u,"mousedown",w=>{D=U(w),P=!1});let q=t.listen(window,"mousemove",w=>{if(!D)return;P=!0;let y=U(w);l={minLon:No(Math.min(D.x,y.x),Mt),maxLon:No(Math.max(D.x,y.x),Mt),maxLat:Ho(Math.min(D.y,y.y),Tt),minLat:Ho(Math.max(D.y,y.y),Tt)},k(),M()}),Z=t.listen(window,"mouseup",()=>{D&&P&&W(),D=null,P=!1}),K=["flexible","strict","file"].map(w=>{let y=c("button",{type:"button",class:w===d?"on":"",title:Ac[w],text:w});return t.listen(y,"click",()=>{d=w;for(let S of K)S.classList.toggle("on",S===y);M(),W()}),y}),ne=c("button",{class:"btn",type:"button",text:"Clear"});t.listen(ne,"click",()=>{a(),e.setBbox(null)});let W=()=>{l&&Ei(l)&&e.setBbox({...l,mode:d})},le=()=>Ba(e,t,s,{editable:!0,bbox:l?{...l,mode:d}:null,onChange:w=>{l={minLon:Ja(w.minLon),maxLon:Ja(w.maxLon),minLat:Ja(w.minLat),maxLat:Ja(w.maxLat)},k(),M(),W()}}).then(w=>{if(!w){$.disabled=!1,$.textContent="Zoom unavailable";return}p.remove(),u.remove(),s.classList.add("has-leaflet"),$.remove()}),$=c("button",{class:"map-zoom",type:"button",title:"Zoomable map","aria-label":"Switch to the zoomable map"},[R(E.search,{size:13}),c("span",{text:"Zoom"})]);t.listen($,"click",()=>{$.disabled=!0,$.textContent="Loading map\u2026",le()}),r&&($.remove(),le());for(let w of[m,A,v,b])t.listen(w.input,"change",()=>W());let pe=c("div",{class:`editor${n?" inline":""}`},[c("h5",{},[R(E.box,{size:16}),c("span",{text:"Bounding box"}),c("span",{class:"sub",text:"bbox_select - drag to draw"})]),c("div",{class:"map-slot"},[s,$]),c("div",{class:"draw-hint",text:"Drag a rectangle, or type bounds below."}),c("div",{class:"bbox-fields"},[m.wrap,A.wrap,v.wrap,b.wrap]),c("div",{class:"modes"},K),...n?[]:[c("div",{class:"actions"},[ne])]]);return l&&(k(),ta(p,{...l,mode:d},Mt,Tt)),M(),{editor:pe,dispose:()=>{q(),Z()}}}function Di(e,t){let a=e.region("popover"),{editor:o,dispose:r}=er(e,a,()=>e.popover.close(),{autoMap:!0});e.popover.open(t,o,{placement:"right",className:"editor-pop",onClose:r,reanchor:()=>e.roots.facetList.querySelector('.special[aria-label="Edit bounding box"]')})}var bc=60;function He(e){Wn({sort:e.state.overviewSort,collapsed:[...e.state.overviewCollapsed],span:e.state.overviewSpan,h:e.state.overviewH,order:e.state.overviewOrder,addOpen:e.state.overviewAddOpen,stacked:e.state.overviewStacked,stackSeen:e.state.overviewStackSeen,snapshot:e.state.overviewSnapshot})}function Bi(e,t,a,o){a.classList.add("clickable"),t.listen(a,"click",r=>{if(r.target.closest('button, .drag-grip, [role="button"]'))return;let n=a.ownerDocument.getSelection?.();n&&!n.isCollapsed&&a.contains(n.anchorNode)||(e.state.overviewCollapsed.has(o)?e.state.overviewCollapsed.delete(o):e.state.overviewCollapsed.add(o),He(e),e.renderOverview())})}var Pi=24,Ya=e=>Math.min(Pi,Math.max(1,e)),vc=e=>Math.max(1,getComputedStyle(e).gridTemplateColumns.split(" ").filter(Boolean).length),wc=2,Gt=e=>Math.min(wc,Math.max(1,e)),Oi=e=>Ya(parseInt(e.style.gridColumn.replace("span ",""),10)||1);function Ri(e){return c("button",{class:"drag-grip",type:"button",title:"Drag, or focus and use \u2190 \u2192 to reorder","aria-label":`Reorder ${e} - use the arrow keys`,text:"\u283F"})}function ca(e){return c("button",{class:"fcard-resize",type:"button",title:"Drag, or focus and use arrow keys to resize","aria-label":`Resize ${e} - \u2190 \u2192 change width, \u2191 \u2193 change height`})}var Ka=new WeakMap;function xc(e){let t=Ka.get(e);if(!t)return;Ka.set(e,null);let a=t.handle==="grip"?".drag-grip":".fcard-resize";for(let o of Array.from(e.roots.overviewGrid.querySelectorAll(".fcard[data-key]")))if(o.dataset.key===t.key){o.querySelector(a)?.focus();break}}function Fi(e,t){let a=new Set(t),o=new Map,r=null;for(let i of e){if(a.has(i)){r=i;continue}let l=o.get(r)??[];l.push(i),o.set(r,l)}let n=[...o.get(null)??[]];for(let i of t){n.push(i);for(let l of o.get(i)??[])n.push(l)}return n}function zi(e){return Array.from(e.querySelectorAll(".fcard[data-key]")).map(t=>t.dataset.key??"").filter(Boolean)}function yc(e,t,a,o){let r=zi(t),n=r.indexOf(a),i=n+o;n<0||i<0||i>=r.length||([r[n],r[i]]=[r[i],r[n]],e.state.overviewOrder=Fi(e.state.overviewOrder,r),Ka.set(e,{key:a,handle:"grip"}),He(e),e.renderOverview())}function Sc(e,t,a,o){let r=e.state.overviewSpan[t]??1,n=e.state.overviewH[t]??1,i=Ya(r+a),l=Gt(n+o);i===r&&l===n||(e.state.overviewSpan[t]=i,e.state.overviewH[t]=l,Ka.set(e,{key:t,handle:"resize"}),He(e),e.renderOverview())}function Cc(e,t){if(!e.state.overviewStacked)return;let a=new Set(e.state.overviewStackSeen),o=!1;for(let r of[...t,"__time","__bbox"])a.has(r)||(a.add(r),e.state.overviewCollapsed.add(r),o=!0);o&&(e.state.overviewStackSeen=[...a],He(e))}var kc=["button","input","textarea","select","a","label","[contenteditable]",'[role="button"]','[role="checkbox"]','[role="textbox"]','[role="listbox"]','[role="option"]','[role="slider"]','[role="menuitem"]',".fcard-resize",".leaflet-container",".fcard-special-body svg",".bbox-map",".map-svg",".te-map"].join(", ");function Mc(e,t){let a=null,o=null,r=null,n=!1,i=null,l=5,d=u=>{a="reorder",o=u,u.classList.add("dragging"),document.body.classList.add("fdb-dragging")};e.dis.listen(t,"pointerdown",u=>{let s=u;if(s.button!==void 0&&s.button!==0)return;let g=s.target,m=g.closest(".fcard");if(m){if(g.closest(".fcard-resize")){let A=Oi(m),v=m.getBoundingClientRect();a="resize",o=m;let b=vc(t),k=Gt(Number(m.dataset.rows)||1);i={startX:s.clientX,startY:s.clientY,startSpan:A,startRows:k,pitch:Math.max(120,v.width/A),rowPitch:Math.max(120,v.height/k),maxSpan:b},m.classList.add("resizing"),document.body.classList.add("fdb-dragging"),s.preventDefault();return}if(g.closest(".drag-grip")){d(m),s.preventDefault();return}g.closest(kc)||(r={x:s.clientX,y:s.clientY,card:m,pointerId:s.pointerId})}}),e.dis.listen(window,"pointermove",u=>{let s=u;if(r&&!o){if(s.pointerId!==r.pointerId)return;let m=s.clientX-r.x,A=s.clientY-r.y;if(Math.hypot(m,A)<l)return;d(r.card),n=!0,t.ownerDocument.getSelection?.()?.removeAllRanges(),s.preventDefault()}if(!o)return;let g=s;if(a==="resize"&&i){let m=Math.round((g.clientX-i.startX)/i.pitch),A=Math.min(i.maxSpan,Ya(i.startSpan+m));o.style.gridColumn=`span ${A}`,o.classList.toggle("wide",A>1);let v=Math.round((g.clientY-i.startY)/i.rowPitch);o.dataset.rows=String(Gt(i.startRows+v))}else if(a==="reorder"){let A=o.ownerDocument.elementFromPoint(g.clientX,g.clientY)?.closest(".fcard");if(A&&A!==o&&A.parentElement===t){let v=A.getBoundingClientRect(),b=g.clientX>v.left+v.width/2;t.insertBefore(o,b?A.nextSibling:A)}}});let p=()=>{if(r=null,!!o){if(a==="resize"){let u=o.dataset.key;u&&(e.state.overviewSpan[u]=Oi(o),e.state.overviewH[u]=Gt(Number(o.dataset.rows)||1)),o.classList.remove("resizing")}else a==="reorder"&&(o.classList.remove("dragging"),e.state.overviewOrder=Fi(e.state.overviewOrder,zi(t)));document.body.classList.remove("fdb-dragging"),a=null,o=null,i=null,He(e),e.renderOverview()}};e.dis.listen(window,"pointerup",p),e.dis.listen(window,"pointercancel",p),e.dis.listen(t,"click",u=>{n&&(n=!1,u.stopPropagation(),u.preventDefault())},!0),e.dis.listen(t,"keydown",u=>{let s=u;if(s.altKey||s.ctrlKey||s.metaKey)return;let g=s.target,m=g.closest(".fcard[data-key]");if(!m)return;let A=m.dataset.key??"";if(g.closest(".drag-grip")){let v=s.key==="ArrowLeft"||s.key==="ArrowUp"?-1:s.key==="ArrowRight"||s.key==="ArrowDown"?1:0;if(!v)return;s.preventDefault(),yc(e,t,A,v)}else if(g.closest(".fcard-resize")){let v=s.key==="ArrowRight"?1:s.key==="ArrowLeft"?-1:0,b=s.key==="ArrowUp"?1:s.key==="ArrowDown"?-1:0;if(!v&&!b)return;s.preventDefault(),Sc(e,A,v,b)}}),e.dis.add(()=>{r=null,n=!1,o&&(o.classList.remove("resizing","dragging"),document.body.classList.remove("fdb-dragging"),a=null,o=null,i=null)})}var Tc=e=>e.toLocaleString("en-US");function Ec(e,t,a,o,r){let n=gt(e.state,a.key,o),i=wa(e.state,a.key,o),l=Pt(e.state,a.key,o),d=At(e.state,a.key,o),p=c("button",{class:`fval${n?" sel":""}${i?" excl":""}${l?" locked":""}`,type:"button",role:"checkbox","aria-checked":n?"true":"false","aria-disabled":l?"true":"false","aria-label":l?`${a.label}: ${o} (locked scope)`:`Include ${a.label} ${o}`,"data-val":o.toLowerCase(),title:l?`${o} - this instance is scoped to this value`:ja(e,o,r,d)},[c("span",{class:"cb"},n?[R(E.check,{size:11})]:[]),c("span",{class:"nm",text:o}),c("span",{class:"n",text:Tc(r)})]),u=la(e,r);if(u!==null&&(p.classList.add("has-bar"),p.style.setProperty("--pct",`${u}%`)),l||t.listen(p,"click",()=>e.toggleFacet(a.key,o)),l)return c("div",{class:"fval-row"},[p]);let s=c("button",{class:`fval-ex${i?" on":""}`,type:"button","aria-pressed":i?"true":"false","aria-label":`Exclude ${a.label} ${o}`,title:i?`Stop excluding ${o}`:`Exclude ${o} from the results`,text:"\u2260"});return t.listen(s,"click",g=>{g.stopPropagation(),e.excludeFacet(a.key,o)}),c("div",{class:`fval-row${i?" excl":""}`},[p,s])}function tr(e,t){let a=e.state.overviewSort[t.key]??"count",o=Vr(e.state,t).slice();return a==="alpha"?o.sort((r,n)=>r.value.localeCompare(n.value)):o.sort((r,n)=>n.count-r.count),o}function Za(e,t,a,o,r){let n=c("button",{class:`exp${a?" on":""}`,type:"button","aria-pressed":a?"true":"false","aria-label":t,title:t},[R(e,{size:14})]);return r.listen(n,"click",i=>{i.stopPropagation(),o(i)}),n}function ar(e,t,a,o,r=()=>e.renderOverview()){let n=o==="alpha",i=c("button",{class:"exp sortbtn",type:"button","aria-label":n?"Sorted A\u2013Z - switch to sorting by count":"Sorted by count - switch to A\u2013Z",title:n?"Sorted A\u2013Z (click: by count)":"Sorted by count (click: A\u2013Z)"},[R(n?E.sortAlpha:E.sortCount,{size:14}),c("span",{class:"sortlbl",text:n?"A\u2013Z":"Count"})]);return t.listen(i,"click",l=>{l.stopPropagation(),e.state.overviewSort[a]=n?"count":"alpha",He(e),r()}),i}function Ii(e,t,a){let o=e.state,r=rt(o,a.key),n=o.overviewCollapsed.has(a.key),i=Math.min(Pi,Math.max(1,o.overviewSpan[a.key]??1)),l=o.overviewSort[a.key]??"count",d=Gt(o.overviewH[a.key]??1),p=c("div",{class:`fcard${n?" collapsed":""}${i>1?" wide":""}`,"data-key":a.key});p.style.gridColumn=`span ${i}`,p.dataset.rows=String(n?1:d);let u=c("div",{class:`fcard-h${r?" active":""}`},[Ri(a.label),c("span",{class:"fh-label",text:a.label})]);if(r)for(let b of ia(na(o,a.key)))u.append(sa(b,a.label,()=>e.clearFacetMode(a.key,b.negative),(k,M,B)=>t.listen(k,M,B)));u.append(c("span",{class:"badge",text:a.hasMore?`${a.values.length}+`:String(a.values.length)})),u.append(ar(e,t,a.key,l)),(i!==1||d!==1)&&u.append(Za(E.reset,"Reset size",!1,()=>{delete o.overviewSpan[a.key],delete o.overviewH[a.key],He(e),e.renderOverview()},t));let s=Za(n?E.chevron:E.minimize,n?"Expand":"Minimize",n,()=>{n?o.overviewCollapsed.delete(a.key):o.overviewCollapsed.add(a.key),He(e),e.renderOverview()},t);if(s.setAttribute("aria-expanded",n?"false":"true"),u.append(s),Bi(e,t,u,a.key),p.append(u),n)return p.append(ca(a.label)),p;let g=c("input",{class:"within",type:"text",placeholder:`filter ${a.label.toLowerCase()}\u2026`,value:o.overviewFilters[a.key]??"","aria-label":`Filter ${a.label}`}),m=c("div",{class:"fcard-vals"});if(a.values.length===0)return p.append(c("div",{class:"fcard-empty",text:"No values in this selection."})),p.append(ca(a.label)),p;let A=null,v=()=>{A?.flush(),A=t.child();let b=(o.overviewFilters[a.key]??"").toLowerCase(),k=tr(e,a),M=b?k.filter(D=>D.value.toLowerCase().includes(b)):k;re(m);let B=A;Ea(B,m,M.length,D=>Ec(e,B,a,M[D].value,M[D].count),bc),b&&M.length===0&&m.append(c("div",{class:"fmore",text:"No values match."})),!b&&a.hasMore};return t.listen(g,"input",()=>{o.overviewFilters[a.key]=g.value,v()}),p.append(g,m),p.append(ca(a.label)),v(),p}function Li(e,t,a){let o=e.state,r=a==="time",n=r?"__time":"__bbox",i=r?"Time range":"Bounding box",l=r?o.time:o.bbox,d=o.overviewCollapsed.has(n),p=Ya(o.overviewSpan[n]??1),u=Gt(o.overviewH[n]??1),s=c("div",{class:`fcard fcard-sp${d?" collapsed":""}${p>1?" wide":""}`,"data-key":n});s.style.gridColumn=`span ${p}`,s.dataset.rows=String(d?1:u);let g=c("div",{class:`fcard-h${l?" active":""}`},[Ri(i),R(r?E.clock:E.box,{size:14}),c("span",{class:"fh-label",text:i})]);g.append(c("span",{class:`badge${l?" on":""}`,text:l?"set":"any"})),(p!==1||u!==1)&&g.append(Za(E.reset,"Reset size",!1,()=>{delete o.overviewSpan[n],delete o.overviewH[n],He(e),e.renderOverview()},t));let m=Za(d?E.chevron:E.minimize,d?"Expand":"Minimize",d,()=>{d?o.overviewCollapsed.delete(n):o.overviewCollapsed.add(n),He(e),e.renderOverview()},t);if(m.setAttribute("aria-expanded",d?"false":"true"),g.append(m),Bi(e,t,g,n),s.append(g),d)return s.append(ca(i)),s;let A=c("div",{class:`fcard-special-body ${r?"time-body":"bbox-body"}`});if(r)A.append($o(e,t.child(),()=>e.renderOverview(),!0));else{let{editor:v,dispose:b}=er(e,t.child(),()=>e.renderOverview(),{inline:!0});t.add(b),A.append(v)}return s.append(A),s.append(ca(i)),s}function Dc(e,t){let a=e.state.overviewOrder;if(a.length===0)return t;let o=new Map(a.map((r,n)=>[r,n]));return t.slice().sort((r,n)=>(o.get(r.key)??1e6)-(o.get(n.key)??1e6))}function Ni(e){let t=e.state,a=!t.overviewStacked;if(t.overviewStacked=a,a){t.overviewSnapshot={collapsed:[...t.overviewCollapsed],span:{...t.overviewSpan},h:{...t.overviewH}};for(let o of t.facets)o.values.length&&t.overviewCollapsed.add(o.key);t.overviewCollapsed.add("__time"),t.overviewCollapsed.add("__bbox"),t.overviewStackSeen=[...t.facets.filter(o=>o.values.length).map(o=>o.key),"__time","__bbox"]}else{let o=t.overviewSnapshot;o?(t.overviewCollapsed=new Set(o.collapsed),t.overviewSpan={...o.span},t.overviewH={...o.h}):t.overviewCollapsed.clear(),t.overviewSnapshot=null,t.overviewStackSeen=[]}He(e),or(e)}function or(e){let t=e.region("overview"),a=e.roots.overviewWrap.querySelector(".overview-cap");if(a){let g=a.querySelector(".stale-pill");e.state.overviewStale&&!g?a.append(c("span",{class:"stale-pill",title:"The last attempt to refresh the facet counts failed, so these numbers may be from an earlier query. They update on the next successful search.",text:"counts may be stale"})):!e.state.overviewStale&&g&&g.remove()}let o=e.roots.overviewGrid;o.classList.toggle("stacked",e.state.overviewStacked),o.dataset.rzwired||(o.dataset.rzwired="1",Mc(e,o));let r=e.cfg.overview.mainFacets??e.state.primaryFacets,n=new Set(r),i=e.cfg.overview.mainFacets!==null||n.size>0,l=Qr(e.state);Cc(e,l.map(g=>g.key));let d=i?l.filter(g=>n.has(g.key)):l,p=i?l.filter(g=>!n.has(g.key)):[],u=[...d.map(g=>({key:g.key,make:()=>Ii(e,t,g)})),{key:"__time",make:()=>Li(e,t,"time")},{key:"__bbox",make:()=>Li(e,t,"bbox")},...e.state.overviewAddOpen?p.map(g=>({key:g.key,make:()=>Ii(e,t,g)})):[]],s=[];for(let g of Dc(e,u))s.push(g.make());if(p.length){let g=c("button",{class:"ov-addbtn",type:"button","aria-expanded":e.state.overviewAddOpen?"true":"false"},[c("span",{text:e.state.overviewAddOpen?"Hide additional facets":`Show additional facets (${p.length})`})]);t.listen(g,"click",()=>{e.state.overviewAddOpen=!e.state.overviewAddOpen,He(e),e.renderOverview()});let m=c("div",{class:"ov-addrow"},[g]);s.push(m)}re(o,...s),xc(e)}function Xa(e){let t=e.roots.overviewGrid;for(let a of t.querySelectorAll(".fcard[data-key]")){let o=a.dataset.key;if(!o||o.startsWith("__"))continue;let r=new Set(Ze(e.state,o));for(let l of a.querySelectorAll(".fval")){let d=l.querySelector(".nm")?.textContent??"",p=gt(e.state,o,d),u=r.has(d);l.classList.toggle("sel",p),l.classList.toggle("excl",u),l.setAttribute("aria-checked",p?"true":"false");let s=l.querySelector(".cb");if(s){let A=s.childElementCount>0;p&&!A?s.append(R(E.check,{size:11})):!p&&A&&(s.textContent="")}let g=l.parentElement;g?.classList.toggle("excl",u);let m=g?.querySelector(".fval-ex");m&&(m.classList.toggle("on",u),m.setAttribute("aria-pressed",u?"true":"false"))}let n=a.querySelector(".fcard-h");if(!n)continue;let i=rt(e.state,o);n.classList.toggle("active",i>0);for(let l of n.querySelectorAll(".fh-count"))l.remove();if(i>0){let l=n.querySelector(".fh-label")?.textContent??o,d=n.querySelector(".fh-label");for(let p of ia(na(e.state,o))){let u=sa(p,l,()=>e.clearFacetMode(o,p.negative),(s,g,m)=>s.addEventListener(g,m));d?.after(u),d=u}}}}var Oc=[".nc",".nc4",".cdf",".netcdf",".grib",".grib2",".grb",".grb2",".hdf",".hdf4",".hdf5",".h5",".he5"];function rr(e){let t=[e.raw?.uri,e.file];for(let a of t){if(typeof a!="string"||!a)continue;let o;try{o=new URL(a)}catch{continue}if(o.protocol!=="http:"&&o.protocol!=="https:")continue;let r=o.pathname.toLowerCase();if(Oc.some(n=>r.endsWith(n)))return o.href}return null}function da(e){try{return decodeURIComponent(new URL(e).pathname.split("/").pop()??"").replace(/[/\\]/g,"_").trim()||"download"}catch{return"download"}}function Hi(e){let t=[],a=0;for(let o of e){let r=rr(o);r?t.push({row:o,href:r}):a++}return{eligible:t,skipped:a}}var Ic=[{kind:"intake",label:"Intake catalogue",desc:"intake-esm JSON",format:"JSON",icon:()=>qe("intake",{size:16})},{kind:"stac",label:"STAC catalogue",desc:"STAC ZIP",format:"ZIP",icon:()=>qe("stac",{size:16})},{kind:"uris",label:"URI manifest",desc:"plain-text URI list",format:"TXT",icon:()=>R(E.uris,{size:16})}];function nr(e){let t=c("button",{class:"xm-item",type:"button",role:"menuitem"},[c("span",{class:"xm-ic","aria-hidden":"true"},[e.icon]),c("span",{class:"xm-text"},[c("span",{class:"xm-label",text:e.label}),c("span",{class:"xm-desc",text:e.desc})]),e.format?c("span",{class:"xm-fmt","aria-hidden":"true",text:e.format}):null]);return t.setAttribute("aria-label",`${e.label} - ${e.desc}`),e.reg.listen(t,"click",e.onPick),t}function _a(e,t){let o=[...Ic.map(n=>nr({icon:n.icon(),label:n.label,desc:n.desc,format:n.format,onPick:()=>t.onPick(n.kind),reg:e})),...t.extra??[]],r=c("div",{class:"xm",role:"menu","aria-label":t.heading},o);return e.listen(r,"keydown",n=>{let i=n,l=Array.from(r.querySelectorAll('[role="menuitem"]'));if(!l.length)return;let d=l.indexOf(document.activeElement),p=u=>{i.preventDefault(),l[Math.max(0,Math.min(l.length-1,u))]?.focus()};i.key==="ArrowDown"?p(d<0?0:d+1):i.key==="ArrowUp"?p(d<0?l.length-1:d-1):i.key==="Home"?p(0):i.key==="End"&&p(l.length-1)}),[c("div",{class:"xm-head",role:"presentation",text:t.heading}),r]}function Ui(e){return`Export all ${e.toLocaleString("en-US")} result${e===1?"":"s"}`}function Gi(e){return`Export ${e.toLocaleString("en-US")} selected file${e===1?"":"s"}`}function Lc(e,t){let a=[...e.state.pickedKeys];if(a.length===0)return;let o=a.map(r=>`file=${encodeURIComponent(r)}`).join("&");e.exportCatalogue(t,o,"file")}function Bc(e,t){let a=e.region("popover"),o=Rc(e,a);t.setAttribute("aria-expanded","true"),e.popover.open(t,_a(a,{heading:Gi(e.state.pickedKeys.size),onPick:r=>{e.popover.close(),Lc(e,r)},...o?{extra:[o]}:{}}),{placement:"below",className:"export-pop",autoFocus:!0,onClose:()=>t.setAttribute("aria-expanded","false")})}function Pc(e){let t=new Map(e.state.rows.map(a=>[a.key,a]));return[...e.state.pickedKeys].map(a=>t.get(a)).filter(a=>!!a)}function Rc(e,t){let a=Pc(e),{eligible:o,skipped:r}=Hi(a);return o.length===0?null:nr({icon:R(E.download,{size:16}),label:`Remote source files (${o.length})`,desc:r?`direct links - ${r} selected file${r===1?"":"s"} not remote`:"direct links - one download per click",onPick:()=>{e.popover.close(),Fc(e,o,r)},reg:t})}function Fc(e,t,a){let o=e.region("popover"),r=c("div",{class:"dl-list",role:"list"});for(let{row:i,href:l}of t)r.append(c("a",{class:"dl-item",role:"listitem",href:l,download:da(l),target:"_blank",rel:"noopener noreferrer",title:l},[R(E.download,{size:14}),c("span",{class:"dl-name",text:da(l)}),c("span",{class:"dl-path",text:i.file})]));let n=[c("div",{class:"dl-head",text:`${t.length} remote source file${t.length===1?"":"s"}`}),c("div",{class:"dl-note",text:a?`Click a file to download it. ${a} selected file${a===1?" is":"s are"} local or not a supported remote format, so ${a===1?"it has":"they have"} no direct link.`:"Click a file to download it."}),r];e.popover.open(e.roots.pickbar,n,{placement:"below",className:"dl-pop",autoFocus:!0,scrollBehavior:"close"}),o.add(()=>{})}function Qt(e){let t=e.region("pickbar"),a=e.roots.pickbar,o=e.state.pickedKeys.size;if(a.classList.toggle("show",o>0),o===0){re(a);return}let r=c("button",{class:"x",type:"button","aria-label":"Clear selection",title:"Clear selection"},[R(E.x,{size:16})]);t.listen(r,"click",()=>e.clearPicks());let n=c("span",{class:`cnt${o>=25?" at-cap":""}`},[c("b",{text:`${o} / ${25}`}),c("span",{text:" selected"})]),i=c("button",{class:"btn",type:"button",title:"Compare the selected files"},[R(E.info,{size:14}),c("span",{text:"Details"})]);t.listen(i,"click",()=>{e.state.detailSource="picks",e.toggleDetails(!0)});let l=c("button",{class:"btn",type:"button",title:"Download for your selection","aria-haspopup":"menu","aria-expanded":"false"},[R(E.download,{size:14}),c("span",{text:"Download"}),R(E.chevronDown,{size:12})]);t.listen(l,"click",()=>Bc(e,l));let d=e.cfg.authEnabled&&e.cfg.enableHeavyOps,p=o>10,u=!d||p,s=p?`Aggregation handles up to ${10} files - deselect ${o-10} to enable it`:e.cfg.authEnabled?e.cfg.enableHeavyOps?"Aggregation isn\u2019t wired up in this build yet":"Aggregate - data-portal not enabled":"Aggregate - needs sign-in",g=c("button",{class:`btn primary${p?" locked":""}`,type:"button",disabled:u?"true":null,title:s},[R(E.aggregate,{size:15}),c("span",{text:"Aggregate"})]);if(u||t.listen(g,"click",()=>e.toast("warn","Aggregation isn\u2019t wired up in this build yet.")),re(a,r,n,c("div",{class:"spacer"}),i,l,g),p)a.append(c("span",{class:"scope-note",style:"margin:0 0 0 4px",text:`Aggregate: max ${10} files`}));else if(!d){let m=e.cfg.authEnabled?"Aggregate needs the data-portal":"Aggregate needs sign-in";a.append(c("span",{class:"scope-note",style:"margin:0 0 0 4px",text:m}))}}var zc={zarr:"zarr",nc:"netcdf",grib:"grib"};function Nc(e){let t=e.toLowerCase(),a=t.split("/").pop()??t;if(/\.zarr(\/|$)/.test(t)||a.endsWith(".zarr"))return"zarr";let o=a.match(/\.([a-z0-9]+)$/),r=o?o[1]:"";return r==="nc"||r==="nc4"||r==="cdf"||r==="netcdf"?"nc":r==="grib"||r==="grib2"||r==="grb"||r==="grb2"?"grib":null}function Qi(e){let a=(e.split("/").pop()??e).match(/\.([a-z0-9]+)$/i);return a?a[1].slice(0,4):"file"}function ir(e){let t=Nc(e);return t?c("div",{class:`ftile ${t}`,title:`${t} file`,"aria-hidden":"true"},[qe(zc[t],{chip:!1,size:22})]):c("div",{class:"ext",title:`${Qi(e)} file`},[document.createTextNode(Qi(e))])}function cr(e){return e.state.pickedKeys.size>=25}var Wi=new WeakMap;function Ji(e){let t=Wi.get(e);return t||(t={renderedCount:-1,view:null,rowsBucket:null,epoch:-1,nodes:new Map,focusedKey:null,atCap:!1},Wi.set(e,t)),t}function Ki(e,t){let a=c("div",{class:t,text:e,title:e});return a.setAttribute("aria-label",e),a}function Zi(e,t,a){let o=e.state.pickedKeys.has(a.key),r=cr(e)&&!o,n=c("span",{class:`cb${r?" capped":""}`,role:"checkbox",tabindex:"0","aria-checked":o?"true":"false","aria-disabled":r?"true":"false","aria-label":r?`Select ${a.file} - unavailable: the ${25}-file selection limit is reached`:`Select ${a.file}`},o?[R(E.check,{size:11})]:[]),i=l=>{l.stopPropagation(),e.togglePick(a.key)};return t.listen(n,"click",i),t.listen(n,"keydown",l=>{let d=l.key;(d===" "||d==="Enter")&&(l.preventDefault(),i(l))}),n}function Yi(e,t,a){let o=c("button",{class:"kebab",type:"button","aria-label":"File actions","aria-haspopup":"menu",title:"File actions"},[R(E.kebab,{size:18})]);return t.listen(o,"click",r=>{r.stopPropagation(),Uc(e,t,o,a)}),o}function pa(e,t,a,o,r){let n=c("div",{class:"pop-item",role:"menuitem",tabindex:"0"},[c("span",{class:"pic"},[a]),c("div",{text:o})]),i=()=>{r(),e.popover.close()};return t.listen(n,"click",i),t.listen(n,"keydown",l=>{let d=l.key;(d==="Enter"||d===" ")&&(l.preventDefault(),i())}),n}function Uc(e,t,a,o){let r=t.child(),n=[pa(e,r,R(E.info,{size:16}),"Details",()=>{e.toggleDetails(!0),e.focusFile(o.key)}),pa(e,r,R(E.inspect,{size:16}),"Inspect (ncdump)",()=>{e.openInspect(o.file)}),c("div",{class:"pop-sep"}),pa(e,r,qe("intake",{size:16}),"Download Intake (.json)",()=>sr(e,"intake",o)),pa(e,r,qe("stac",{size:16}),"Download STAC (.zip)",()=>sr(e,"stac",o)),pa(e,r,R(E.uris,{size:16}),"Download URI manifest (.txt)",()=>sr(e,"uris",o))],i=rr(o);i&&(n.push(c("div",{class:"pop-sep"})),n.push(c("a",{class:"pop-item",role:"menuitem",href:i,download:da(i),target:"_blank",rel:"noopener noreferrer"},[c("span",{class:"pic"},[R(E.download,{size:16})]),c("div",{text:"Download source file"})]))),e.popover.open(a,n,{placement:"below",onClose:()=>r.flush()})}function sr(e,t,a){e.exportCatalogue(t,`file=${encodeURIComponent(a.file)}`)}function Gc(e,t,a){let o=e.state.pickedKeys.has(a.key),r=e.state.focusKey===a.key,n=c("div",{class:`row${o?" picked":""}${r?" focus":""}`,"data-key":a.key,"data-file":a.file,tabindex:"0"},[Zi(e,t,a),c("div",{class:"uricell"},[ir(a.file),c("div",{class:"meta"},[Ki(a.file,"path")])]),c("span",{class:"fs",text:a.fsType}),Yi(e,t,a)]);return t.listen(n,"click",()=>e.focusFile(a.key)),t.listen(n,"keydown",i=>{i.key==="Enter"&&e.focusFile(a.key)}),n}function Qc(e,t,a){let o=e.state.pickedKeys.has(a.key),r=e.state.focusKey===a.key,n=c("div",{class:`gcard${o?" picked":""}${r?" focus":""}`,"data-key":a.key,"data-file":a.file,tabindex:"0"},[c("div",{class:"top2"},[Zi(e,t,a),ir(a.file),c("span",{class:"fs",style:"margin-left:auto",text:a.fsType}),Yi(e,t,a)]),Ki(a.file,"path gpath")]);return t.listen(n,"click",()=>e.focusFile(a.key)),t.listen(n,"keydown",i=>{i.key==="Enter"&&e.focusFile(a.key)}),n}function Wc(){let e=c("div",{class:"skeleton-rows","aria-hidden":"true"});for(let t=0;t<6;t++)e.append(c("div",{class:"sk-row"},[c("div",{class:"sk",style:"width:30px;height:30px;border-radius:7px"}),c("div",{class:"sk",style:"flex:1;height:14px"}),c("div",{class:"sk",style:"width:48px;height:14px"})]));return e}function qc(e,t){let a=c("button",{class:"btn",type:"button",text:"Clear all"});return t.listen(a,"click",()=>e.clearAllFacets()),c("div",{class:"state-msg"},[c("div",{class:"big"},[R(E.search,{size:30})]),c("p",{text:"No files match these facets."}),a])}function Vc(e,t){let a=c("button",{class:"btn",type:"button"},[R(E.retry,{size:15}),c("span",{text:"Retry"})]);return t.listen(a,"click",()=>e.retrySearch()),c("div",{class:"state-msg err"},[c("p",{text:e.state.searchError??"Search failed."}),a])}function qi(e,t){let a=e.roots.moreWrap;re(a);let o=e.state.rows.length;if(!(e.state.totalCount>o&&o>0))return;let r=e.state.totalCount,n=e.state.search==="loading",i=Math.min(100,Math.max(1,Math.round(o/r*100))),l=c("button",{class:"btn load-next",type:"button",disabled:n?"true":null,"aria-label":"Load next 100 results"},n?[c("span",{class:"spin","aria-hidden":"true"}),c("span",{text:"Loading\u2026"})]:[c("span",{text:"Load next 100"})]);t.listen(l,"click",()=>e.loadNextPage());let d=c("div",{class:"more-info"},[c("span",{text:`Showing ${o.toLocaleString("en-US")} of ${r.toLocaleString("en-US")}`}),c("span",{class:"more-pct",text:`${i}%`})]),p=c("div",{class:"more-bar"},[c("div",{class:"more-bar-fill",style:`width:${i}%`})]);a.append(c("div",{class:"more-loader"},[d,p,l]))}function Xi(e){let t=Ji(e),a=e.roots.results,o=e.region("results-more");e.roots.listHead.hidden=!0,lr(e);let r=()=>(t.rowsBucket=e.region("results-rows"),t.rowsBucket);if(e.state.search==="loading"&&e.state.rows.length===0){r(),t.renderedCount=-1,t.view=null,a.className="",t.nodes.clear(),re(a,Wc()),re(e.roots.moreWrap);return}if(e.state.search==="error"){let s=r();t.renderedCount=-1,t.view=null,a.className="",t.nodes.clear(),re(a,Vc(e,s)),re(e.roots.moreWrap);return}if(e.state.search==="empty"||e.state.search==="loaded"&&e.state.rows.length===0){let s=r();t.renderedCount=-1,t.view=null,a.className="",t.nodes.clear(),re(a,qc(e,s)),re(e.roots.moreWrap);return}let n=e.state.rows,i=e.state.view,l=i==="list"?Gc:Qc;if(e.roots.listHead.hidden=i!=="list",t.view===i&&t.rowsBucket!==null&&t.epoch===e.state.rowsEpoch&&t.renderedCount>=0&&n.length>=t.renderedCount&&a.childElementCount===t.renderedCount){let s=t.rowsBucket;if(n.length>t.renderedCount){let g=document.createDocumentFragment();for(let m=t.renderedCount;m<n.length;m++){let A=l(e,s,n[m]);t.nodes.set(n[m].key,A),g.append(A)}a.append(g),t.renderedCount=n.length}Vi(e),qi(e,o);return}let p=r();a.className=i==="list"?"rows":"grid",a.textContent="",t.nodes.clear();let u=document.createDocumentFragment();for(let s of n){let g=l(e,p,s);t.nodes.set(s.key,g),u.append(g)}a.append(u),t.renderedCount=n.length,t.view=i,t.epoch=e.state.rowsEpoch,t.focusedKey=e.state.focusKey,t.atCap=cr(e),Vi(e),qi(e,o)}function Vi(e){e.roots.app.classList.toggle("many-results",e.state.rows.length>=500)}function ua(e,t){let a=Ji(e),o=cr(e),r=o!==a.atCap;if(a.atCap=o,t===void 0||r){for(let[i,l]of a.nodes)ji(e,i,l,o);a.focusedKey=e.state.focusKey,lr(e);return}let n=new Set(t);a.focusedKey&&a.focusedKey!==e.state.focusKey&&n.add(a.focusedKey),e.state.focusKey&&n.add(e.state.focusKey);for(let i of n){let l=a.nodes.get(i);l&&ji(e,i,l,o)}a.focusedKey=e.state.focusKey,lr(e)}function ji(e,t,a,o){let r=e.state.pickedKeys.has(t),n=e.state.focusKey===t;a.classList.toggle("picked",r),a.classList.toggle("focus",n);let i=a.querySelector(".cb");if(!i)return;i.setAttribute("aria-checked",r?"true":"false");let l=o&&!r;i.classList.toggle("capped",l),i.setAttribute("aria-disabled",l?"true":"false");let d=a.dataset.file??t;i.setAttribute("aria-label",l?`Select ${d} - unavailable: the ${25}-file selection limit is reached`:`Select ${d}`);let p=i.childElementCount>0;r&&!p?i.append(R(E.check,{size:11})):!r&&p&&(i.textContent="")}function dr(e){let t=e.state.rows.map(p=>p.key),a=t.length>25,o=a?t.slice(0,25):t,r=o.length>0&&o.every(p=>e.state.pickedKeys.has(p)),n=t.length-o.length,i=e.state.pickedKeys.size,l=r?`Clear ${i} selected`:a?`Select first ${25}`:"Select all",d=r?`Clear the ${i} selected file${i===1?"":"s"}`:a?`Select the first ${25} of ${t.length} listed files`:"Select all listed files";return{target:o,capped:a,omitted:n,willClear:r,label:l,ariaLabel:d}}function lr(e){let t=e.roots.selectAllBtn;if(!t)return;let a=t.querySelector(".cb"),o=t.querySelector(".ctrl-lbl"),r=e.state.rows.length,n=0;for(let p of e.state.rows)e.state.pickedKeys.has(p.key)&&n++;let i=dr(e),l=r>0&&i.willClear,d=n>0&&!l;t.disabled=r===0,a&&(a.classList.toggle("on",l),a.classList.toggle("mixed",d),a.textContent="",l&&a.append(R(E.check,{size:11}))),o&&(o.textContent=i.label),t.setAttribute("aria-label",i.ariaLabel),t.setAttribute("aria-checked",l?"true":d?"mixed":"false")}var jc=60,Jc=8;function Kc(e){if(!(e.state.sidebarSeeded||e.state.facets.length===0)){for(let t of e.state.facets)rt(e.state,t.key)&&e.state.sidebarOpen.add(t.key);e.state.sidebarSeeded=!0}}var Zc=e=>e.toLocaleString("en-US");function Yc(e,t,a,o,r){let n=gt(e.state,a.key,o),i=wa(e.state,a.key,o),l=Pt(e.state,a.key,o),d=At(e.state,a.key,o),p=c("span",{class:"cb"},n?[R(E.check,{size:11})]:[]),u=c("button",{class:`fval${n?" sel":""}${i?" excl":""}${l?" locked":""}`,type:"button",role:"checkbox","aria-checked":n?"true":"false","aria-disabled":l?"true":"false","aria-label":l?`${a.label}: ${o} (locked scope)`:`Include ${a.label} ${o}`,title:l?`${o} - this instance is scoped to this value`:ja(e,o,r,d)},[p,c("span",{class:"nm",text:o}),c("span",{class:"n",text:Zc(r)})]),s=la(e,r);if(s!==null&&s>0&&(u.classList.add("has-bar"),u.style.setProperty("--pct",`${s}%`)),l||t.listen(u,"click",()=>e.toggleFacet(a.key,o)),l)return c("div",{class:"fval-row"},[u]);let g=c("button",{class:`fval-ex${i?" on":""}`,type:"button","aria-pressed":i?"true":"false","aria-label":`Exclude ${a.label} ${o}`,title:i?`Stop excluding ${o}`:`Exclude ${o} from the results`,text:"\u2260"});return t.listen(g,"click",m=>{m.stopPropagation(),e.excludeFacet(a.key,o)}),c("div",{class:`fval-row${i?" excl":""}`},[u,g])}function _i(e,t,a){let o=e.state.selected[a.key]??[],r=Ze(e.state,a.key),n=rt(e.state,a.key),i=e.state.sidebarOpen.has(a.key),l=c("div",{class:`facet${i?" open":""}`,"data-key":a.key}),d=c("div",{class:`facet-head${n?" active":""}`}),p=c("button",{class:"fh-toggle",type:"button","aria-expanded":i?"true":"false"}),u=c("span",{class:"fh-text"},[c("span",{class:"fh-label",text:a.label})]);if(n){let D=[...o,...r.map(P=>`${Mi} ${P}`)];u.append(c("span",{class:"fh-sel",text:D.join(", ")}))}if(p.append(u),d.append(p),n){let D=na(e.state,a.key);for(let P of ia(D))d.append(sa(P,a.label,()=>e.clearFacetMode(a.key,P.negative),(U,q,Z)=>t.listen(U,q,Z)))}else{let D=a.hasMore?`${a.values.length}+`:String(a.values.length);d.append(c("span",{class:"badge",text:D}))}let s=c("span",{class:"chev"},[R(E.chevron,{size:12})]);d.append(s);let g=c("div",{class:"facet-body"}),m=c("div",{class:"fval-list"}),A=c("input",{class:"fval-search",type:"search",placeholder:`Search ${a.label.toLowerCase()}\u2026`,"aria-label":`Search ${a.label} values`,autocomplete:"off"}),v=c("div",{class:"fmore",text:"No matching values."});v.style.display="none";let b=D=>{let P=D.trim().toLowerCase(),U=tr(e,a),q=P?U.filter(Z=>Z.value.toLowerCase().includes(P)):U;re(m),v.style.display=q.length?"none":"",Ea(t,m,q.length,Z=>Yc(e,t,a,q[Z].value,q[Z].count),jc)},k=()=>{let D=e.state.overviewSort[a.key]??"count",P=ar(e,t,a.key,D,()=>{P.remove(),k(),b(A.value)});P.classList.add("fh-sort"),d.insertBefore(P,s)},M=!1,B=()=>{M||(M=!0,k(),a.values.length>Jc&&(t.listen(A,"input",()=>b(A.value)),t.listen(A,"keydown",D=>{D.key==="Escape"&&(A.value="",b(""))}),g.append(A)),g.append(m,v),b(""))};return i&&B(),t.listen(d,"click",()=>{let D=!l.classList.contains("open");l.classList.toggle("open",D),p.setAttribute("aria-expanded",D?"true":"false"),D?(e.state.sidebarOpen.add(a.key),B()):e.state.sidebarOpen.delete(a.key)}),l.append(d,g),l}function $i(e,t,a){let o=a==="time",r=o?!!e.state.time:!!e.state.bbox,n=c("button",{class:`special${r?" set":""}`,type:"button","aria-label":o?"Edit time range":"Edit bounding box"});n.append(c("span",{class:"lead"},[R(o?E.clock:E.box,{size:15})])),n.append(c("span",{text:o?"Time range":"Bounding box"}));let i=o?"time_select":"draw on map";return o&&e.state.time?i=`${e.state.time.from}\u2192${e.state.time.to}`:!o&&e.state.bbox&&(i="on map"),n.append(c("span",{class:"val",text:i})),t.listen(n,"click",l=>{l.stopPropagation(),o?e.openTimeEditor(n):e.openBboxEditor(n)}),n}function es(e){let t=e.region("sidebar"),a=e.roots.facetList;Kc(e);let o=new Set(e.state.primaryFacets),r=e.state.facets.filter(p=>o.size===0||o.has(p.key)),n=e.state.facets.filter(p=>o.size>0&&!o.has(p.key)),i=Object.values(e.state.selected).reduce((p,u)=>p+u.length,0)+(e.state.time?1:0)+(e.state.bbox?1:0),l=[c("span",{class:"sf-title",text:"Filter"})];if(i){let p=c("button",{class:"sf-badge",type:"button",title:"Clear all filters","aria-label":`Clear all ${i} filter${i===1?"":"s"}`},[c("span",{class:"sf-n",text:String(i)}),c("span",{class:"sf-x","aria-hidden":"true",text:"\xD7"})]);p.style.setProperty("--fb-ch",String(String(i).length)),t.listen(p,"click",()=>e.clearAllFacets()),l.push(p);let u=c("button",{class:"sf-clear",type:"button",title:"Clear all filters","aria-label":`Clear all ${i} filter${i===1?"":"s"}`,text:"Clear all"});t.listen(u,"click",()=>e.clearAllFacets()),l.push(u)}let d=[c("div",{class:"side-filterhead"},l)];if(e.state.facets.length===0)d.push(c("div",{class:"fmore",text:"Run a search to load facet values."}));else{for(let p of r)d.push(_i(e,t,p));if(d.push($i(e,t,"time")),d.push($i(e,t,"bbox")),n.length){let p=c("button",{class:"addbtn",type:"button",text:e.state.sidebarAddOpen?"\u2212 Hide additional facets":"\uFF0B Show additional facets"});if(t.listen(p,"click",()=>{e.state.sidebarAddOpen=!e.state.sidebarAddOpen,e.renderSidebar()}),d.push(p),e.state.sidebarAddOpen)for(let u of n)d.push(_i(e,t,u))}else d.push(c("div",{"aria-hidden":"true"}))}re(a,...d)}function L(e,t,a){let o=document.createElement(e);if(t)for(let[r,n]of Object.entries(t))n==null||n===!1||(r==="class"?o.className=String(n):r==="text"?o.textContent=String(n):o.setAttribute(r,String(n)));if(a)for(let r of a)r==null||r===!1||o.append(typeof r=="string"?document.createTextNode(r):r);return o}function et(e,...t){e.textContent="";for(let a of t)a==null||a===!1||e.append(typeof a=="string"?document.createTextNode(a):a)}var Xc="http://www.w3.org/2000/svg";function $a(e,t=14){let a=document.createElementNS(Xc,"svg");return a.setAttribute("viewBox","0 0 24 24"),a.setAttribute("width",String(t)),a.setAttribute("height",String(t)),a.setAttribute("fill","none"),a.setAttribute("aria-hidden","true"),a.innerHTML=e,a}var Wt=class{constructor(){this.items=[],this.disposed=!1}get isDisposed(){return this.disposed}get size(){return this.items.length}add(t){return this.disposed?(t(),()=>{}):(this.items.push(t),()=>this.remove(t))}listen(t,a,o,r){t.addEventListener(a,o,r);let n=!0,i=()=>{},l=()=>{n&&(n=!1,t.removeEventListener(a,o,r),i())};return i=this.add(l),l}setTimeout(t,a){let o=window.setTimeout(()=>{this.remove(r),t()},a),r=()=>window.clearTimeout(o);return this.add(r),o}remove(t){let a=this.items.indexOf(t);a>=0&&this.items.splice(a,1)}flush(){if(!this.disposed)for(this.disposed=!0;this.items.length;){let t=this.items.pop();try{t?.()}catch{}}}};function ts(e){try{let t=e.createElement("span");return t.setAttribute("contenteditable","plaintext-only"),t.contentEditable==="plaintext-only"}catch{return!1}}var _c={prompt:"te-prompt prompt",fixed:"te-fixed fixed",accent:"te-accent term-flav",muted:"te-muted term-scope",key:"te-key k",eq:"te-eq eq",value:"te-value v",bad:"te-bad bad"};function gr(e){return!e||e==="plain"?null:_c[e]??`te-${e}`}function $c(e){let t=gr(e.kind);return t?L("span",{class:t,text:e.text}):document.createTextNode(e.text)}function pr(e,t){et(e);for(let a of t)a.text&&e.append($c(a))}function fr(e,t,a){if(!e.contains(t))return-1;let r=e.ownerDocument.createRange();return r.selectNodeContents(e),r.setEnd(t,a),r.toString().length}function ed(e){let a=e.ownerDocument.getSelection?.();if(!a||a.rangeCount===0)return-1;let o=a.getRangeAt(0);return fr(e,o.startContainer,o.startOffset)}function ur(e){let t=e.ownerDocument.getSelection?.();if(!t||t.rangeCount===0)return null;let a=t.getRangeAt(0),o=fr(e,a.startContainer,a.startOffset),r=fr(e,a.endContainer,a.endOffset);return o<0||r<0?null:{start:o,end:r}}function hr(e,t){let a=e.ownerDocument.createTreeWalker(e,4),o=Math.max(0,t);for(;a.nextNode();){let r=a.currentNode;if(o<=r.data.length)return{node:r,at:o};o-=r.data.length}return{node:null,at:0}}function mr(e,t,a){let o=e.ownerDocument,r=o.getSelection?.();if(!r)return;let n=hr(e,t),i=hr(e,a),l=o.createRange();n.node?l.setStart(n.node,Math.min(n.at,n.node.data.length)):(l.selectNodeContents(e),l.collapse(!1)),i.node?l.setEnd(i.node,Math.min(i.at,i.node.data.length)):l.collapse(!1);try{r.removeAllRanges(),r.addRange(l)}catch{}}function td(e,t){mr(e,t,t)}var eo=class{constructor(t,a,o){this.mode="plain",this.composing=!1,this.prefixSegments=[],this.cfg=a,this.richCapable=!a.multiline&&ts(document),this.ghostClass=`te-ghost ${a.cssPrefix}-ghost`,this.richPrefix=L("span",{class:"cli-prefix cli-line","aria-hidden":"true"}),this.cmd=L("span",{class:"te-cmd",role:"textbox","aria-multiline":a.multiline?"true":"false","aria-label":a.ariaLabel,spellcheck:"false",autocapitalize:"off",tabindex:"0"}),this.ghostLayer=L("span",{class:this.ghostClass,"aria-hidden":"true"}),this.richCaret=L("span",{class:"te-caret","aria-hidden":"true"}),this.flow=L("div",{class:"te-flow"},[this.richPrefix,this.cmd,this.ghostLayer,this.richCaret]),this.plainPrefix=L("div",{class:"cli-prefix cli-prefix-block cli-line","aria-hidden":"true"}),this.hl=L("pre",{class:`${a.cssPrefix}-hl`,"aria-hidden":"true"}),this.ta=L("textarea",{class:`${a.cssPrefix}-input`,rows:"1",spellcheck:"false",autocapitalize:"off",autocomplete:"off","aria-label":a.ariaLabel,placeholder:a.placeholder}),this.plain=L("div",{class:"te-plain"},[this.plainPrefix,L("div",{class:"te-plainwrap"},[this.hl,this.ta])]),this.root=L("div",{class:`${a.cssPrefix}-wrap te-editor`},[this.flow,this.plain]);let r=()=>{this.composing||(this.cfg.multiline||this.stripNewlines(),o.onInput())};t.listen(this.ta,"input",r),t.listen(this.cmd,"input",r);for(let n of[this.ta,this.cmd])t.listen(n,"compositionstart",()=>{this.composing=!0,this.clearGhost()}),t.listen(n,"compositionend",()=>{this.composing=!1,r()}),t.listen(n,"keydown",i=>o.onKeyDown(i)),t.listen(n,"keyup",i=>{let l=i.key;["ArrowLeft","ArrowRight","ArrowUp","ArrowDown","Home","End"].includes(l)&&o.onCaretMove()}),t.listen(n,"click",()=>o.onCaretMove()),t.listen(n,"focus",()=>o.onFocus()),t.listen(n,"blur",()=>o.onBlur());t.listen(this.cmd,"paste",n=>{let i=n,l=i.clipboardData?.getData("text/plain");l!==void 0&&(i.preventDefault(),this.insertText(l),r())}),this.setMode("plain")}get inputEl(){return this.mode==="rich"?this.cmd:this.ta}get canBeRich(){return this.richCapable}setMode(t){let a=t==="rich"&&this.richCapable?"rich":"plain";if(a!==this.mode){let o=this.mode==="rich"?this.cmdText():this.ta.value;this.mode=a,a==="rich"?this.setCmdText(o):this.ta.value=o}this.root.dataset.mode=a,this.flow.style.display=a==="rich"?"":"none",this.plain.style.display=a==="plain"?"":"none",a==="rich"?this.cmd.setAttribute("contenteditable","plaintext-only"):this.cmd.removeAttribute("contenteditable"),this.clearGhost(),this.setPrefix(this.prefixSegments)}cmdText(){return this.cmd.textContent??""}setCmdText(t){this.cmd.textContent=t}clearGhost(){this.ghostLayer.textContent=""}get isComposing(){return this.composing}setPrefix(t){this.prefixSegments=t,pr(this.richPrefix,t),pr(this.plainPrefix,t),this.plainPrefix.hidden=t.length===0,t.length&&this.richPrefix.append(document.createTextNode(" "))}get value(){return this.mode==="rich"?this.cmdText():this.ta.value}set value(t){this.mode==="rich"?this.setCmdText(t):this.ta.value=t}get caret(){if(this.mode==="plain")return this.ta.selectionStart??this.ta.value.length;let t=ed(this.cmd);return t<0?this.cmdText().length:t}get selection(){if(this.mode==="plain"){let o=this.ta.value.length;return{start:this.ta.selectionStart??o,end:this.ta.selectionEnd??o}}let t=ur(this.cmd),a=this.cmdText().length;return t??{start:a,end:a}}setSelection(t,a){this.mode==="plain"?this.ta.setSelectionRange(t,a):mr(this.cmd,t,a)}setCaret(t){this.mode==="plain"?this.ta.setSelectionRange(t,t):td(this.cmd,t)}isFocused(){let t=document.activeElement;return this.mode==="rich"?t===this.cmd:t===this.ta}focus(){this.inputEl.focus()}contains(t){return!!t&&this.root.contains(t)}paint(t,a){if(this.composing)return;let o=this.isFocused();if(this.mode==="rich"){let s=o?ur(this.cmd):null;pr(this.cmd,t),s&&mr(this.cmd,s.start,s.end),this.ghostLayer.textContent=a&&o?a:"",this.cmd.classList.toggle("is-empty",this.cmdText()===""),this.cmd.dataset.placeholder=this.cfg.placeholder,this.placeRichCaret();return}this.ta.placeholder!==this.cfg.placeholder&&(this.ta.placeholder=this.cfg.placeholder);let r=L("span",{class:"te-caret"}),n=o?this.caret:this.value.length;et(this.hl);let i=(s,g)=>{let m=gr(g);return m?L("span",{class:m,text:s}):document.createTextNode(s)},l=a&&o?L("span",{class:this.ghostClass,text:a}):null,d=this.cfg.multiline,p=0,u=!1;for(let s of t){if(!s.text)continue;let g=p+s.text.length;if(!u&&n>=p&&n<=g){let m=n-p;this.hl.append(i(s.text.slice(0,m),s.kind),r),l&&d&&this.hl.append(l),this.hl.append(i(s.text.slice(m),s.kind)),u=!0}else this.hl.append(i(s.text,s.kind));p=g}u||(this.hl.append(r),l&&d&&this.hl.append(l)),l&&!d&&this.hl.append(l),this.fit()}placeRichCaret(){let t=this.flow.getBoundingClientRect();if(t.width===0&&t.height===0)return;let a=this.isFocused(),o=a?ur(this.cmd):null;if(o&&o.start!==o.end){this.richCaret.classList.add("hide"),this.ghostLayer.classList.remove("after-cursor");return}this.richCaret.classList.remove("hide");let r=this.cmdText(),n=a?Math.min(this.caret,r.length):r.length,l=this.cmd.ownerDocument.createRange(),d=hr(this.cmd,n);d.node?(l.setStart(d.node,Math.min(d.at,d.node.data.length)),l.collapse(!0)):(l.selectNodeContents(this.cmd),l.collapse(!1));let p=l.getBoundingClientRect(),u=p&&!(p.width===0&&p.height===0)?p:null;if(!u){let A=this.cmd.getClientRects(),v=n<=0?A[0]:A[A.length-1];if(v&&v.height>0)u={left:n<=0?v.left:v.right,top:v.top,height:v.height};else{let b=this.cmd.getBoundingClientRect();u=b.height>0?{left:b.left,top:b.top,height:b.height}:null}}if(!u)return;let s=this.cmd.getClientRects(),g=0;for(let A of s)A.height>g&&(g=A.height);let m=g>0?Math.min(u.height,g):u.height;this.richCaret.style.left=`${Math.round((u.left-t.left)*100)/100}px`,this.richCaret.style.top=`${Math.round((u.top-t.top)*100)/100}px`,m>0&&(this.richCaret.style.height=`${Math.round(m*100)/100}px`),this.ghostLayer.classList.toggle("after-cursor",this.ghostLayer.textContent!==""&&n===r.length)}refreshCaret(){this.mode==="rich"&&this.placeRichCaret()}fit(){if(this.mode!=="plain"){this.refreshCaret();return}if(this.ta.offsetParent===null&&this.ta.clientWidth===0)return;this.ta.style.height="auto";let t=this.ta.scrollHeight;t>0&&(this.ta.style.height=`${t}px`)}insertText(t){let a=this.value,{start:o,end:r}=this.selection,n=this.cfg.multiline?t:t.replace(/[\r\n]+/g," ");this.value=a.slice(0,o)+n+a.slice(r),this.setCaret(o+n.length)}stripNewlines(){let t=this.value;if(!/[\n\r]/.test(t))return;let a=this.caret;this.value=t.replace(/[\n\r]+/g," "),this.setCaret(Math.min(a,this.value.length))}};function as(e,t){let a=gr(e.kind);return a?L("span",{class:a,text:e.text}):t.createTextNode(e.text)}var os={kebab:'<circle cx="12" cy="5" r="1.6" fill="currentColor"/><circle cx="12" cy="12" r="1.6" fill="currentColor"/><circle cx="12" cy="19" r="1.6" fill="currentColor"/>',terminal:'<rect x="3" y="4" width="18" height="16" rx="2.5" stroke="currentColor" stroke-width="1.7"/><path d="M7 9l3 3-3 3M12.5 15h4" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"/>'};var to=`/* styles.css - @freva-org/freva-client-terminal. Scoped under \`.freva-term\`, which IS the window
   root, and injected into the package's own subtree, so a host without the databrowser still gets a
   complete terminal. Host tokens (--shadow, --mono, --ui, --border-2) are inherited inside a themed
   app; the fallbacks below make standalone use work on a bare page. */

.freva-term {
  --font: var(--mono, "JetBrains Mono", ui-monospace, "SF Mono", Menlo, monospace);
  --shadow: var(--host-shadow, 0 1px 2px rgba(0, 0, 0, 0.5), 0 6px 20px rgba(0, 0, 0, 0.4));
  --border-2: var(--host-border-2, #2c4267);
  font-family: var(--font);
}
.freva-term *,
.freva-term *::before,
.freva-term *::after {
  box-sizing: border-box;
}

/* Keeps its own dark tokens so it stays dark in day theme. Colours are user-chosen and PERSISTED;
   each preset ships its own foreground so text can never be unreadable. Defaults to black. */
.freva-term {
  --term-bg: #0b0f16;
  --term-fg: #d8e2f2;
  --term-alpha: 0.94;
  /* Text size multiplier on CONTENT only: every content size is \`calc(Npx * var(--term-scale))\`,
     no chrome size is - a title bar grown with its text stops fitting its own controls and drifts
     the traffic lights away from the OS conventions they imitate. */
  --term-scale: 1;
  /* token colours; the light presets override these (see [data-term-light]) */
  --term-prompt: #28c840;
  --term-key: #8fb6ff;
  --term-val: #f0b86b;
  --term-dim: #6f7f9c;
  --term-ghost: #4d5d78;
  --term-hint: #7fd7c4;
  --term-ph: #5b6a86;
  display: none;
  border-radius: 10px;
  overflow: visible;
  /* a hint of the page behind the window - tunable in the \u22EE menu, persisted */
  background: color-mix(in srgb, var(--term-bg) calc(var(--term-alpha) * 100%), transparent);
  backdrop-filter: blur(10px) saturate(120%);
  -webkit-backdrop-filter: blur(10px) saturate(120%);
  border: 1px solid rgba(255, 255, 255, 0.16);
  box-shadow: var(--shadow);
  color: var(--term-fg);
}
/* corners still clip their own content, but the window doesn't clip its popovers (the \u22EE menu) */
.freva-term .term-bar {
  border-radius: 10px 10px 0 0;
}
/* light presets (e.g. Paper): the token palette has to flip too, or the text is unreadable */
.freva-term[data-term-light="true"] {
  --term-prompt: #1f7a33;
  --term-key: #2d5fb8;
  --term-val: #a05a12;
  --term-dim: #6a7383;
  --term-ghost: #a3acbb;
  --term-hint: #1d7d6c;
  --term-ph: #99a2b0;
  border-color: rgba(0, 0, 0, 0.18);
}
.freva-term[data-term-light="true"] .term-bar {
  background: color-mix(in srgb, var(--term-bg) 88%, #000 6%);
  border-bottom-color: rgba(0, 0, 0, 0.12);
}
.freva-term[data-term-light="true"] .term-menu {
  background: color-mix(in srgb, var(--term-bg) 94%, #000 5%);
  border-color: rgba(0, 0, 0, 0.18);
}
.freva-term[data-term-light="true"] .tm-item {
  color: #22262b;
}
.freva-term[data-term-light="true"] .te-menu,
.freva-term[data-term-light="true"] .py-menu {
  border-color: rgba(0, 0, 0, 0.16);
  background: rgba(0, 0, 0, 0.03);
}
.freva-term[data-term-light="true"] .cmd-tab {
  color: #5b6472;
}
.freva-term[data-term-light="true"] .cmd-tab:not(.on):hover {
  background: rgba(0, 0, 0, 0.05);
}
.freva-term[data-term-light="true"] .term-kebab,
.freva-term[data-term-light="true"] .copy-btn {
  color: #5b6472;
}
.freva-term[data-term-light="true"] .term-kebab:hover,
.freva-term[data-term-light="true"] .copy-btn:hover {
  background: rgba(0, 0, 0, 0.06);
  color: #22262b;
}
.freva-term .term-body {
  border-radius: 0 0 10px 10px;
  overflow-y: auto;
  overflow-x: hidden;
}
.freva-term.zoomed {
  left: 20px !important;
  top: 20px !important;
  right: 20px !important;
  bottom: 20px !important;
  width: auto !important;
  height: auto !important;
  transform: none !important;
}
/* Gmail-style dock: minimized collapses to just the title bar, pinned to the bottom. The
   horizontal position is a variable so the dock can be dragged left/right (never up/down). */
.freva-term.minimized {
  height: auto !important;
  /* \`.freva-term.show\` sets \`min-height: 220px\` for an OPEN window and \`height: auto\` cannot shrink
     past a minimum, so the dock needs the minimum reset too, not just the height. */
  min-height: 0 !important;
  width: 300px !important;
  right: var(--dock-right, 20px) !important;
  bottom: 0 !important;
  left: auto !important;
  top: auto !important;
  transform: none !important;
  border-radius: 10px 10px 0 0;
  cursor: pointer;
}
.freva-term.minimized .term-body {
  display: none;
  min-height: 0;
  height: 0;
}
.freva-term .term-bar {
  cursor: move;
  user-select: none;
}
.freva-term.minimized .term-bar {
  cursor: pointer;
}
/* maximized windows don't move (Gmail) - say so with the cursor */
.freva-term.zoomed .term-bar {
  cursor: default;
}
.freva-term .tl,
.freva-term .cmd-tab,
.freva-term .copy-btn,
.freva-term .term-add,
.freva-term .term-info-btn,
.freva-term .term-bg-btn {
  cursor: pointer;
}
.freva-term .term-resize {
  position: absolute;
  right: 2px;
  bottom: 2px;
  width: 14px;
  height: 14px;
  cursor: nwse-resize;
  z-index: 2;
  background: linear-gradient(
    135deg,
    transparent 50%,
    var(--border-2) 50%,
    var(--border-2) 60%,
    transparent 60%,
    transparent 72%,
    var(--border-2) 72%,
    var(--border-2) 82%,
    transparent 82%
  );
}
.freva-term.minimized .term-resize,
.freva-term.zoomed .term-resize {
  display: none;
}
.freva-term .term-add {
  font-size: 13px;
  font-weight: 700;
  color: #7b8aa6;
  background: none;
  border: none;
  padding: 2px 7px;
  border-radius: 6px;
  cursor: pointer;
}
.freva-term .term-add:hover {
  color: #8fb6ff;
  background: rgba(79, 141, 247, 0.15);
}
.freva-term .cmd-tab .tab-x {
  margin-left: 6px;
  opacity: 0.6;
  cursor: pointer;
}
.freva-term .cmd-tab .tab-x:hover {
  opacity: 1;
}
.freva-term .term-bar {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 9px 12px;
  background: #0e1626;
  border-bottom: 1px solid #1b2942;
}
/* Two groups with a gap between them. \`min-width: 0\` on the start group so a long session title
   ellipsises instead of pushing the right group off the edge; \`flex: 0 0 auto\` on the end group so
   its controls never squeeze - \`fitBar\` moves them to the kebab menu instead. */
.freva-term .term-bar-group {
  display: flex;
  align-items: center;
  gap: 8px;
  min-width: 0;
}
.freva-term .term-bar-start {
  flex: 0 1 auto;
}
.freva-term .term-bar-end {
  flex: 0 0 auto;
}
.freva-term .spacer {
  flex: 1 1 auto;
  min-width: 8px;
}
.freva-term .traffic {
  display: inline-flex;
  gap: 8px;
  align-items: center;
  margin-right: 4px;
}
.freva-term .tl {
  width: 12px;
  height: 12px;
  border-radius: 999px;
  border: none;
  cursor: pointer;
  padding: 0;
  display: inline-grid;
  place-items: center;
}
.freva-term .tl.close {
  background: #ff5f56;
}
.freva-term .tl.min {
  background: #febc2e;
}
.freva-term .tl.zoom {
  background: #28c840;
}
.freva-term .tl span {
  font-size: 9px;
  line-height: 1;
  font-weight: 700;
  color: rgba(0, 0, 0, 0.55);
  opacity: 0;
  transition: opacity 0.1s;
}
.freva-term .traffic:hover .tl span {
  opacity: 1;
}

/* OS-specific window controls */
/* Windows: labelled buttons on the RIGHT, order min \xB7 max \xB7 close, red close hover. */
.freva-term[data-os="windows"] .traffic {
  gap: 0;
  margin: 0 0 0 4px;
}
.freva-term[data-os="windows"] .tl {
  width: 34px;
  height: 26px;
  border-radius: 0;
  background: transparent !important;
  color: #aab8d4;
}
.freva-term[data-os="windows"] .tl span {
  opacity: 1;
  color: currentColor;
  font-size: 12px;
}
.freva-term[data-os="windows"] .tl.min {
  order: 1;
}
.freva-term[data-os="windows"] .tl.zoom {
  order: 2;
}
.freva-term[data-os="windows"] .tl.close {
  order: 3;
}
.freva-term[data-os="windows"] .tl:hover {
  background: #1b2942 !important;
  color: #fff;
}
.freva-term[data-os="windows"] .tl.close:hover {
  background: #e81123 !important;
  color: #fff;
}
.freva-term[data-os="windows"] .tl.min span::before {
  content: "\\2013";
} /* \u2013 */
.freva-term[data-os="windows"] .tl.zoom span::before {
  content: "\\25A1";
} /* \u25A1 */
.freva-term[data-os="windows"] .tl.close span::before {
  content: "\\2715";
} /* \u2715 */
.freva-term[data-os="windows"] .tl span {
  font-size: 0;
}
.freva-term[data-os="windows"] .tl span::before {
  font-size: 12px;
}

/* Linux (GNOME-ish): rounded symbolic buttons on the RIGHT. */
.freva-term[data-os="linux"] .traffic {
  gap: 7px;
  margin: 0 0 0 4px;
}
.freva-term[data-os="linux"] .tl {
  width: 22px;
  height: 22px;
  border-radius: 999px;
  background: #26364f !important;
  color: #d3ddf0;
}
.freva-term[data-os="linux"] .tl.min {
  order: 1;
}
.freva-term[data-os="linux"] .tl.zoom {
  order: 2;
}
.freva-term[data-os="linux"] .tl.close {
  order: 3;
}
.freva-term[data-os="linux"] .tl:hover {
  background: #33496b !important;
}
.freva-term[data-os="linux"] .tl.close {
  background: #3a2730 !important;
  color: #ffb4a8;
}
.freva-term[data-os="linux"] .tl.close:hover {
  background: #c0392b !important;
  color: #fff;
}
.freva-term[data-os="linux"] .tl span {
  opacity: 1;
  color: currentColor;
  font-size: 0;
}
.freva-term[data-os="linux"] .tl.min span::before {
  content: "\\2013";
}
.freva-term[data-os="linux"] .tl.zoom span::before {
  content: "\\25A1";
}
.freva-term[data-os="linux"] .tl.close span::before {
  content: "\\2715";
}
.freva-term[data-os="linux"] .tl span::before {
  font-size: 11px;
}
.freva-term .cmd-tab {
  font-size: 12px;
  font-weight: 600;
  color: #7b8aa6;
  padding: 4px 9px;
  border-radius: 6px;
  cursor: pointer;
  border: none;
  background: none;
  font-family: inherit;
}
.freva-term .cmd-tab.on {
  background: rgba(79, 141, 247, 0.18);
  color: #8fb6ff;
}
.freva-term .copy-ic {
  width: 30px;
  height: 28px;
  border-radius: 7px;
  border: 1px solid #243349;
  background: #121d31;
  color: #aebbd4;
  cursor: pointer;
  display: inline-grid;
  place-items: center;
}
.freva-term .copy-ic:hover {
  color: #fff;
  border-color: #34507c;
}
.freva-term .copy-ic.done {
  color: #28c840;
  border-color: #28c840;
}
.freva-term .term-body {
  padding: 14px;
  font-family: var(--mono);
  font-size: calc(12.5px * var(--term-scale));
  line-height: 1.85;
  color: var(--term-fg);
  position: relative;
  flex: 1 1 auto;
  min-height: 0;
  /* Fills the window height and scrolls inside, so the footer stays pinned to the bottom. */
  overflow-y: auto;
  overflow-x: hidden;
}
.freva-term.zoomed .term-body {
  max-height: none;
}
.freva-term .term-body::-webkit-scrollbar {
  width: 10px;
}
.freva-term .term-body::-webkit-scrollbar-thumb {
  background: rgba(255, 255, 255, 0.16);
  border-radius: 6px;
}
.freva-term.minimized .term-body {
  display: none;
}
.freva-term.zoomed .term-body {
  min-height: 220px;
}
.freva-term .prompt {
  color: var(--term-prompt);
  font-weight: 700;
}
.freva-term .fixed {
  color: var(--term-fg);
  font-weight: 600;
  opacity: 0.92;
}
.freva-term .fixed.cont {
  color: #44566f;
  font-weight: 400;
}
.freva-term .k {
  color: var(--term-key);
}
.freva-term .v {
  color: var(--term-val);
}
.freva-term .eq {
  color: var(--term-dim);
}
.freva-term .term-flav {
  color: #c79bf0;
}
.freva-term .term-scope {
  color: #6f7f9c;
  opacity: 0.85;
} /* the base scope: shown so a copied command reproduces results, but visibly not typed */
.freva-term .bad {
  color: #f0795f;
  text-decoration: underline wavy #f0795f;
  text-underline-offset: 3px;
}
.freva-term .cli-line {
  white-space: pre-wrap;
  word-break: break-word;
}
.freva-term .term-edit {
  margin-top: 2px;
}
.freva-term .te-wrap {
  position: relative;
  font-family: var(--mono);
  font-size: calc(12.5px * var(--term-scale));
  line-height: 1.85;
}
.freva-term .te-hl,
.freva-term .te-input {
  margin: 0;
  font: inherit;
  line-height: inherit;
  white-space: pre-wrap;
  word-break: break-word;
  padding: 2px 0;
  border: none;
}
.freva-term .te-hl {
  position: absolute;
  inset: 0;
  color: #d7e2f4;
  pointer-events: none;
}
.freva-term .te-input {
  position: relative;
  display: block;
  width: 100%;
  background: transparent;
  color: transparent;
  caret-color: transparent;
  outline: none;
  resize: none;
  overflow: hidden;
}
.freva-term .te-input::placeholder {
  color: #44566f;
}
.freva-term.fallback .te-hl {
  display: none;
}
.freva-term.fallback .te-input {
  color: var(--term-fg);
  caret-color: var(--term-fg);
}
.freva-term .te-warn {
  display: none;
  margin-top: 8px;
  font-family: var(--ui);
  font-size: calc(11.5px * var(--term-scale));
  color: #f0b86b;
  background: rgba(240, 121, 95, 0.12);
  border: 1px solid rgba(240, 121, 95, 0.4);
  border-radius: 6px;
  padding: 5px 9px;
}
.freva-term .te-warn.show {
  display: block;
}
.freva-term .py-view {
  margin: 0;
  white-space: pre-wrap;
  word-break: break-word;
  font-family: var(--mono);
  font-size: calc(12.5px * var(--term-scale));
  line-height: 1.9;
  padding: 4px 2px;
}
/* The generic multi-line edit row. NO \`gap\`: the gutter is exactly the read-only prompt column's
   width (4 monospace columns for python) and the editable layers match it, so a gap pushes typed
   kwargs 8px right of the \`>>> \` lines. Above the per-tab rules, so a tab's own metrics win. */
.freva-term .term-editrow {
  display: flex;
  align-items: flex-start;
  min-width: 0;
}
.freva-term .term-editrow > .te-editor {
  flex: 1;
  min-width: 0;
}
.freva-term .term-gutter {
  flex-shrink: 0;
  white-space: pre;
  color: var(--term-dim);
  font-family: var(--font);
  font-size: calc(12.5px * var(--term-scale));
  line-height: 1.65;
  user-select: none;
}

.freva-term .py-line {
  display: flex;
  align-items: baseline;
}
/* The prompt (\`>>> \` / \`... \`) and the editable line's gutter MUST be the same width, or the typed
   kwargs won't line up under the read-only ones. Both are exactly 4 monospace columns. */
.freva-term .py-prompt,
.freva-term .py-gutter {
  display: inline-block;
  flex: 0 0 4ch;
  width: 4ch;
  padding-right: 0;
}
.freva-term .py-prompt {
  color: var(--term-prompt);
  font-weight: 700;
}
.freva-term .py-line.cont .py-prompt {
  color: #44566f;
  font-weight: 400;
}
.freva-term .py-code {
  color: var(--term-key);
}
.freva-term .py-ml {
  display: flex;
  align-items: flex-start;
}
.freva-term .py-gutter {
  white-space: pre;
  color: #44566f;
  font-family: var(--mono);
  font-size: calc(12.5px * var(--term-scale));
  line-height: 1.9;
  user-select: none;
}
.freva-term .py-wrap {
  position: relative;
  flex: 1;
  min-width: 40px;
}
/* BOTH text layers carry the 4-space indent of the read-only \`    key=\` lines; padding the WRAPPER
   cannot do it, as absolutely-positioned \`.py-hl\` ignores it and drifts out of alignment. */
.freva-term .py-hl,
.freva-term .py-input {
  font-family: var(--mono);
  font-size: calc(12.5px * var(--term-scale));
  line-height: 1.9;
  white-space: pre-wrap;
  word-break: break-word;
  margin: 0;
  padding: 0 0 0 4ch;
}
.freva-term .py-hl {
  position: absolute;
  inset: 0;
  color: var(--term-val);
  pointer-events: none;
}
/* caret-color TRANSPARENT: the block cursor is drawn here, a native caret would be a second one. */
.freva-term .py-input {
  position: relative;
  display: block;
  width: 100%;
  background: transparent;
  color: transparent;
  caret-color: transparent;
  border: none;
  outline: none;
  resize: none;
  overflow: hidden;
}
.freva-term .py-input::placeholder {
  color: #44566f;
}
.freva-term .py-ghost {
  color: #4d5d78;
}
.freva-term .py-out {
  color: #aeb9cf;
  margin: 0 0 2px;
  white-space: pre-wrap;
  word-break: break-word;
}
.freva-term .py-list {
  font-family: var(--mono);
  font-size: calc(12.5px * var(--term-scale));
}

.freva-term .term-host {
  color: #6f9cf0;
  word-break: break-all;
}

/* Windows: blue title bar */
.freva-term[data-os="windows"] .term-bar {
  background: linear-gradient(#1257c4, #0e46a0);
  border-bottom-color: #0a3a86;
}
.freva-term[data-os="windows"] .cmd-tab {
  color: #cfe0ff;
}
.freva-term[data-os="windows"] .cmd-tab.on {
  background: rgba(255, 255, 255, 0.18);
  color: #fff;
}
.freva-term[data-os="windows"] .term-add,
.freva-term[data-os="windows"] .copy-ic {
  color: #dceaff;
}
.freva-term[data-os="windows"] .tl {
  color: #eaf1ff;
}
.freva-term[data-os="windows"] .tl:hover {
  background: rgba(255, 255, 255, 0.16) !important;
  color: #fff;
}
.freva-term[data-os="windows"] .tl.close:hover {
  background: #e81123 !important;
  color: #fff;
}

/* Inline ghost autocomplete */
.freva-term .te-ghost {
  color: #4d5d78;
}
.freva-term .te-hint {
  font-size: 10.5px;
  color: #4d5d78;
  margin-top: 3px;
  font-family: var(--mono);
}
.freva-term.fallback .te-hint {
  display: none;
}

/* In-terminal completion menu (shell-style, not a floating popover) */
.freva-term .te-menu,
.freva-term .py-menu {
  display: none;
  margin: 6px 0 2px;
  border: 1px solid #26364f;
  border-radius: 6px;
  max-height: 168px;
  overflow-y: auto;
  background: rgba(255, 255, 255, 0.03);
}
.freva-term .te-menu.show,
.freva-term .py-menu.show {
  display: block;
}
.freva-term .tm-item {
  display: flex;
  justify-content: space-between;
  gap: 14px;
  padding: 3px 10px;
  font-family: var(--mono);
  font-size: calc(12px * var(--term-scale));
  color: #c7d4ea;
  cursor: pointer;
}
.freva-term .tm-item.hl {
  background: rgba(79, 141, 247, 0.22);
  color: #fff;
}
.freva-term .tm-item:hover {
  background: rgba(79, 141, 247, 0.12);
}
.freva-term .tm-val {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.freva-term .tm-cnt {
  color: #6f7f9c;
  flex-shrink: 0;
}

/* Terminal: real tabs, blinking cursor, read-only prefix, panels */

/* title bar + tabs */
.freva-term .term-bar {
  background: color-mix(in srgb, var(--term-bg) 82%, #fff 6%);
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}
.freva-term .cmd-tab {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 5px 10px 6px;
  border: 1px solid transparent;
  border-bottom: none;
  border-radius: 7px 7px 0 0;
  margin-bottom: -1px;
  color: #93a3bd;
  font-size: 12px;
  line-height: 1.4;
}
.freva-term .cmd-tab .tab-ic {
  display: inline-flex;
  opacity: 0.75;
}
.freva-term .cmd-tab[data-cmd="cli"] .tab-ic {
  color: #7ee0a8;
} /* bash */
.freva-term .cmd-tab[data-cmd="py"] .tab-ic {
  color: #f0c04d;
} /* python */
.freva-term .cmd-tab.on {
  background: var(--term-bg);
  color: var(--term-fg);
  border-color: rgba(255, 255, 255, 0.14);
  border-bottom: 1px solid var(--term-bg);
}
.freva-term .cmd-tab.on .tab-ic {
  opacity: 1;
}
.freva-term .cmd-tab:not(.on):hover {
  background: rgba(255, 255, 255, 0.05);
  color: var(--term-fg);
}

/* [copy] / info / colour controls */
.freva-term .copy-btn {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  font-family: var(--mono);
  font-size: 11.5px;
  color: #93a3bd;
  background: none;
  border: none;
  padding: 3px 7px;
  border-radius: 6px;
}
.freva-term .copy-btn .cb-caret {
  color: #7ee0a8;
  opacity: 0.8;
}
.freva-term .copy-btn:hover {
  color: var(--term-fg);
  background: rgba(255, 255, 255, 0.07);
}
.freva-term .copy-btn:hover .cb-caret {
  opacity: 1;
}
.freva-term .copy-btn.done,
.freva-term .copy-btn.done .cb-caret {
  color: #7ee0a8;
  opacity: 1;
}
.freva-term .term-kebab {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 24px;
  height: 22px;
  color: #93a3bd;
  background: none;
  border: none;
  border-radius: 6px;
}
.freva-term .term-kebab:hover {
  color: var(--term-fg);
  background: rgba(255, 255, 255, 0.08);
}

/* blinking block cursor (a real terminal, not a text field) */
.freva-term .te-caret {
  display: inline-block;
  width: 7px;
  height: 1.05em;
  vertical-align: text-bottom;
  background: var(--term-fg);
  animation: te-blink 1.05s step-end infinite;
}
@keyframes te-blink {
  0%,
  45% {
    opacity: 1;
  }
  50%,
  100% {
    opacity: 0;
  }
}
@media (prefers-reduced-motion: reduce) {
  .freva-term .te-caret {
    animation: none;
  }
}

/* Blinks focused or not - the "start typing here" cue: solid when focused, hollow when not. */
.freva-term .te-wrap:not(:focus-within) .te-caret,
.freva-term .py-wrap:not(:focus-within) .te-caret {
  background: transparent;
  box-shadow: inset 0 0 0 1px var(--term-fg);
}
.freva-term .py-fixedline .py-code,
.freva-term .py-ro {
  color: #7f8da3;
  font-style: italic;
  opacity: 0.82;
}

/* overflow (\\22ee) menu: terminal settings; the install guide lives in app-level Help */
.freva-term .term-menu {
  display: none;
  position: absolute;
  right: 8px;
  top: 42px;
  z-index: 120;
  min-width: 214px;
  padding: 8px;
  border-radius: 8px;
  border: 1px solid rgba(255, 255, 255, 0.16);
  background: color-mix(in srgb, var(--term-bg) 88%, #fff 8%);
  box-shadow: 0 14px 34px rgba(0, 0, 0, 0.5);
}
.freva-term .term-menu.show {
  display: block;
}
.freva-term .tmn-h {
  font-size: 10.5px;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: #7b8aa6;
  font-family: var(--font);
  margin: 2px 2px 7px;
}
.freva-term .tmn-item {
  display: block;
  width: 100%;
  text-align: left;
  margin-top: 4px;
  padding: 6px 8px;
  border: none;
  border-radius: 6px;
  background: none;
  color: var(--term-fg);
  font-family: var(--font);
  font-size: 12px;
  text-decoration: none;
  cursor: pointer;
}
.freva-term .tmn-item:hover {
  background: rgba(255, 255, 255, 0.09);
}
.freva-term .tmn-item:disabled {
  opacity: 0.45;
  cursor: default;
}
.freva-term .tmn-item:disabled:hover {
  background: none;
}
/* A row that throws state away is coloured like one, so it cannot be hit by muscle memory. */
.freva-term .tmn-danger {
  color: #ff9a8b;
}
.freva-term[data-term-light="true"] .tmn-danger {
  color: #a3352a;
}

/* The nested appearance group, OPEN by default: the nesting bounds the controls, not hides them. */
.freva-term .tmn-sub {
  display: flex;
  align-items: center;
  justify-content: space-between;
  width: 100%;
  padding: 6px 8px;
  border: none;
  border-radius: 6px;
  background: none;
  color: var(--term-fg);
  font-family: var(--font);
  font-size: 12px;
  cursor: pointer;
}
.freva-term .tmn-sub:hover {
  background: rgba(255, 255, 255, 0.09);
}
.freva-term .tmn-sub-chev {
  transition: transform 0.12s;
  color: #7b8aa6;
}
.freva-term .tmn-group.open .tmn-sub-chev {
  transform: rotate(90deg);
}
.freva-term .tmn-subpanel {
  display: none;
  padding: 6px 2px 8px;
  margin-bottom: 4px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}
.freva-term .tmn-group.open .tmn-subpanel {
  display: block;
}

/* THE SAME GROUP, OPENING SIDEWAYS when the host has placed it among its own rows: unrolled inline
   there, the panel pushes everything below it off the bottom of the menu, so it hangs off the row
   instead and the row stays one row tall open or shut. LEFT by default because the menu hangs from
   the right of the title bar; \`--right\` is measured when there is no room that side. Neither is a
   coordinate: the panel anchors to the group's own box, so a dragged window carries it along. */
/* The menu stops clipping while a flyout is out: it is a scroll container, and a scroll container
   clips both axes whatever the other one says. See the note beside the class in \`window.ts\`. */
.freva-term .term-menu.has-flyout {
  overflow: visible;
}
.freva-term .tmn-group--side {
  position: relative;
}
/* The offset and the cap are \`placeFlyout\`'s, measured against the viewport when the panel opens.
   The fallbacks - level with the row, no cap - apply when it has not run. */
.freva-term .tmn-group--side.open .tmn-subpanel {
  display: block;
  position: absolute;
  top: var(--tmn-flyout-top, -8px);
  right: calc(100% + 10px);
  z-index: 1;
  min-width: 196px;
  max-height: var(--tmn-flyout-max, none);
  overflow-y: auto;
  overscroll-behavior: contain;
  margin: 0;
  padding: 8px;
  border: 1px solid rgba(255, 255, 255, 0.16);
  border-radius: 8px;
  background: color-mix(in srgb, var(--term-bg) 88%, #fff 8%);
  box-shadow: 0 14px 34px rgba(0, 0, 0, 0.5);
}
.freva-term .tmn-group--side.tmn-group--right.open .tmn-subpanel {
  right: auto;
  left: calc(100% + 10px);
}
/* Pointing at where the panel appears, rather than down into a list that is not there. */
.freva-term .tmn-group--side .tmn-sub-chev {
  transform: rotate(180deg);
}
.freva-term .tmn-group--side.tmn-group--right .tmn-sub-chev {
  transform: none;
}
.freva-term .tmn-group--side.open .tmn-sub-chev {
  transform: rotate(180deg);
}
.freva-term .tmn-group--side.tmn-group--right.open .tmn-sub-chev {
  transform: none;
}
.freva-term .tmn-group--side.open > .tmn-sub {
  background: rgba(255, 255, 255, 0.09);
}
.freva-term .tmn-sub:disabled {
  opacity: 0.45;
  cursor: default;
}
.freva-term .tmn-block + .tmn-block {
  margin-top: 8px;
  padding-top: 6px;
  border-top: 1px solid rgba(255, 255, 255, 0.1);
}

/* colour palette (persisted) */
.freva-term .term-bg-panel {
  display: flex;
  gap: 7px;
  flex-wrap: wrap;
  padding: 0 2px 8px;
  margin-bottom: 4px;
}
.freva-term .bg-sw {
  width: 24px;
  height: 24px;
  border-radius: 6px;
  border: 1px solid rgba(255, 255, 255, 0.2);
  cursor: pointer;
}
.freva-term .bg-sw.on {
  outline: 2px solid #8fb6ff;
  outline-offset: 1px;
}

/* completion menu: the empty state says so, instead of showing nothing */
.freva-term .tm-empty {
  color: #6f7f9c;
  font-style: italic;
  cursor: default;
}
.freva-term .tm-empty:hover {
  background: none;
}

/* bash: the prefix and the command share ONE inline text flow - \`.te-flow\` further down, plain
   inline siblings in one pre-wrap container. An absolute prefix layer, a custom indent on the
   input, a width threshold or a "prefix on its own line" mode each break wrapping - an indent
   shifts only the first line - so all four are absent, and a test asserts they stay absent. */

/* no focus ring inside the terminal: the BLINKING CURSOR is the focus cue */
.freva-term .te-input:focus-visible,
.freva-term .py-input:focus-visible {
  outline: none;
}
.freva-term .te-wrap,
.freva-term .py-wrap {
  border: none;
  box-shadow: none;
}

/* Terminal: hint vs suggestion, and a bar that survives a narrow window */

/* The GHOST is the only thing Tab accepts, so nothing else may look like it: it keeps the
   "type-ahead" grey, and the placeholder and hint are italic in a different hue entirely. */
.freva-term .te-ghost,
.freva-term .py-ghost {
  color: var(--term-ghost);
  font-style: normal;
}
.freva-term .te-input::placeholder,
.freva-term .py-input::placeholder {
  color: var(--term-ph);
  font-style: italic;
  opacity: 0.8;
}
.freva-term .te-hint {
  margin-top: 6px;
  font-family: var(--font);
  font-size: 11px;
  font-style: italic;
  color: var(--term-hint);
  opacity: 0.85;
  letter-spacing: 0.01em;
}
.freva-term .te-hint kbd,
.freva-term .tm-empty {
  font-family: var(--mono);
  font-style: normal;
}
/* keycaps: subtle fill, thicker bottom border, hairline shadow; per theme, dark tokens default. */
.freva-term .te-hint kbd {
  display: inline-flex;
  align-items: center;
  padding: 0 5px;
  margin: 0 1px;
  min-width: 16px;
  justify-content: center;
  border-radius: 4px;
  font-size: 10px;
  line-height: 1.7;
  color: var(--term-fg);
  background: rgba(255, 255, 255, 0.09);
  border: 1px solid rgba(255, 255, 255, 0.2);
  border-bottom-width: 2px;
  box-shadow: 0 1px 0 rgba(0, 0, 0, 0.3);
}
.freva-term[data-term-light="true"] .te-hint kbd {
  background: rgba(0, 0, 0, 0.06);
  border-color: rgba(0, 0, 0, 0.22);
  box-shadow: 0 1px 0 rgba(0, 0, 0, 0.12);
}
/* Terminal footer: the keyboard-hint strip under the body. The window is overflow:visible for the
   \u22EE menu, so the footer rounds its OWN bottom corners. Hidden when docked or in the fallback. */
.freva-term .term-foot {
  padding: 5px 12px;
  border-top: 1px solid rgba(255, 255, 255, 0.09);
  border-radius: 0 0 10px 10px;
  background: color-mix(in srgb, var(--term-bg) 96%, #000 5%);
  display: flex;
  align-items: center;
  min-height: 26px;
  flex-shrink: 0;
}
.freva-term .term-foot .te-hint {
  margin: 0;
  opacity: 0.9;
}
.freva-term[data-term-light="true"] .term-foot {
  border-top-color: rgba(0, 0, 0, 0.1);
  background: color-mix(in srgb, var(--term-bg) 92%, #000 4%);
}
.freva-term.minimized .term-foot,
.freva-term.fallback .term-foot {
  display: none;
}
/* the "how to type this" rows in the menu are guidance, not completions */
.freva-term .tm-empty {
  color: var(--term-hint);
  font-style: italic;
  cursor: default;
}
.freva-term .tm-empty:hover {
  background: none;
}

/* narrow window: the window controls must never be pushed out of the bar */
.freva-term.show {
  min-width: 340px;
}
.freva-term .term-bar {
  flex-wrap: nowrap;
}
.freva-term .traffic,
.freva-term .term-add,
.freva-term .copy-btn,
.freva-term .term-kebab {
  flex: 0 0 auto;
}
.freva-term .cmd-tab {
  flex: 0 1 auto;
  min-width: 0;
  overflow: hidden;
}
.freva-term .cmd-tab .tab-label {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
/* below this WINDOW width (set by a ResizeObserver on the terminal itself - a viewport media
   query can't see the window's own size) the tab labels give way before the controls do */
.freva-term.narrow .cmd-tab .tab-label {
  display: none;
}
.freva-term.narrow .copy-btn .cb-word {
  display: none;
}

/* opacity slider in the \u22EE menu */
.freva-term .term-scale {
  width: 100%;
  accent-color: #8fb6ff;
}
.freva-term .tmn-alpha {
  padding: 2px 2px 8px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
  margin-bottom: 4px;
}
.freva-term .term-alpha {
  width: 100%;
  accent-color: #8fb6ff;
  cursor: pointer;
}

/* completion menu placement: the window sits bottom-right, so a list under the caret is often below
   the fold. With no room beneath, the menu FLIPS above the prompt, as a shell does; explicit flex
   \`order\` values (not DOM order) move it alone. */
.freva-term .term-view {
  display: flex;
  flex-direction: column;
}
.freva-term .term-view > .term-head {
  order: 10;
}
.freva-term .term-view > .term-edit {
  order: 20;
}
.freva-term .term-view > .tm-menu {
  order: 30;
}
.freva-term .term-view > .te-warn {
  order: 40;
}
.freva-term .term-view > .term-foot-lines {
  order: 50;
}
.freva-term .term-view.menu-above > .tm-menu {
  order: 15; /* between the read-only header and the prompt line */
  margin: 0 0 6px;
}

/* NO blanket dimming of \`.term-head .te-prompt\`: it would sweep up python's real \`>>>\` lines along
   with the \`...\` continuations. Continuations are dimmed by their own kind, \`.te-contprompt\`. */

/* LAYOUT OVERRIDES for this package's markup. These come last on purpose: everything above is the
   shared visual identity, and this block re-states only the geometry that differs. */

/* Container-relative window. \`position: fixed\` sized against \`100vw/58vh\` assumes the window owns
   the top-level page; in a clipped, \`overflow: hidden\`, transformed host container that puts it
   outside its own component and out of reach. It is clamped against its MOUNT, host-supplied. */
.freva-term.show {
  display: flex;
  flex-direction: column;
  position: absolute;
  z-index: 80;
  right: 20px;
  bottom: 20px;
  width: min(760px, calc(100% - 40px));
  height: min(58%, 440px);
  min-height: 220px;
  box-shadow:
    0 24px 60px rgba(0, 0, 0, 0.55),
    0 0 0 1px rgba(255, 255, 255, 0.06);
}

/* The shared inline flow: the immutable prefix and the editable command are ordinary inline
   content in ONE \`pre-wrap\` flow, so the command starts right after the last prefix token at every
   width and a wrapped line continues at the container's normal left edge, like a shell. */
.freva-term .te-flow {
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  word-break: normal;
  font-family: var(--font);
  font-size: calc(12.5px * var(--term-scale));
  line-height: 1.65;
  padding: 2px 0;
}
.freva-term .cli-prefix {
  /* explicitly NOT positioned and NOT nowrap - it wraps with the command */
  position: static;
  display: inline;
  white-space: inherit;
  padding: 0;
  pointer-events: none;
  user-select: none;
}
.freva-term .te-cmd {
  display: inline;
  outline: none;
  white-space: inherit;
  min-width: 1px;
  color: var(--term-fg);
  /* TRANSPARENT, like the plain fallback's textarea: the cursor is the blinking BLOCK drawn beside
     it, so a native caret on top would be a second one - \`.py-input\` does the same for that. */
  caret-color: transparent;
}
.freva-term .te-cmd.is-empty::after {
  content: attr(data-placeholder);
  color: var(--term-ph);
  pointer-events: none;
}
/* The block cursor in the rich flow cannot be an inline node: inserting one at the caret would
   split the EDITABLE text, where only the buffer may live. It is an absolutely positioned sibling
   on the caret's client rect from \`paint()\`, already on the right visual line, wrapped or not. */
.freva-term .te-flow {
  position: relative;
}
.freva-term .te-flow > .te-caret {
  position: absolute;
  left: 0;
  top: 0;
  vertical-align: baseline;
}
/* Hidden only while a RANGE is selected: the selection highlight is the cue, not a block cursor. */
.freva-term .te-flow > .te-caret.hide {
  display: none;
}
/* Unfocused: the hollow parked box, the terminal convention and the "start typing here" cue.
   \`.te-flow\` is the focus scope for the rich surface, as \`.te-wrap\` is for the plain one. */
.freva-term .te-flow:not(:focus-within) > .te-caret {
  background: transparent;
  box-shadow: inset 0 0 0 1px var(--term-fg);
}
/* With a suggestion showing and the cursor at the end of the buffer, the ghost starts where the
   cursor is drawn; reserving the cursor's width puts it AFTER the block rather than underneath. */
.freva-term .te-flow > .te-ghost.after-cursor {
  padding-left: 7px;
}
/* Presentation-only siblings of the editable node: they must not take a click, a caret or a
   selection, or the user could put the cursor "inside" a suggestion that is not in the buffer. */
.freva-term .te-flow > .te-ghost,
.freva-term .te-flow > .te-caret {
  pointer-events: none;
  user-select: none;
}

/* The explicit plain-textarea fallback */
.freva-term .te-plain {
  display: none;
}
/* The reveal keys off \`.te-editor\`, which is on EVERY editor root; the per-tab \`\${prefix}-wrap\`
   class is not (python's root is \`.py-wrap\`), so keying off \`.te-wrap\` never reveals python. */
.freva-term .te-editor[data-mode="plain"] .te-plain {
  display: block;
}
.freva-term .te-editor[data-mode="plain"] .te-flow {
  display: none;
}
.freva-term .te-editor[data-mode="rich"] .te-plain {
  display: none;
}
.freva-term .te-plainwrap {
  position: relative;
}
.freva-term .cli-prefix-block {
  display: block;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  position: static;
  padding: 2px 0 0;
}
.freva-term .te-plain .te-hl,
.freva-term .te-plain .te-input {
  text-indent: 0;
}

/* Tabs / views */
.freva-term .term-view {
  display: flex;
  flex-direction: column;
  min-width: 0;
}
.freva-term .term-head,
.freva-term .term-foot-lines {
  font-family: var(--font);
  font-size: calc(12.5px * var(--term-scale));
  line-height: 1.65;
  color: var(--term-dim);
}
.freva-term .term-line {
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}
.freva-term .term-edit {
  min-width: 0;
}
.freva-term .tm-menu {
  display: none;
}
.freva-term .tm-menu.show {
  display: block;
}

/* Segment colours. The package emits kind-prefixed classes (\`te-key\`, \`te-value\`, \u2026) rather than
   bare \`.k\`/\`.v\`/\`.eq\`, because an unprefixed class in a shared subtree collides. Same palette. */
.freva-term .te-prompt {
  color: var(--term-prompt);
  font-weight: 700;
}
/* NO generic \`.te-fixed\` colour: one would sit after \`.fixed\` here and repaint BOTH tabs' fixed
   text the same grey. bash's command and flags are foreground-weight; python's \`import\` line and
   \`databrowser(\` / \`)\` are KEY-coloured, being code not a command line. \`.fixed\` carries bash's,
   python states its own below. */
.freva-term .term-view[data-cmd="py"] .te-fixed {
  color: var(--term-key);
  font-weight: 400;
  opacity: 1;
}
/* The \`...\` gutter of a read-only continuation line is quiet; a real \`>>>\` prompt is not. Both are
   painted from segments, so the host says which is which and this styles the answer. */
.freva-term .te-contprompt {
  color: #44566f;
  font-weight: 400;
}
.freva-term .te-key {
  color: var(--term-key);
}
.freva-term .te-value {
  color: var(--term-val);
}
.freva-term .te-eq {
  color: var(--term-dim);
}
.freva-term .te-accent {
  color: var(--term-hint);
}
.freva-term .te-muted {
  color: var(--term-dim);
  font-style: italic;
  opacity: 0.82;
}
.freva-term .te-bad {
  color: #ff6b6b;
  text-decoration: underline wavy currentColor 1px;
  text-underline-offset: 3px;
}
.freva-term .te-ghost {
  color: var(--term-ghost);
  pointer-events: none;
}

/* Settings menu: it must survive being MINIMIZED. A minimized window is pinned to the bottom of
   its container, so a menu anchored under the title bar opens off the bottom edge, out of reach;
   \`.above\` flips it over the bar. Both placements are clamped by the controller's max-height. */
.freva-term .term-menu {
  max-height: none;
  overflow-y: auto;
}
.freva-term .term-menu.above {
  top: auto;
  bottom: calc(100% + 6px);
}
.freva-term.minimized .term-menu.show {
  display: block;
}
/* The minimized dock hides most of the window, but the settings menu is still reachable from it,
   so these three opt back IN to the display each actually uses. \`display: revert\` will not do for
   the colour panel: the UA default (\`block\`) drops the flex row and with it the 7px swatch gaps. */
.freva-term.minimized .term-bg-panel {
  display: flex;
  gap: 7px;
  flex-wrap: wrap;
}
.freva-term.minimized .bg-sw {
  flex: 0 0 24px; /* a fixed track, so a wrapped row keeps the same rhythm as an unwrapped one */
}
.freva-term.minimized .tmn-alpha,
.freva-term.minimized .tmn-item {
  display: revert;
}

/* confirmation */

/* The window's own question, in its own colours: inside \`.freva-term\`, absolutely positioned
   against it, so it dims this window and nothing else of the host's. Not \`<dialog showModal()>\`
   either - that paints in the top layer, above anything a host ordered above the window. */
.freva-term .term-confirm-scrim {
  position: absolute;
  inset: 0;
  z-index: 30;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 16px;
  background: rgba(4, 8, 16, 0.62);
  border-radius: inherit;
}
.freva-term .term-confirm {
  max-width: 30rem;
  padding: 16px 18px;
  border: 1px solid #2a3d5e;
  border-radius: 10px;
  background: #101b2e;
  color: #dbe4f3;
  box-shadow: 0 18px 48px rgba(0, 0, 0, 0.55);
}
.freva-term .term-confirm-title {
  margin: 0 0 6px;
  font-size: 14px;
  font-weight: 600;
  line-height: 1.35;
}
.freva-term .term-confirm-body {
  margin: 0 0 14px;
  font-size: 12.5px;
  line-height: 1.55;
  color: #aebbd4;
}
.freva-term .term-confirm-actions {
  display: flex;
  justify-content: flex-end;
  gap: 8px;
}
.freva-term .term-confirm-btn {
  padding: 6px 12px;
  border: 1px solid #2a3d5e;
  border-radius: 7px;
  background: #16233a;
  color: #dbe4f3;
  font: inherit;
  font-size: 12.5px;
  cursor: pointer;
}
.freva-term .term-confirm-btn:hover {
  border-color: #34507c;
  color: #fff;
}
.freva-term .term-confirm-btn:focus-visible {
  outline: 2px solid #4f8df7;
  outline-offset: 2px;
}
.freva-term .term-confirm-ok {
  border-color: #34507c;
  background: #1d3050;
}
/* The destructive answer LOOKS destructive and is still not focused: focus is what protects. */
.freva-term .term-confirm-danger {
  border-color: #8a3b34;
  background: #3a1c19;
  color: #ffd5cf;
}
.freva-term .term-confirm-danger:hover {
  border-color: #c1584c;
  color: #fff;
}

/* Reduced motion */
@media (prefers-reduced-motion: reduce) {
  .freva-term .te-caret {
    animation: none;
  }
}
`;var fa=[{id:"black",label:"Black",bg:"#0b0f16",fg:"#d8e2f2"},{id:"ink",label:"Ink",bg:"#131a26",fg:"#d8e2f2"},{id:"graphite",label:"Graphite",bg:"#22262b",fg:"#e4e7ea"},{id:"midnight",label:"Midnight",bg:"#0d1b2a",fg:"#cfe3f7"},{id:"forest",label:"Forest",bg:"#10201a",fg:"#cfe9d9"},{id:"plum",label:"Plum",bg:"#1d1526",fg:"#e6d7f2"},{id:"paper",label:"Paper",bg:"#f4f1ea",fg:"#22262b",light:!0}],rs=360,ns=220,ad=4,is="black",ss=.85,ls=.55,cs=1,ds=.8,ps=1.6,us=new WeakMap;function od(e,t){try{if(typeof CSSStyleSheet=="function"&&Array.isArray(e.adoptedStyleSheets)){if(!us.has(e)){let o=new CSSStyleSheet;o.replaceSync(to),e.adoptedStyleSheets=[...e.adoptedStyleSheets,o],us.set(e,!0)}return}}catch{}let a=L("style",{type:"text/css"});a.textContent=to,t.append(a)}function fs(e){try{let t=document.createElement("textarea");t.value=e,t.setAttribute("readonly",""),t.style.position="fixed",t.style.opacity="0",document.body.appendChild(t),t.select();let a=document.execCommand("copy");return t.remove(),a}catch{return!1}}function rd(e){let t=a=>!!(a&&a.isCollapsed===!1&&a.toString().trim()!=="");if(t(e?.ownerDocument?.defaultView?.getSelection?.()))return!1;for(let a=e;a;a=a.parentElement){let o=a.shadowRoot;if(t(o?.getSelection?.()))return!1}return!0}var hs=(e,t,a)=>Math.min(a,Math.max(t,Number.isFinite(e)?e:t));function Ar(e,t){if(t!=="title")for(let a of e.querySelectorAll("[title]")){let o=a.getAttribute("title")??"";a.removeAttribute("title"),a.setAttribute(t,o),!a.getAttribute("aria-label")&&!(a.textContent??"").trim()&&a.setAttribute("aria-label",o)}}function br(e,t={}){let a=new Wt,o=t.storage,r=L("div",{class:"freva-term cmd","data-os":t.os??"mac"});od(e.ownerDocument??document,r);let n=(T,N,G)=>L("button",{class:`tl ${T}`,type:"button",title:N,"aria-label":N},[L("span",{text:G})]),i=n("close",t.closeLabel??"Close","\u2715"),l=n("min","Minimize","\u2013"),d=n("zoom","Maximize","+"),p=L("button",{class:"copy-btn",type:"button",title:t.copyTitle??"Copy command","aria-label":t.copyTitle??"Copy command"},[L("span",{class:"cb-caret",text:"\u276F"}),L("span",{class:"cb-word",text:t.copyLabel??"copy"})]),u=L("button",{class:"term-kebab",type:"button",title:"Terminal settings","aria-label":"Terminal settings","aria-haspopup":"true","aria-expanded":"false"},[$a(os.kebab,14)]),s=L("span",{class:"traffic"},[i,l,d]),g=L("div",{class:"spacer"}),m=L("div",{class:"term-bar-group term-bar-start"}),A=L("div",{class:"term-bar-group term-bar-end"}),v=L("div",{class:"term-bar"},[m,g,A]),b=L("div",{class:"term-body"}),k=t.foot===!0?L("div",{class:"term-foot"}):null,M=L("div",{class:"term-bg-panel",role:"listbox","aria-label":"Terminal colour"}),B=L("input",{class:"term-alpha",type:"range",min:String(ls),max:"1",step:"0.01","aria-label":"Terminal opacity",title:"Terminal opacity"}),D=L("input",{class:"term-scale",type:"range",min:String(ds),max:String(ps),step:"0.05","aria-label":"Terminal text size",title:"Terminal text size"}),P=L("button",{class:"tmn-item tmn-reset",type:"button",role:"menuitem",text:"Reset appearance"}),U=L("div",{class:"tmn-subpanel",id:"term-appearance"},[L("div",{class:"tmn-h",text:"Colour"}),M,L("div",{class:"tmn-h",text:"Opacity"}),L("div",{class:"tmn-alpha"},[B]),L("div",{class:"tmn-h",text:"Text size"}),L("div",{class:"tmn-alpha tmn-scale"},[D]),P]),q=L("button",{class:"tmn-sub",type:"button","aria-expanded":"true","aria-controls":"term-appearance"},[L("span",{class:"tmn-sub-label",text:"Terminal settings"}),L("span",{class:"tmn-sub-chev","aria-hidden":"true",text:"\u203A"})]),Z=L("div",{class:"tmn-group open"},[q,U]),K=q.querySelector(".tmn-sub-label"),ne=L("div",{class:"tmn-sections"}),W=L("div",{class:"term-menu",role:"menu"},[ne]);a.listen(q,"click",()=>{let T=Z.classList.toggle("open");q.setAttribute("aria-expanded",T?"true":"false");let N=Z.classList.contains("tmn-group--side");W.classList.toggle("has-flyout",T&&N),T&&N&&pe()});let le=8;function $(){let T=U.getBoundingClientRect(),N=q.getBoundingClientRect(),G=W.getBoundingClientRect(),te=window.innerHeight||0,se=T.width||220;Z.classList.toggle("tmn-group--right",G.left-se<le);let Me=Math.max(U.offsetHeight,U.scrollHeight,T.height),Fe=te-le*2,Le=Me>Fe;Z.style.setProperty("--tmn-flyout-max",Le?`${Math.max(120,Fe)}px`:"none");let Ve=Le?Fe:Me,Vt=N.top-le,Ee=te-le-Ve,Xe=Math.max(le,Math.min(Vt,Ee));Z.style.setProperty("--tmn-flyout-top",`${Math.round(Xe-N.top)}px`)}function pe(){$(),a.setTimeout(()=>{Z.classList.contains("open")&&Z.classList.contains("tmn-group--side")&&$()},0)}function w(T){ne.textContent="",Z.remove();let N=!1;for(let G of T){if(G.items.length===0)continue;let te=L("div",{class:"tmn-block"});G.title&&te.append(L("div",{class:"tmn-h",text:G.title}));for(let se of G.items){if(se.appearance===!0){N=!0,K&&(K.textContent=se.label),Z.classList.add("tmn-group--side"),Z.classList.remove("open","tmn-group--right"),q.setAttribute("aria-expanded","false"),q.disabled=se.disabled===!0,te.append(Z);continue}te.append(y(se))}ne.append(te)}N||(Z.classList.remove("tmn-group--side","tmn-group--right"),Z.classList.add("open"),q.disabled=!1,q.setAttribute("aria-expanded","true"),W.prepend(Z)),Ar(ne,t.tooltipAttribute??"title")}function y(T){let N=`tmn-item${T.danger?" tmn-danger":""}`;if(T.href)return L("a",{class:N,href:T.href,target:"_blank",rel:"noopener noreferrer",role:"menuitem",text:T.label,...T.title?{title:T.title}:{}});let G=L("button",{class:N,type:"button",role:"menuitem",text:T.label,disabled:T.disabled===!0,...T.title?{title:T.title}:{}});return a.listen(G,"click",()=>{Oe=G,Ae(),T.onSelect?.()}),G}let S=L("div",{class:"term-resize","aria-hidden":"true"}),O={mac:"start",windows:"end",linux:"end"},F=t.controlsSide??O[t.os??"mac"]??"start";F==="start"&&m.append(s),t.copyText&&A.append(p),A.append(u),F==="end"&&A.append(s),r.append(v,b,...k?[k]:[],W,S),e.append(r),w([...t.menuSections??[],...t.menuItems&&t.menuItems.length>0?[{items:t.menuItems}]:[]]);let I=()=>t.bounds?.()??e,ee=()=>I().getBoundingClientRect(),V=null;function X(){V={left:r.style.left,top:r.style.top,width:r.style.width,height:r.style.height}}function he(){r.style.left="",r.style.top="",r.style.right="",r.style.bottom="",r.style.transform="",r.style.width="",r.style.height=""}function ue(){V&&(r.style.left=V.left,r.style.top=V.top,r.style.width=V.width,r.style.height=V.height,V.left&&(r.style.right="auto",r.style.bottom="auto",r.style.transform="none"),V=null)}let ce=()=>r.classList.contains("minimized"),ke=()=>r.classList.contains("zoomed");function me(T){if(T===ce())return;let N=T&&ke();r.classList.toggle("minimized",T),r.classList.remove("zoomed"),Ae(),T?(X(),he()):ue(),N&&t.onMaximize?.(!1),t.onMinimize?.(T)}function we(T){if(T===ke())return;let N=T&&ce();r.classList.toggle("zoomed",T),r.classList.remove("minimized"),Ae(),T?(X(),he()):ue(),N&&t.onMinimize?.(!1),t.onMaximize?.(T)}a.listen(i,"click",()=>pt()),a.listen(l,"click",()=>me(!ce())),a.listen(d,"click",()=>we(!ke()));let Y=null,f=null,x=null,H=!1,j=".tl, .copy-btn, .term-kebab, .term-menu",oe=T=>!!T.closest(j)||(t.dragExclude?!!T.closest(t.dragExclude):!1);a.listen(v,"click",T=>{if(ce()&&!oe(T.target)){if(H){H=!1;return}me(!1)}}),a.listen(v,"mousedown",T=>{let N=T;if(oe(N.target)||ke())return;let G=ee(),te=r.getBoundingClientRect();if(ce()){x={startX:N.clientX,startRight:G.right-te.right,moved:!1},H=!1,N.preventDefault();return}Y={dx:N.clientX-te.left,dy:N.clientY-te.top,started:!1},N.preventDefault()}),a.listen(S,"mousedown",T=>{let N=T,G=r.getBoundingClientRect();f={x:N.clientX,y:N.clientY,w:G.width,h:G.height},N.preventDefault(),N.stopPropagation()}),a.listen(window,"mousemove",T=>{let N=T,G=ee();if(Y){if(!Y.started){let Ve=r.getBoundingClientRect();r.classList.remove("zoomed"),r.style.transform="none",r.style.right="auto",r.style.bottom="auto",r.style.left=`${Ve.left-G.left}px`,r.style.top=`${Ve.top-G.top}px`,Y.started=!0}let te=r.getBoundingClientRect(),se=Math.max(0,G.width-te.width),Me=Math.max(0,G.height-te.height),Fe=Math.min(se,Math.max(0,N.clientX-Y.dx-G.left)),Le=Math.min(Me,Math.max(0,N.clientY-Y.dy-G.top));r.style.left=`${Fe}px`,r.style.top=`${Le}px`}else if(x){let te=N.clientX-x.startX;if(!x.moved&&Math.abs(te)>ad-1&&(x.moved=!0,H=!0),x.moved){let se=r.getBoundingClientRect().width||300,Me=Math.max(0,G.width-se),Fe=Math.min(Me,Math.max(0,x.startRight-te));r.style.setProperty("--dock-right",`${Fe}px`)}}else if(f){let te=r.getBoundingClientRect(),se=Math.max(rs,G.right-te.left-8),Me=Math.max(ns,G.bottom-te.top-8);r.style.width=`${Math.min(se,Math.max(rs,f.w+(N.clientX-f.x)))}px`,r.style.height=`${Math.min(Me,Math.max(ns,f.h+(N.clientY-f.y)))}px`,t.onResize?.()}}),a.listen(window,"mouseup",()=>{Y=null,f=null,x=null});function ie(){let T=r.getBoundingClientRect().width;T>0&&r.classList.toggle("narrow",T<460)}if(a.listen(window,"resize",()=>{Et()&&(ie(),t.onResize?.())}),typeof ResizeObserver=="function"){let T=new ResizeObserver(()=>{ie(),t.onResize?.()});T.observe(r),a.add(()=>T.disconnect())}function Ae(){W.classList.remove("show","has-flyout"),u.setAttribute("aria-expanded","false"),Z.classList.contains("tmn-group--side")&&(Z.classList.remove("open"),q.setAttribute("aria-expanded","false"))}let xe=0,ae=null,Oe=null;function Be(T){if(ae)return Promise.resolve(!1);Ae();let N=T.returnFocus??Oe??(r.contains(r.ownerDocument.activeElement)?r.ownerDocument.activeElement:u);Oe=null;let G=`term-confirm-title-${xe}`,te=`term-confirm-body-${xe}`;xe+=1;let se=L("button",{class:"term-confirm-btn",type:"button",text:T.cancelLabel??"Cancel"}),Me=L("button",{class:`term-confirm-btn term-confirm-ok${T.danger?" term-confirm-danger":""}`,type:"button",text:T.confirmLabel??"Confirm"}),Fe=L("div",{class:"term-confirm",role:"alertdialog","aria-modal":"true","aria-labelledby":G,"aria-describedby":te},[L("h2",{class:"term-confirm-title",id:G,text:T.title}),L("p",{class:"term-confirm-body",id:te,text:T.body}),L("div",{class:"term-confirm-actions"},[se,Me])]),Le=L("div",{class:"term-confirm-scrim"},[Fe]),Ve=Ee=>{if(!ae||ae.root!==Le)return;let{resolve:Xe,returnFocus:ze}=ae;ae=null,Le.remove(),ze?.isConnected&&ze.focus(),Xe(Ee)};Le.addEventListener("keydown",Ee=>{let Xe=Ee.key;if(Xe==="Escape"){Ee.preventDefault(),Ee.stopPropagation(),Ve(!1);return}if(Xe!=="Tab")return;let ze=[se,Me],Dt=ze.indexOf(r.ownerDocument.activeElement),ga=Ee.shiftKey?Dt-1:Dt+1;Ee.preventDefault(),ze[(ga%ze.length+ze.length)%ze.length]?.focus()}),Le.addEventListener("mousedown",Ee=>{Ee.target===Le&&Ee.preventDefault()}),se.addEventListener("click",()=>Ve(!1)),Me.addEventListener("click",()=>Ve(!0)),r.append(Le);let Vt=new Promise(Ee=>{ae={resolve:Ee,root:Le,returnFocus:N}});return se.focus(),Vt}function ct(){let T=ce();W.classList.toggle("above",T);let N=ee(),G=r.getBoundingClientRect(),te=v.getBoundingClientRect().height||40;T?(W.style.bottom="calc(100% + 6px)",W.style.top="auto",W.style.maxHeight=`${Math.max(120,Math.floor(G.top-N.top-12))}px`):(W.style.top=`${Math.round(te+2)}px`,W.style.bottom="auto",W.style.maxHeight=`${Math.max(120,Math.floor(N.bottom-G.top-te-16))}px`)}a.listen(u,"click",T=>{T.stopPropagation();let N=W.classList.toggle("show");u.setAttribute("aria-expanded",N?"true":"false"),N&&ct()}),a.listen(document,"mousedown",T=>{W.classList.contains("show")&&(T.target.closest(".term-menu, .term-kebab")||Ae())});function tt(T,N=!0){let G=fa.find(te=>te.id===T)??fa[0];r.style.setProperty("--term-bg",G.bg),r.style.setProperty("--term-fg",G.fg),r.setAttribute("data-term-light",G.light?"true":"false");for(let te of r.querySelectorAll(".bg-sw")){let se=te.dataset.bg===G.id;te.classList.toggle("on",se),te.setAttribute("aria-selected",se?"true":"false")}N&&o?.setTheme(G.id)}for(let T of fa)M.append(L("button",{class:"bg-sw",type:"button",role:"option","data-bg":T.id,title:T.label,"aria-label":T.label,style:`background:${T.bg}`}));a.listen(M,"click",T=>{let N=T.target.closest(".bg-sw");N?.dataset.bg&&tt(N.dataset.bg)});function Ue(T,N=!0){let G=hs(T,ls,1);r.style.setProperty("--term-alpha",String(G)),B.value=String(G),N&&o?.setAlpha(G)}a.listen(B,"input",()=>Ue(Number(B.value)));function be(T,N=!0){let G=hs(T,ds,ps);r.style.setProperty("--term-scale",String(G)),D.value=String(G),N&&o?.setTextScale?.(G)}let Ye=T=>{be(T),t.onResize?.()};a.listen(D,"input",()=>Ye(Number(D.value)));function dt(){tt(is),Ue(ss),Ye(cs)}a.listen(P,"click",()=>{Ae(),dt()}),tt(o?.getTheme()??is,!1),Ue(o?.getAlpha()??ss,!1),be(o?.getTextScale?.()??cs,!1),a.listen(p,"click",()=>{let T=t.copyText?.()??"",N=p.querySelector(".cb-word"),G=p.querySelector(".cb-caret"),te=t.copyLabel??"copy",se=()=>{p.classList.add("done"),G&&(G.textContent="\u2713"),N&&(N.textContent="copied"),a.setTimeout(()=>{p.classList.remove("done"),G&&(G.textContent="\u276F"),N&&(N.textContent=te)},1200)},Me=()=>t.onCopyFailed?.("Copy failed - select and copy manually.");navigator.clipboard?.writeText?navigator.clipboard.writeText(T).then(se,()=>{fs(T)?se():Me()}):fs(T)?se():Me()}),a.listen(b,"mousedown",T=>{T.target===b&&T.preventDefault()}),a.listen(b,"click",T=>{let N=T.target;N.closest(".tm-item, .term-menu, a, button, textarea, input, [contenteditable]")||rd(N)&&t.onBodyActivate?.()});function Et(){return r.classList.contains("show")}function pt(){r.classList.remove("show"),Ae(),t.onHide?.(),t.onClose?.()}function Ge(){r.classList.add("show"),r.classList.remove("minimized"),t.onShow?.(),ie()}let qt=t.tooltipAttribute??"title";return Ar(r,qt),{el:r,bar:v,barSpacer:g,barStart:m,barEnd:A,addBarControl(T,N="start"){N==="end"?A.insertBefore(T,u):m.append(T),ie()},body:b,foot:k,settings:W,show:Ge,hide:pt,toggle(T){T??!Et()?Ge():pt()},isShown:Et,isMinimized:ce,setMinimized:me,isMaximized:ke,setMaximized:we,closeSettings:Ae,setMenuSections:w,confirm:Be,applyTheme:tt,applyAlpha:Ue,applyTextScale:Ye,resetAppearance:dt,fitBar:ie,applyTooltips(){Ar(r,qt)},destroy(){a.flush(),r.remove()}}}function ha(e){return L("kbd",{text:e})}function ms(e){return e?[ha("Tab \u21E5")," now leaves the terminal \xB7 type to keep completing"]:[ha("Tab \u21E5")," completes \xB7 ",ha("\u2193")," lists all options \xB7 ",ha("Esc")," then ",ha("Tab \u21E5")," to leave"]}function vr(e,t){if(!t.tabs.length)throw new Error("freva-client-terminal: at least one tab is required");let a=new Wt,o=br(e,{os:t.os,storage:t.storage,tooltipAttribute:t.tooltipAttribute,bounds:t.bounds,foot:!0,dragExclude:".cmd-tab, .term-add",copyText:()=>s.get(m)?.tab.copyText()??"",menuItems:t.menuItems,onClose:t.onClose,onCopyFailed:t.onCopyFailed,onShow:()=>{Y(),v()||s.get(m)?.editor.focus()},onHide:()=>$(),onMinimize:f=>{f&&$()},onResize:()=>{me();for(let f of s.values())f.editor.fit()},onBodyActivate:()=>{v()||s.get(m)?.editor.focus()}}),r=e.ownerDocument,n=o.el,i=o.bar,l=o.body,d=o.foot,p=L("button",{class:"term-add",type:"button",title:"Reopen closed tab","aria-label":"Reopen closed tab",text:"+"}),u=L("span",{class:"te-hint"},ms(!1));d.append(u);let s=new Map,g=new Set(t.tabs.map(f=>f.id)),m=t.activeTab&&g.has(t.activeTab)?t.activeTab:t.tabs[0].id,A=!1,v=()=>t.fallback?.()===!0,b=f=>{et(u,...ms(f))},k=()=>{A=!0,b(!0)},M=()=>{A=!1},B=f=>f.shiftKey?!0:A?(A=!1,!0):!1,D=[],P=0,U=null,q=null,Z=()=>U!==null&&D.length>0;function K(){U&&(et(U),D.forEach((f,x)=>{U.append(L("div",{class:`tm-item${x===P?" hl":""}`,role:"option","data-i":String(x)},[L("span",{class:"tm-val",text:f.value}),f.count===null||f.count===void 0?null:L("span",{class:"tm-cnt",text:f.count.toLocaleString("en-US")})]))}))}function ne(f){let x=f.parentElement,H=l.getBoundingClientRect();if(!x||H.height===0)return;let j=s.get(m)?.editor.inputEl;if(!j)return;let oe=j.getBoundingClientRect(),ie=H.bottom-oe.bottom-10,Ae=oe.top-H.top-10,xe=96,ae=ie<xe&&Ae>ie;x.classList.toggle("menu-above",ae);let Oe=Math.max(xe,Math.floor(ae?Ae:ie));f.style.maxHeight=`${Math.min(220,Oe)}px`,f.scrollIntoView?.({block:"nearest"})}function W(f,x,H){if(!x.length){$();return}let j=U===f?D[P]?.value??null:null;D=x;let oe=j===null?-1:x.findIndex(ie=>ie.value===j);P=oe>=0?oe:0,q=H,U=f,K(),f.classList.add("show"),ne(f)}function le(f,x){D=[],P=0,q=null,U=f,et(f,L("div",{class:"tm-item tm-empty",text:x})),f.classList.add("show"),ne(f)}function $(){U&&(U.parentElement?.classList.remove("menu-above"),U.style.maxHeight="",U.classList.remove("show"),et(U)),D=[],P=0,q=null,U=null}function pe(f){D.length&&(P=(P+f+D.length)%D.length,K(),U?.children[P]?.scrollIntoView?.({block:"nearest"}))}function w(){let f=D[P],x=q;$(),f&&x&&x(f.value)}function y(f,x){et(f),f.style.display=x.length?"":"none";for(let H of x){let j=L("div",{class:"term-line"});for(let oe of H)oe.text&&j.append(as(oe,r));f.append(j)}}function S(f){let x=f.cssPrefix??f.id,H=L("span",{class:"cmd-tab","data-cmd":f.id,role:"tab",tabindex:"0","aria-selected":"false"},[f.icon?L("span",{class:"tab-ic","aria-hidden":"true"},[$a(f.icon,13)]):null,L("span",{class:"tab-label",text:f.label}),L("span",{class:"tab-x",role:"button",tabindex:"0","aria-label":`Close ${f.label} tab`,title:`Close ${f.label}`,text:"\xD7"})]),j=L("div",{class:`term-head ${x}-fixed`,"aria-hidden":"true"}),oe=L("div",{class:`term-foot-lines ${x}-close`,"aria-hidden":"true"}),ie=L("div",{class:`tm-menu te-menu ${x}-menu`,role:"listbox"}),Ae=L("div",{class:`te-warn ${x}-warn`,role:"alert"}),xe=f.multiline?L("div",{class:`term-gutter ${x}-gutter`,"aria-hidden":"true"}):null,ae={tab:f,prefix:x,chip:H,view:L("div",{class:`term-view ${x}-view`,"data-cmd":f.id}),head:j,foot:oe,gutter:xe,menu:ie,warn:Ae,editor:null,completion:null,ghost:"",dirty:!1,commitWarn:"",revision:f.revision?.()??0};ae.editor=new eo(a,{multiline:f.multiline===!0,get placeholder(){return f.placeholder??""},ariaLabel:f.ariaLabel??f.label,cssPrefix:x},{onInput:()=>{A&&(M(),b(!1)),O(ae,!1),ee(ae),I(ae),Z()&&X(ae)},onCaretMove:()=>{ee(ae),I(ae)},onFocus:()=>{t.onFocusChange?.(!0),I(ae)},onBlur:()=>{t.onFocusChange?.(!1),M(),b(!1),O(ae,!0),I(ae),a.setTimeout(()=>$(),120)},onKeyDown:Be=>he(ae,Be)});let Oe=xe?L("div",{class:`term-editrow ${x}-ml`},[xe,ae.editor.root]):ae.editor.root;return ae.view.append(j,L("div",{class:"term-edit"},[Oe]),ie,Ae,oe),ae}for(let f of t.tabs){let x=S(f);s.set(f.id,x),i.insertBefore(x.chip,o.barSpacer),l.append(x.view)}i.insertBefore(p,o.barSpacer);function O(f,x){let H=f.tab.commit(f.editor.value,f.editor.caret,x);f.dirty=H.dirty,f.commitWarn=H.warning??"",I(f)}function F(f,x){x?(f.warn.textContent="\u26A0 "+x,f.warn.classList.add("show")):(f.warn.classList.remove("show"),f.warn.textContent="")}function I(f){let x=f.editor.value,{segments:H,warning:j}=f.tab.highlight(x);if(F(f,j||f.commitWarn),f.editor.paint(H,f.editor.isFocused()?f.ghost:""),f.gutter){let oe=Math.max(1,x.split(`
`).length);f.gutter.textContent=Array.from({length:oe},()=>"...").join(`
`)}}function ee(f){if(f.completion=null,f.ghost="",v()||!f.editor.isFocused())return;let x=f.tab.complete(f.editor.value,f.editor.caret);x&&(f.completion=x,f.ghost=x.ghost??"")}function V(f){let x=f.completion;if(!x||!f.ghost||!x.ghostValue)return!1;f.ghost="";let H=x.apply(x.ghostValue);return f.editor.value=H.text,f.editor.setCaret(H.caret),O(f,!1),ee(f),I(f),f.editor.focus(),!0}function X(f){if(v())return $();let x=f.tab.complete(f.editor.value,f.editor.caret);if(f.completion=x,!x)return $();if(x.message)return le(f.menu,x.message);if(!x.items.length)return le(f.menu,"(no matching values)");W(f.menu,x.items,H=>{let j=x.apply(H);f.editor.value=j.text,f.editor.setCaret(j.caret),O(f,!1),ee(f),I(f),f.editor.focus()})}function he(f,x){if(Z()){if(x.key==="ArrowDown"){x.preventDefault(),pe(1);return}if(x.key==="ArrowUp"){x.preventDefault(),pe(-1);return}if(x.key==="Enter"||x.key==="Tab"){x.preventDefault(),w();return}if(x.key==="Escape"){x.preventDefault(),$();return}return}let H=f.editor.value,j=f.editor.caret,oe=j===H.length||H[j]===`
`;if(x.key==="ArrowDown"&&(!f.tab.multiline||H.slice(j).indexOf(`
`)<0)){x.preventDefault(),X(f);return}if(x.key==="Tab"){if(B(x)){b(!1);return}x.preventDefault(),f.ghost?V(f):X(f);return}if(f.ghost&&(x.key==="ArrowRight"||x.key==="End")&&oe){x.preventDefault(),V(f);return}if(x.key==="Escape"){x.preventDefault(),k(),f.ghost="",f.completion=null,I(f);return}x.key==="Enter"&&!f.tab.multiline&&(x.preventDefault(),O(f,!0))}for(let f of s.values())a.listen(f.menu,"mousedown",x=>{let H=x.target.closest(".tm-item");!H||H.classList.contains("tm-empty")||(x.preventDefault(),P=Number(H.dataset.i??"0"),w())});function ue(){for(let[f,x]of s){let H=g.has(f);x.chip.style.display=H?"":"none";let j=m===f;x.chip.classList.toggle("on",j),x.chip.setAttribute("aria-selected",j?"true":"false"),x.view.style.display=j?"":"none"}p.style.display=g.size<s.size?"":"none"}function ce(f){if(!g.has(f))return;m=f,ue(),$();let x=s.get(f);x&&(we(x),v()||x.editor.focus()),t.onTabChange?.(f)}function ke(f){if(g.delete(f),g.size===0){for(let x of s.keys())g.add(x);m=t.tabs[0].id,ue(),o.hide();return}m===f&&(m=[...g][0]),ue()}for(let[f,x]of s)a.listen(x.chip,"click",H=>{if(H.target.closest(".tab-x")){ke(f);return}ce(f)}),a.listen(x.chip,"keydown",H=>{let j=H.key;j!=="Enter"&&j!==" "||(H.preventDefault(),H.target.closest(".tab-x")?ke(f):ce(f))});a.listen(p,"click",()=>{let f=t.tabs.find(x=>!g.has(x.id));f&&(g.add(f.id),ce(f.id))});function me(){let f=v();n.classList.toggle("fallback",f);for(let x of s.values())x.editor.setMode(f?"plain":"rich")}function we(f){f.editor.setPrefix(f.tab.prefix()),y(f.head,f.tab.headerLines?.()??[]),y(f.foot,f.tab.footerLines?.()??[]);let x=f.tab.revision?.()??0;if(x!==f.revision){f.revision=x;let H=f.editor.isFocused(),j=H||f.dirty?f.tab.retain?.(f.editor.value)??"":"",oe=f.tab.multiline?`
`:" ";f.editor.value=[f.tab.text(),j].filter(Boolean).join(oe),H&&f.editor.setCaret(f.editor.value.length),f.dirty=j!==""}else!f.editor.isFocused()&&!f.dirty&&(f.editor.value=f.tab.text());I(f)}function Y(){me();for(let f of s.values())we(f);o.fitBar()}return o.applyTooltips(),ue(),Y(),{el:n,render:Y,toggle(f){o.toggle(f)},isShown:()=>o.isShown(),focusEditor(){s.get(m)?.editor.focus()},activeTab:()=>m,setActiveTab:f=>ce(f),destroy(){$(),a.flush(),o.destroy()}}}var wr=/^[A-Za-z0-9_./:@%+=-]+$/;function gs(e){return e===""?"''":wr.test(e)?e:`'${e.replace(/'/g,"'\\''")}'`}function nd(e){return e===""?"''":wr.test(e)?e:`'${e.replace(/'/g,"''")}'`}function id(e){return e===""?'""':wr.test(e)?e:`"${e.replace(/"/g,'""')}"`}var lt={bash:{id:"bash",label:"bash",prompt:"$",cont:"\\",quote:gs},zsh:{id:"zsh",label:"zsh",prompt:"%",cont:"\\",quote:gs},powershell:{id:"powershell",label:"PowerShell",prompt:"PS>",cont:"`",quote:nd},cmd:{id:"cmd",label:"cmd",prompt:"C:\\>",cont:"^",quote:id}};function As(e){return e==="windows"?"powershell":"bash"}function bs(e){let t=e??(typeof navigator<"u"?navigator:void 0);if(!t)return"unknown";let a=t.userAgentData?.platform;if(typeof a=="string"&&a){let n=a.toLowerCase();if(n.includes("win"))return"windows";if(n.includes("mac"))return"mac";if(n.includes("linux")||n.includes("chrome os")||n.includes("android"))return"linux"}let o=(t.platform??"").toLowerCase();if(o){if(o.includes("win"))return"windows";if(o.includes("mac"))return"mac";if(o.includes("linux")||o.includes("x11"))return"linux"}let r=(t.userAgent??"").toLowerCase();return r.includes("windows")?"windows":r.includes("mac os")||r.includes("macintosh")?"mac":r.includes("linux")||r.includes("x11")?"linux":"unknown"}function vs(e,t){let a=lt[t?.shell??"bash"],o=["freva-client databrowser data-search"];e.flavour!=="freva"&&o.push(`--flavour ${a.quote(e.flavour)}`);for(let[r,n]of Zt(e))o.push(`${r}=${a.quote(n)}`);return o.push(ld(e,a)),o.filter(Boolean).join(" ").trim()}function sd(e,t=lt.bash){let a=t.quote,o=[],r=e.time;r&&(r.from||r.to)&&(o.push(`time=${a(`${r.from||"1"} TO ${r.to||"9999"}`)}`),o.push(`time_select=${r.mode}`));let n=e.bbox;return n&&(o.push(`bbox=${n.minLon},${n.maxLon},${n.minLat},${n.maxLat}`),o.push(`bbox_select=${n.mode}`)),o.join(" ")}function ld(e,t=lt.bash){let a=t.quote,o=sd(e,t),r=Sa(e).map(([n,i])=>`${n}=${a(i)}`).join(" ");return[o,r].filter(Boolean).join(" ")}var ws=160,cd=200,xr=60,yr={bbox:"bbox=minLon,maxLon,minLat,maxLat - e.g. bbox=-10,10,35,60 \xB7 add bbox_select (defaults to flexible)",time:'time="2000 TO 2010" - e.g. time="2000-01 TO 2010-12" \xB7 add time_select (defaults to flexible)'};function xs(e){return e==="bbox_select"||e==="time_select"?[...Jt]:null}var dd=/_n(?:o(?:t_?)?)?$/;function ys(e){let t=Ft(e.dis),a=Ft(e.dis),o=new Map,r=-1,n=e.cfg.terminal.os??bs(),i=e.cfg.terminal.shell??As(n);function l(){let w=new Set(Xt(e.state));if(w.size)for(let y of We)w.add(y);return w}function d(){let w=l(),y=w.size?[...w]:[...ht,...Bt];return[...new Set([...y,...We])].filter(O=>!Qe(e.state,O))}function p(w){let y=w.toLowerCase(),S=d();return dd.test(y)?[...S.filter(F=>!We.has(F)).map(mt).filter(F=>F.startsWith(y)),...S.filter(F=>F.startsWith(y))]:S.filter(F=>F.startsWith(y))}function u(w){let y=e.state.facets.find(S=>S.key===fe(w));return y?y.values.map(S=>({value:S.value,count:S.count})):[]}function s(w){let y=fe(w),S=o.get(y);return S&&S.length?S:u(y).map(O=>O.value)}async function g(w){let y=fe(w);if(!Qe(e.state,y))try{let S=e.api.channelSignal("autocomplete"),O=await e.api.metadataSearch(e.state.flavour,e.state.uniqKey,on(e.state,y),S),F=xa(O.facets[y]??[]);F.length&&(o.set(y,F.map(([I])=>I)),pe?.render())}catch{}}function m(w,y){for(let S of vt(w))if(S.kind==="tok"&&y>=S.start&&y<=S.end)return S;return null}function A(w,y){let S=m(w,y);return S?{start:S.start,typed:w.slice(S.start,y).replace(/"/g,"")}:{start:y,typed:""}}function v(w,y){for(let S of vt(w))if(S.kind==="tok"&&S.start===y)return S.end;return y}function b(w,y,S){let O=v(w,y);if(!w.slice(y,O).includes("=")){let X=`${S}=`;return{text:w.slice(0,y)+X+w.slice(O),caret:y+X.length}}let I=w.indexOf("=",y),V=`${w.slice(y,I)}=${It(S)} `;return{text:w.slice(0,y)+V+w.slice(O),caret:y+V.length}}function k(){let w=lt[i],y=[{text:w.prompt,kind:"prompt"},{text:" "},{text:"freva-client databrowser data-search",kind:"fixed"}];e.state.flavour!=="freva"&&(y.push({text:" "},{text:"--flavour",kind:"fixed"}),y.push({text:" "},{text:e.state.flavour,kind:"accent"}));for(let[S,O]of Zt(e.state))y.push({text:" "},{text:S,kind:"key"},{text:"=",kind:"eq"},{text:w.quote(O),kind:"value"});return y}function M(w){let y=l(),S=y.size>0,O=[],F="";for(let I of vt(w)){if(I.kind==="ws"){O.push({text:I.raw});continue}let ee=I.value.indexOf("=");if(ee<0){let Y=!S||y.has(fe(I.value.toLowerCase()))||[...y].some(f=>f.startsWith(fe(I.value.toLowerCase())));O.push({text:I.raw,kind:Y?"key":"bad"}),!Y&&!F&&(F=`\u201C${I.value}\u201D is not a facet`);continue}let V=I.value.slice(0,ee),X=I.value.slice(ee+1),he=I.raw.indexOf("="),ue=he<0?I.raw:I.raw.slice(0,he),ce=he<0?"":I.raw.slice(he+1),{baseKey:ke}=Pe(V.toLowerCase()),me=!S||y.has(ke);O.push({text:ue,kind:me?"key":"bad"}),he>=0&&O.push({text:"=",kind:"eq"}),!me&&!F&&(F=`\u201C${V}\u201D is not a facet`);let we=!1;if(me&&X){let Y=yo(e.state,V.toLowerCase());Y&&!Y.has(X)&&(we=!0,F||(F=`\u201C${X}\u201D isn\u2019t a ${V} value`))}O.push({text:ce,kind:we?"bad":"value"})}return{segments:O,warning:F}}function B(w,y){let{start:S,typed:O}=A(w,y),F=y===w.length,I=Y=>b(w,S,Y),ee=O.indexOf("=");if(ee<0){let Y=O.toLowerCase(),f=p(Y),x=f.filter(H=>H.length>Y.length).sort((H,j)=>H.length-j.length)[0];return{items:f.slice(0,xr).map(H=>({value:H,count:null})),ghost:F&&O&&x?x.slice(O.length):"",ghostValue:x,apply:I}}let V=O.slice(0,ee).toLowerCase(),X=O.slice(ee+1),he=X.toLowerCase(),ue=yr[V];if(ue)return{items:[],message:ue,apply:I};let ce=xs(V);if(ce){let Y=ce.filter(x=>x.startsWith(he)),f=Y.filter(x=>x.length>X.length)[0];return{items:Y.map(x=>({value:x,count:null})),ghost:F&&f?f.slice(X.length):"",ghostValue:f,apply:I}}t(()=>void g(V),ws);let ke=new Map(u(V).map(Y=>[Y.value,Y.count??null])),me=s(V).filter(Y=>Y.toLowerCase().startsWith(he)),we=me.filter(Y=>Y.length>X.length).sort((Y,f)=>Y.length-f.length)[0];return{items:me.slice(0,xr).map(Y=>({value:Y,count:ke.get(Y)??null})),ghost:F&&we?we.slice(X.length):"",ghostValue:we,apply:I}}function D(w,y,S){if(S)return w;let O=m(w,y);return O?w.slice(0,O.start)+w.slice(O.end):w}function P(w){let y={};for(let S of Object.keys(w))We.has(S.toLowerCase())||(y[S]=w[S]);return y}function U(w){let{rejected:y}=_t(e.state,P(wt(w))),S=[...y];for(let F of vt(w)){if(F.kind!=="tok")continue;let I=w.slice(F.start,F.end),ee=I.indexOf("=");if(ee<0)continue;let V=I.slice(0,ee).toLowerCase();We.has(V)&&I.slice(ee+1).trim()===""&&!S.includes(I)&&S.push(I)}let O=/\S+$/.exec(w)?.[0]??"";if(O&&!S.includes(O)){let F=P(wt(O)),I=Object.keys(wt(O)).every(V=>We.has(V.toLowerCase()));!(Object.keys(F).length&&!_t(e.state,F).rejected.length&&Object.values(F).every(V=>V.every(Boolean)))&&!I&&S.push(O)}return[...new Set(S)].join(" ")}function q(w){let y=O=>Qe(e.state,O),S=[];for(let O of e.state.facets){if(S.length>=(w==="cli"?2:1))break;let F=O.key.toLowerCase();if(y(F)||We.has(F))continue;let I=O.values[0]?.value;I&&S.push([F,I])}if(!S.length){let O=ht.find(F=>!y(F));return O?w==="cli"?`${O}=\u2026`:`${O}="\u2026"`:""}return S.map(([O,F])=>w==="cli"?`${O}=${lt[i].quote(F)}`:`${O}=${JSON.stringify(F)}`).join(" ")}let Z={id:"cli",cssPrefix:"te",label:lt[i].label,icon:E.bashTab,multiline:!1,get placeholder(){return q("cli")},ariaLabel:"Command facets",prefix:k,text:()=>dn(e.state),highlight:M,complete:B,commit(w,y,S){let O=D(w,y,S),F=e.applyTerminalDraft(O),I=e.terminalDraftErrors(O),V=(M(w).warning??"")||I[0]||"",X=O!==w,he=vt(O).some(ue=>{if(ue.kind!=="tok")return!1;let ce=ue.value.indexOf("=");return ce<1||ue.value.slice(ce+1)===""});return{dirty:F>0||X||V!==""||he,warning:I[0]}},copyText:()=>vs(e.state,{shell:i}),revision:()=>e.state.externalEdits,retain:U};function K(w){let y=0,S="",O=0;for(let F=0;F<w.length;F++){let I=w[F];if(S){if(I==="\\"){F++;continue}I===S&&(S="");continue}if(I==='"'||I==="'"){S=I;continue}I==="["||I==="{"||I==="("?y++:I==="]"||I==="}"||I===")"?y=Math.max(0,y-1):I===","&&y===0&&(O=F+1)}return w.slice(O)}function ne(w,y){let S=w[y];if(S!==void 0&&S!==`
`&&S!==",")return null;let O=w.lastIndexOf(`
`,y-1)+1,F=K(w.slice(O,y)),I=F.indexOf("=");if(I<0){let X=F.match(/([\w-]*)$/);return{word:X?X[1]:"",isValue:!1,key:""}}let ee=F.slice(0,I).trim().replace(/["']/g,""),V=F.slice(I+1).match(/([^\s,"'[\]]*)$/);return{word:V?V[1]:"",isValue:!0,key:ee}}function W(w,y,S,O,F){let I=y-S,ee;return O?((w[I-1]==='"'||w[I-1]==="'")&&I--,ee=`${JSON.stringify(F)},`):ee=`${F}=`,{text:w.slice(0,I)+ee+w.slice(y),caret:I+ee.length}}function le(w,y,{cont:S=!1,code:O="fixed"}={}){return[{text:w,kind:S?"contprompt":"prompt"},{text:y,kind:O}]}let $={id:"py",label:"python",icon:E.pySnake,multiline:!0,get placeholder(){return q("py")},ariaLabel:"databrowser keyword arguments",prefix:()=>[],headerLines:()=>{let w=[le(">>> ","from freva_client import databrowser"),le(">>> ","databrowser(")];for(let y of Eo(e.state))w.push(le("... ",`    ${y.code},`,{cont:!0,code:y.scope?"fixed":"muted"}));return w},footerLines:()=>[[{text:"... ",kind:"contprompt"},{text:")",kind:"fixed"}]],text:()=>Oo(e.state),highlight:w=>({segments:[{text:w}]}),complete(w,y){let S=ne(w,y);if(!S)return null;let O=S.word.toLowerCase(),F=X=>W(w,y,S.word.length,S.isValue,X);if(S.isValue&&yr[S.key])return{items:[],message:yr[S.key],apply:F};let I=S.isValue?xs(S.key):null,ee=I?I.filter(X=>X.startsWith(O)):S.isValue?s(S.key).filter(X=>X.toLowerCase().startsWith(O)):p(O);S.isValue&&S.key&&!I&&t(()=>void g(S.key),ws);let V=ee.filter(X=>X.toLowerCase().startsWith(O)&&X.length>S.word.length).sort((X,he)=>X.length-he.length)[0];return{items:ee.slice(0,xr).map(X=>({value:X,count:null})),ghost:S.word&&V?V.slice(S.word.length):"",ghostValue:V,apply:F}},commit(w,y,S){let O=()=>{e.applyTerminalDraft(rn(w))};return S?O():a(O,cd),{dirty:!1}},copyText:()=>nn(e.state),revision:()=>e.state.externalEdits},pe=vr(e.roots.overlay,{tabs:[Z,$],activeTab:e.state.terminalTab,os:n==="unknown"?"mac":n,bounds:()=>e.roots.overlay,tooltipAttribute:"data-tip",storage:{getTheme:()=>Vn(),setTheme:w=>jn(w),getAlpha:()=>Kn(),setAlpha:w=>Zn(w)},fallback:()=>e.roots.app.dataset.terminalFallback==="true"||typeof window.matchMedia=="function"&&window.matchMedia("(max-width: 720px)").matches,menuItems:[{label:"How to install freva-client",onSelect:()=>e.openHelp()},{label:"Documentation \u2197",href:"https://freva-org.github.io/freva-nextgen/"}],onTabChange:w=>{e.state.terminalTab=w==="py"?"py":"cli"},onFocusChange:w=>{e.state.terminalFocused=w},onClose:()=>{e.state.terminalFocused=!1,e.roots.app.querySelector('[aria-label="Command terminal"]')?.focus()},onCopyFailed:w=>e.toast("error",w)});return e.dis.add(()=>pe.destroy()),{render(){e.state.facetsVersion!==r&&(r=e.state.facetsVersion,o.clear()),pe.render()},toggle(){pe.toggle()},isShown(){return pe.isShown()},destroy(){pe.destroy()}}}var pd=[{title:"load/{flavour} is a GET",body:"The data-load endpoint is a GET that returns 201 and streams zarr URLs; it is auth.required. It is never issued as a POST."},{title:"Time is unbracketed",body:"Time queries are sent as time=<from> TO <to> with a separate time_select=<mode>. The prototype\u2019s [ \u2026 TO \u2026 ] bracket form is not used anywhere."},{title:"Catalogue export guard",body:'Intake/STAC export requests max-results=100000; the server answers 413 with the exact detail "Result stream too big." Export is disabled client-side past that ceiling.'},{title:"Browsing fetches no per-file metadata",body:"Result rows are the thin {file|uri, fs_type} only. Full per-file facets are fetched lazily - one ?file= call per inspected file - and rendered in the Details panel."},{title:"V4 - strict/file bbox+time deferred",body:"Only flexible (Intersects) ships enabled. strict/file modes for both the time and bbox editors are gated behind config.enableStrictBBoxModes until verified against the backend."},{title:"V10 - per-file extent deferred",body:"Per-file bbox renders whenever the ?file= response carries a bbox (the backend must include bbox in the file field list). Time range is derived from the filename, else shown as not available. Coordinates are never fabricated."},{title:"Embedding - transformed ancestors break fixed overlays",body:"Popovers/menus/tooltips position as fixed relative to the viewport. If any ANCESTOR of the mount establishes a containing block for fixed elements, they anchor to that ancestor instead and appear offset. Triggers: transform, filter, backdrop-filter, perspective, contain: paint/layout/strict, or will-change of any of those. Mount outside such wrappers, or drop the property on the ancestor."},{title:"Leaflet is a page-global install (survives destroy)",body:"The Leaflet stylesheet (and window.L) are installed once per PAGE, tied to no component - because tying the stylesheet to a component lifecycle caused maps in other components to lose their layout when that component re-rendered. Consequently they persist after destroy(): the stylesheet stays in <head> and window.L stays defined. destroy() fully tears down THIS widget (DOM, observers, in-flight requests); it does not, by design, uninstall this shared page-global. The script tag itself is removed once it registers window.L."}];function Ss(e){let t=c("div",{class:"notes-list"});for(let n of pd)t.append(c("div",{class:"nl"},[c("div",{class:"h",text:n.title}),c("p",{text:n.body})]));let a=c("button",{class:"x",type:"button","aria-label":"Close developer notes"},[R(E.x,{size:16})]),o=c("div",{class:"notes-drawer",role:"complementary","aria-label":"Developer notes"},[c("h4",{},[R(E.notes,{size:16}),c("span",{text:"Developer notes"}),a]),t]);e.roots.app.append(o),e.dis.add(()=>o.remove());let r=n=>{o.classList.toggle("show",n)};return e.dis.listen(a,"click",()=>r(!1)),re(t,...Array.from(t.childNodes)),{toggle(){r(!o.classList.contains("show"))},isShown(){return o.classList.contains("show")}}}function Cs(e,t,a={}){let o=t.trim().toLowerCase();if(!o)return[];let r=[],n=[];for(let l of e){let d=a.label?a.label(l):l.label;for(let p of l.values){if(a.isApplied?.(l.key,p.value))continue;let u=p.value.toLowerCase(),s=a.describe?.(l.key,p.value)??null,g=u.includes(o),m=s?s.toLowerCase().includes(o):!1;if(!g&&!m)continue;let A={key:l.key,label:d,value:p.value,count:p.count,desc:s};u.startsWith(o)?r.push(A):n.push(A)}}let i=(l,d)=>d.count-l.count;return r.sort(i),n.sort(i),[...r,...n].slice(0,a.limit??40)}function ud(e,t){return Cs(e.state.facets,t,{label:a=>Yt(e.state,a.key),describe:(a,o)=>At(e.state,a,o),isApplied:(a,o)=>gt(e.state,a,o)})}function Ds(e,t){let a=e.dis,o=c("div",{class:"vsearch-pop",role:"listbox","aria-label":"Facet value matches"});o.style.position="absolute",e.roots.app.append(o);let r=[],n=0,i=!1,l=null,d=()=>{i=!1,o.classList.remove("show"),re(o),l?.flush(),l=null},p=()=>{let m=t.getBoundingClientRect().width;Ht(e.roots.app,o,t,{placement:"below",gap:5,minWidth:Math.max(m,280),maxWidth:Math.max(m,420),maxHeight:340})},u=()=>{o.querySelectorAll(".vs-item").forEach((m,A)=>{let v=A===n;m.classList.toggle("hl",v),m.setAttribute("aria-selected",v?"true":"false"),v&&m.scrollIntoView({block:"nearest"})})},s=m=>{let A=r[m];A&&(t.value="",d(),e.toggleFacet(A.key,A.value))},g=()=>{r=ud(e,t.value),l?.flush(),l=a.child();let m=l;if(r.length===0){t.value.trim()?(re(o,c("div",{class:"vs-empty",text:"No matching facet values."})),o.classList.add("show"),i=!0,p()):d();return}n=0;let A=r.map((v,b)=>c("div",{class:`vs-item${b===0?" hl":""}`,role:"option","aria-selected":b===0?"true":"false",title:v.desc?`${v.value} - ${v.desc}`:`${v.label}: ${v.value}`},[c("span",{class:"vs-badge",text:v.label}),c("span",{class:"vs-val",text:v.value}),v.desc?c("span",{class:"vs-desc",text:v.desc}):null,c("span",{class:"vs-cnt",text:v.count.toLocaleString("en-US")})]));A.forEach((v,b)=>{m.listen(v,"mousedown",k=>{k.preventDefault(),s(b)})}),re(o,...A),o.classList.add("show"),i=!0,p()};return a.listen(t,"input",()=>g()),a.listen(t,"focus",()=>{t.value.trim()&&g()}),a.listen(t,"blur",()=>a.setTimeout(()=>d(),120)),a.listen(t,"keydown",m=>{let A=m;if(!i){A.key==="ArrowDown"&&t.value.trim()&&(A.preventDefault(),g());return}A.key==="ArrowDown"?(A.preventDefault(),n=Math.min(r.length-1,n+1),u()):A.key==="ArrowUp"?(A.preventDefault(),n=Math.max(0,n-1),u()):A.key==="Enter"?(A.preventDefault(),s(n)):A.key==="Escape"&&(A.preventDefault(),d())}),a.listen(window,"resize",()=>i&&p()),a.listen(window,"scroll",m=>{if(!i)return;let A=m.target;A&&typeof A.nodeType=="number"&&o.contains(A)||d()},!0),a.listen(t,"blur",()=>{i&&!Nt(e.roots.app,t)&&d()}),a.add(()=>o.remove()),{destroy:d}}var ks=62,fd=28,Ms=1900,hd=6,md="Search values \u2013 e.g. ",Sr=["a variable","a model","an experiment"],Ts=30;function Es(e){let t=[],a=[];for(let r of e.state.facets)for(let{value:n}of r.values.slice(0,4)){if(typeof n!="string"||n.length===0)continue;let i=At(e.state,r.key,n),l=i!==null&&i.length>0&&i.length<=Ts,d=l?i:n;if(!(d.length>Ts)){(l?t:a).push(d);break}}let o=[];for(let r of[...t,...a])if(o.includes(r)||o.push(r),o.length>=hd)break;return o}function gd(e,t,a){let o=typeof matchMedia=="function"&&matchMedia("(prefers-reduced-motion: reduce)").matches,r=typeof IntersectionObserver=="function",n=Sr,i=0,l=0,d=!1,p=0,u=!1,s=!1,g=()=>t.value.length===0&&t.ownerDocument.activeElement!==t,m=k=>{t.placeholder=`${md}${k}`},A=()=>{if(u||!s||!t.isConnected)return;if(!d&&l===0){let M=Es(e);M.length&&(n=M),i>=n.length&&(i=0)}let k=n[i]??Sr[0];if(!g()){p=a.setTimeout(A,Ms);return}if(d){if(l-=1,m(k.slice(0,Math.max(0,l))),l<=0){d=!1,l=0,i=(i+1)%n.length,p=a.setTimeout(A,ks*4);return}p=a.setTimeout(A,fd);return}if(l+=1,m(k.slice(0,l)),l>=k.length){d=!0,p=a.setTimeout(A,Ms);return}p=a.setTimeout(A,ks)},v=()=>{let k=Es(e);m((k.length?k:Sr)[0]??"")};if(o||!r){m(""),p=a.setTimeout(v,600),a.add(()=>window.clearTimeout(p));return}m("");let b=new IntersectionObserver(k=>{let M=k.some(B=>B.isIntersecting);M!==s&&(s=M,window.clearTimeout(p),M&&(p=a.setTimeout(A,400)))},{threshold:0});b.observe(t),a.add(()=>{u=!0,b.disconnect(),p&&window.clearTimeout(p)})}function Ad(){if(typeof navigator>"u")return!1;let t=navigator.userAgentData?.platform??navigator.platform??"";return/mac|iphone|ipad|ipod/i.test(t)}function bd(e,t){let a=Ad(),o=c("span",{class:"search-kbd","aria-hidden":"true"},[c("kbd",{text:a?"\u2318":"Ctrl"}),c("kbd",{text:"K"})]),r=e.getAttribute("aria-label")??"";return e.setAttribute("aria-keyshortcuts",a?"Meta+K":"Control+K"),e.setAttribute("aria-label",`${r} (${a?"Command":"Control"}+K)`.trim()),t.listen(e.ownerDocument,"keydown",n=>{let i=n;if(i.key!=="k"&&i.key!=="K"||(a?!i.metaKey:!i.ctrlKey)||i.altKey||!e.isConnected)return;let l=e.ownerDocument.activeElement;l!==null&&l!==e&&(l.tagName==="TEXTAREA"||l.tagName==="INPUT"&&l.type!=="checkbox"||l.isContentEditable)||(i.preventDefault(),e.focus(),e.select())}),o}function Os(e,t,a){gd(e,t,e.dis),a.appendChild(bd(t,e.dis))}var vd=4200;function Is(e){let t=e.dis,{statusDot:a,statusMsg:o,toastHost:r}=e.roots;function n(l,d){a.className=`status-dot ${l}`,o.className=`mono status-msg ${l}`,o.textContent=d}function i(l,d){n(l,d);let p=c("div",{class:`toast ${l}`,role:"status"},[R(E.info,{size:15}),c("span",{class:"toast-msg",text:d})]);r.append(p),t.setTimeout(()=>p.classList.add("in"),10);let u=()=>{},s=()=>{u(),p.classList.remove("in"),t.setTimeout(()=>p.remove(),220)};t.setTimeout(s,vd),u=t.listen(p,"click",s)}return{log:n,toast:i}}var wd=80,xd=8;function Ls(e,t){let a=c("div",{class:"fdb-tip",role:"tooltip"});e.appendChild(a);let o=null,r=0,n=p=>{let u=p.getAttribute("data-tip")??"";if(!u||!p.isConnected)return;if(!Nt(e,p)){i();return}a.textContent=u,a.classList.add("show"),l?.observe(e,{childList:!0,subtree:!0});let s=a.getBoundingClientRect().top;Ht(e,a,p,{placement:"below",gap:xd,margin:6}),a.classList.toggle("above",a.getBoundingClientRect().top<s)},i=()=>{window.clearTimeout(r),o=null,a.classList.remove("show"),l?.disconnect()},l=typeof MutationObserver<"u"?new MutationObserver(()=>{o&&!o.isConnected&&i()}):null,d=p=>{let s=p.target?.closest?.("[data-tip]");return s&&e.contains(s)&&s.getAttribute("data-tip")?s:null};t.listen(e,"pointerover",p=>{let u=d(p);!u||u===o||(o=u,window.clearTimeout(r),r=window.setTimeout(()=>{o===u&&u.isConnected&&n(u)},wd))}),t.listen(e,"pointerout",p=>{let u=d(p);if(!u||u!==o)return;let s=p.relatedTarget;s&&u.contains(s)||i()}),t.listen(e,"focusin",p=>{let u=d(p);u&&(o=u,n(u))}),t.listen(e,"focusout",i),t.listen(e,"pointerdown",i,!0),t.listen(window,"scroll",i,!0),t.listen(window,"resize",()=>{o&&n(o)}),t.add(()=>{window.clearTimeout(r),l?.disconnect(),a.remove()})}var Bs=1,ma="siv",ao="q",yd=/^[a-z][a-z0-9-]{0,31}$/,Sd=/([+\-!(){}[\]^"~?:\\/&|])/g;function oo(e){let t=e.trim().replace(/\s+/g," ");return t===""?"":`*${t.replace(Sd,"\\$1").replace(/ /g,"*")}*`}function Ps(e,t="file"){let a=new URLSearchParams;a.set(ma,String(e.v??1)),e.flavour&&a.set("flavour",e.flavour);for(let[r,n]of Object.entries(e.facets??{}).sort(([i],[l])=>i.localeCompare(l)))for(let i of n)a.append(r,i);let o=e.q?oo(e.q):"";return o&&a.append(t,o),a.toString()}function ro(e,t="file"){let a=new URLSearchParams(e),o=a.get(ma),r=o===null?void 0:Number(o);a.delete(ma);let n=a.get(ao)??void 0;a.delete(ao);let i=a.get("flavour"),l=i&&yd.test(i)?i:void 0,d={};for(let[s,g]of a.entries())s!=="flavour"&&(d[s]??(d[s]=[])).push(g);if(n&&n.trim()!==""){let s=oo(n);s&&(a.append(t,s),(d[t]??(d[t]=[])).push(s))}let p={v:1};n&&n.trim()!==""&&(p.q=n.trim()),l&&(p.flavour=l),Object.keys(d).length>0&&(p.facets=d);let u={intent:p,normalizedSearch:a.toString()};return r!==void 0&&r!==1&&(u.unsupportedVersion=r),u}function no(e,t){return t.flavour?{...e,flavour:t.flavour}:{...e}}function Cd(e,t={},a={}){let o=a.search??(typeof window<"u"?window.location.search:""),r=ro(o,a.uniqKey??"file");if(a.normalizeUrl!==!1&&typeof window<"u"&&window.history&&r.normalizedSearch!==o.replace(/^\?/,"")){let n=new URL(window.location.href);n.search=r.normalizedSearch?`?${r.normalizedSearch}`:"",window.history.replaceState(null,"",n.toString())}return Cr(e,no(t,r.intent))}var Md="/api/freva-nextgen/databrowser",Td=250,Ed=300;function Dd(e){return{map:{...gn,...e.map??{}},...e.inspectorUrl?{inspectorUrl:e.inspectorUrl}:{},...e.overlayRoot?{overlayRoot:e.overlayRoot}:{},apiBase:e.apiBase??Md,flavour:e.flavour??"freva",devNotes:e.devNotes??!1,authEnabled:e.authEnabled??!1,enableHeavyOps:e.enableHeavyOps??!1,syncUrl:e.syncUrl??!0,baseFilters:e.baseFilters,enableStrictBBoxModes:e.enableStrictBBoxModes??!1,metadata:e.metadata??{},metadataScriptUrl:e.metadataScriptUrl??null,defaultLayout:e.defaultLayout==="overview"?"overview":"results",overview:{order:e.overview?.order?[...e.overview.order]:[],mainFacets:e.overview?.mainFacets?[...e.overview.mainFacets]:null},scopeRemovable:e.scopeRemovable??!1,features:{themeToggle:e.features?.themeToggle??!0,terminal:e.features?.terminal??!0,overview:e.features?.overview??!0,export:e.features?.export??!0,details:e.features?.details??!0,search:e.features?.search??!0,lensSwitcher:e.features?.lensSwitcher??!0,inspect:e.features?.inspect??!0,brand:e.features?.brand??!0,footer:e.features?.footer??!0},theme:e.theme??{},brand:{title:e.brand?.title??"Freva",mark:e.brand?.mark??"\u2248",description:e.brand?.description??"",showMark:e.brand?.showMark??!0,showTitle:e.brand?.showTitle??!0},terminal:{host:e.terminal?.host??null,shell:e.terminal?.shell??null,os:e.terminal?.os??null},getAuthToken:e.getAuthToken??(()=>null),getCsrfToken:e.getCsrfToken??(()=>null)}}function Od(e){let t=c("span",{class:"knob"},[R(E.moon,{size:15})]),a=c("button",{class:"theme",type:"button","aria-label":"Toggle theme",title:"Toggle day / night"},[t]),o=c("span",{class:"v",text:e.flavour}),r=c("button",{class:"lens",type:"button","aria-haspopup":"dialog","aria-expanded":"false","aria-label":"Naming flavour",title:"Change the naming flavour (metadata lens)"},[c("span",{class:"k",text:"Flavour"}),o,R(E.caret,{size:14})]),n=c("input",{class:"input",type:"text",placeholder:"Search","aria-label":"Search facet values",autocomplete:"off",spellcheck:!1}),i=c("span",{class:"search-spin","aria-hidden":"true"},[c("span",{class:"spin"})]),l=c("span",{class:"sr-only",role:"status","aria-live":"polite"}),d=c("span",{class:"search-aside"},[i]),p=c("div",{class:"search"},[c("span",{class:"ic"},[R(E.search,{size:16})]),n,d,l]),u=e.devNotes?c("button",{class:"icon-btn",type:"button","aria-label":"Developer notes",title:"Developer notes"},[R(E.notes,{size:16})]):null,s=e.features,g=c("button",{class:"iconbtn",type:"button","aria-label":"Command terminal",title:"Terminal"},[R(E.terminal,{size:20})]),m=c("button",{class:"iconbtn",type:"button","aria-label":"Inspect data",title:"Inspect a dataset by URL (metadata & 3D viewer)"},[R(E.inspect,{size:18})]),A=window.matchMedia?.("(prefers-reduced-motion: reduce)")?.matches===!0,v=e.brand.showMark?e.brand.mark==="\u2248"?A?c("span",{class:"mark brand-mark-static",text:"\u2248","aria-hidden":"true"}):c("img",{class:"brand-logo",src:mn,alt:"","aria-hidden":"true",decoding:"async"}):c("span",{class:"mark",text:e.brand.mark}):null,b=e.brand.showMark||e.brand.showTitle?c("div",{class:"brand"},[v,e.brand.showTitle?c("span",{text:e.brand.title}):null]):null,k=[s.brand?b:null,s.lensSwitcher?r:null,s.search?p:null,u,s.inspect?m:null,s.terminal?g:null,s.themeToggle?a:null],M=c("header",{class:"top"},k),B=c("button",{class:"help-x",type:"button","aria-label":"Close help"},[R(E.close,{size:15})]),D=c("div",{class:"help-pop",role:"dialog","aria-modal":"false","aria-label":"Help and setup"},[c("div",{class:"help-head"},[R(E.terminal,{size:16}),c("span",{class:"t",text:"Run this search yourself"}),B]),c("p",{text:"The terminal shows the exact CLI command and python call for whatever you have selected. To run them, install the client library:"}),c("pre",{class:"help-code",text:"python3 -m pip install freva-client"}),c("p",{text:"That gives you both the freva-client command line tool and the freva_client python library."}),c("div",{class:"help-h2",text:"Pointing it at this instance"}),c("p",{text:"On a centrally administered freva instance there is nothing to do. If you installed the client yourself, set the host once in your config file:"}),c("pre",{class:"help-code",text:"~/.config/freva/freva.toml"}),c("p",{class:"help-dim",text:"FREVA_CONFIG can also point at a freva.toml elsewhere. That is why the commands here carry no --host flag."}),c("a",{class:"help-link",href:"https://freva-org.github.io/freva-nextgen/",target:"_blank",rel:"noopener noreferrer",text:"freva-client documentation \u2197"})]),P=c("div",{class:"side-scroll"}),U=c("button",{class:"side-collapse",type:"button","aria-label":"Collapse filters",title:"Collapse filter sidebar"},[c("span",{class:"chev"},[R(E.chevron,{size:14})])]),q=c("aside",{class:"side"},[c("div",{class:"side-head"},[U]),P,c("div",{class:"side-flavour-veil","aria-hidden":"true"},[c("span",{class:"spin"})])]),Z=c("div",{class:"chips"}),K=c("button",{class:"clear-btn",type:"button",text:"Clear all"}),ne=c("button",{class:"ctrl on",type:"button","aria-label":"Browse results",title:"Browse the matching results"},[R(E.resultsFocus,{size:15}),c("span",{class:"ctrl-lbl",text:"Browse"})]),W=c("button",{class:"ctrl",type:"button","aria-label":"Overview",title:"Overview of the whole result set"},[R(E.overview,{size:15}),c("span",{class:"ctrl-lbl",text:"Overview"})]),le=c("button",{class:"iconbtn tbtn",type:"button","aria-label":"Details panel",title:"Details panel"},[R(E.info,{size:15}),c("span",{class:"tbtn-lbl",text:"Details"})]),$=c("div",{class:"ctrl-cluster"},[ne,s.overview?W:null]),pe=c("div",{class:"toprow"},[Z,K,$]),w=c("div",{class:"facet-grid"}),y=c("div",{class:"overview-mode"},[c("div",{class:"overview-cap",text:"Metadata overview - every facet for the current query at a glance."}),w]),S=c("span",{class:"res-count",text:"-"}),O=c("span",{class:"res-spin","aria-hidden":"true"},[c("span",{class:"spin"})]),F=c("button",{type:"button",class:"on","aria-label":"List view",title:"List view"},[R(E.list,{size:15})]),I=c("button",{type:"button","aria-label":"Grid view",title:"Grid view"},[R(E.grid,{size:15})]),ee=c("div",{class:"seg"},[F,I]),V=c("button",{class:"iconbtn tbtn",type:"button","aria-label":"Export catalogue",title:"Export catalogue","aria-haspopup":"menu","aria-expanded":"false"},[R(E.download,{size:15}),c("span",{class:"tbtn-lbl",text:"Export"})]),X=c("button",{class:"iconbtn tbtn ov-shelve",type:"button",hidden:"true","aria-label":"Minimize all blocks to full-width rows",title:"Minimize every block to a full-width row - then expand them one at a time"},[R(E.shelve,{size:15}),c("span",{class:"tbtn-lbl",text:"Stack"})]),he=c("span",{class:"cb","aria-hidden":"true"}),ue=c("button",{class:"selall",type:"button","aria-label":"Select all listed files",title:"Select all currently listed files"},[he,c("span",{class:"ctrl-lbl",text:"Select all"})]),ce=c("span",{class:"panelctl in"},[ue,c("span",{class:"bar-div"}),c("span",{class:"view-lbl",text:"View"}),ee,s.details?c("span",{class:"bar-div"}):null,s.details?le:null,s.export?c("span",{class:"bar-div"}):null]),ke=[c("span",{class:"scope-lbl",text:"Whole result set"}),S,O,e.brand.description?c("span",{class:"scope-desc",text:e.brand.description}):null,c("span",{class:"spacer"}),ce,X,s.export?V:null],me=c("div",{class:"res-bar"},ke),we=c("div",{class:"list-head",hidden:"true"},[c("span",{class:"lh-uri",text:"uri"}),c("span",{class:"lh-fs",text:"fs type"})]),Y=c("div",{class:"rows",id:"fdb-results"}),f=c("div",{class:"more-wrap"}),x=c("div",{class:"center-fixed"},[pe,me]),H=c("div",{class:"results-scroll"},[y,we,Y,f]),j=c("div",{class:"pickbar"}),oe=c("main",{class:"center"},[x,H,j]),ie=c("div",{class:"info-scroll"}),Ae=c("button",{class:"x",type:"button","aria-label":"Close details",title:"Close details"},[R(E.close,{size:16})]),xe=c("aside",{class:"details-panel collapsed"},[c("div",{class:"info-head"},[R(E.info,{size:16}),c("span",{class:"t",text:"Details"}),Ae]),ie]);Ae.dataset.role="details-close";let ae=c("div",{class:"body"},[q,oe,xe]),Oe=c("span",{class:"status-dot info"}),Be=c("span",{class:"mono"}),ct=s.footer?c("footer",{class:"status","aria-live":"polite","aria-label":"Status"},[Oe,Be]):null,tt=s.footer?null:c("div",{class:"sr-status","aria-live":"polite","aria-label":"Status",role:"status"},[Oe,Be]),Ue=c("div",{class:"toast-host","aria-live":"polite"}),be=c("div",{class:`fdb-app${s.footer?"":" no-footer"}`},[M,ae,ct]),Ye=c("div",{class:"freva-db","data-theme":"night"},[be,tt,Ue,D]);typeof window.matchMedia=="function"&&window.matchMedia("(prefers-reduced-motion: reduce)").matches&&be.setAttribute("data-reduced-motion","true");let dt={app:Ye,overlay:e.overlayRoot??Ye,facetList:P,chips:Z,clearAllBtn:K,overviewWrap:y,overviewGrid:w,resCount:S,resSpin:O,exportBtn:V,shelveBtn:X,selectAllBtn:ue,listHead:we,results:Y,moreWrap:f,pickbar:j,info:xe,infoScroll:ie,infoBtn:le,statusMsg:Be,statusDot:Oe,toastHost:Ue,lensValue:o};return dt.infoClose=Ae,{outer:Ye,shell:be,roots:dt,lensBtn:r,themeBtn:a,themeKnob:t,notesBtn:u,helpPanel:D,sideCollapse:U,searchInput:n,searchRegion:p,searchAside:d,searchSpin:i,searchStatus:l,resultsCtrl:ne,overviewCtrl:W,shelveBtn:X,listSeg:F,gridSeg:I,selectAllBtn:ue,cmdBtn:g,inspectBtn:m,resBar:me,panelCtl:ce,resultsScroll:H}}function Cr(e,t={}){var a;let o=Dd(t),r=Od(o),{roots:n,shell:i}=r,l=c("style",{type:"text/css"});l.textContent=En,n.app.insertBefore(l,n.app.firstChild),n.app.insertBefore(hn(),n.app.firstChild);let d=new Ta,p=new Ma(o,d),u=new Oa(n.app,d),s=Gr(o);s.flavour=o.flavour,s.theme=o.theme.mode??Bn(),s.layout=Rn(o.defaultLayout),s.view=zn(),s.flavours=[...Lt];let g=typeof matchMedia=="function"&&matchMedia("(max-width: 560px)").matches;s.sidebarCollapsed=Hn(g),o.overview.order.length&&(s.overviewOrder=[...o.overview.order]);let m=Qn();m&&(s.overviewSort=m.sort,s.overviewCollapsed=new Set(m.collapsed),s.overviewSpan=m.span,s.overviewH=m.h??{},s.overviewOrder=m.order,s.overviewAddOpen=m.addOpen,s.overviewStacked=m.stacked??!1,s.overviewStackSeen=m.stackSeen??[],s.overviewSnapshot=m.snapshot??null),s.metadata=Mn(o);let A=tn(o.baseFilters);if(o.scopeRemovable){s.baseFilters={};let h=o.syncUrl&&typeof window<"u"&&window.location?To(window.location.search):null;if(!!!(h&&(Object.keys(h.selected).length>0||h.time||h.bbox)))for(let[Q,J]of Object.entries(A)){let{baseKey:de,negated:Ie}=Pe(Q),ye=jt(Rt(s,de,"freva",s.flavour),Ie),je=(a=s.selected)[ye]??(a[ye]=[]);for(let Je of J)je.includes(Je)||je.push(Je)}}else s.baseFilters=A;let v=new Set,b=!1,k=!1,M=null;if(o.syncUrl&&typeof window<"u"&&window.location)try{let h=To(window.location.search);h.flavour&&s.flavours.includes(h.flavour)?(s.flavour=h.flavour,n.lensValue.textContent=s.flavour):h.flavour&&(M=h.flavour);let C=nt(s);for(let Q of Object.keys(h.selected))C.has(fe(Q).toLowerCase())&&delete h.selected[Q];if(Object.keys(h.selected).length){s.selected=h.selected;for(let Q of Object.keys(h.selected))v.add(Q)}h.time&&(s.time=h.time,v.add("time"),v.add("time_select")),h.bbox&&(s.bbox=h.bbox,v.add("bbox"),v.add("bbox_select")),h.flavour&&s.flavours.includes(h.flavour)&&v.add("flavour")}catch{}let B=Ft(d),D=Ft(d),P=yn(o.devNotes),U=null,q=null,Z=null,K=0;function ne(h){let C=h.toLowerCase();return/fail|error|could not|too big|couldn|unable|denied/.test(C)?"error":/not applied|warn|narrow it|up to \d|still loading/.test(C)?"warn":/downloaded|done|copied|complete|added|applied/.test(C)?"success":"info"}function W(h,C=0){let Q=Date.now();C<=0&&Q<K||(s.status=h,Z&&h?Z.log(ne(h),h):n.statusMsg.textContent=h,K=C>0?Q+C:0)}function le(){K=0}function $(){let h;s.search==="loading"&&s.rows.length===0?h="Searching\u2026":s.search==="error"?h="Search failed":s.rows.length===0?h="No files match":h=`${s.totalCount.toLocaleString("en-US")} ${s.totalCount===1?"file":"files"}`,n.resCount.textContent=h;let C=s.search==="loading";n.resSpin.classList.toggle("show",C),r.searchSpin.classList.toggle("show",C),r.searchRegion.setAttribute("aria-busy",C?"true":"false");let Q=C?"Searching":"";r.searchStatus.textContent!==Q&&(r.searchStatus.textContent=Q)}function pe(){return s.totalCount>1e5}function w(){let h=pe(),C=h?`Too many files to export (>${1e5.toLocaleString("en-US")}) - narrow the query`:"Export catalogue";n.exportBtn.setAttribute("aria-disabled",h?"true":"false"),n.exportBtn.classList.toggle("is-disabled",h),n.exportBtn.setAttribute("data-tip",C),n.exportBtn.setAttribute("aria-label",h?`Export catalogue - unavailable: too many files (>${1e5.toLocaleString("en-US")}); narrow the query`:"Export catalogue")}let y=["bg","surface","surface-2","surface-3","text","dim","faint","border","border-2","accent","accent-2","accent-soft","good","warn","danger","ocean","land"];function S(){let h=n.app.style,C=J=>typeof J=="string"&&J.length>0&&J.length<200&&!/[{}<>;]/.test(J);for(let J of y)h.removeProperty(`--${J}`);let Q={...o.theme.both??{},...o.theme[s.theme]??{}};for(let[J,de]of Object.entries(Q))C(de)&&h.setProperty(`--${J}`,de);C(o.theme.font)?h.setProperty("--ui",o.theme.font):h.removeProperty("--ui")}function O(){n.app.setAttribute("data-theme",s.theme),S(),r.themeKnob.replaceChildren(R(s.theme==="night"?E.moon:E.sun,{size:15}))}function F(){if(i.classList.toggle("metaview",s.layout==="overview"),s.layout==="overview"){let h=r.shelveBtn.querySelector(".tbtn-lbl");h&&(h.textContent=s.overviewStacked?"Unstack":"Stack")}r.resultsCtrl.classList.toggle("on",s.layout==="results"),r.resultsCtrl.setAttribute("aria-pressed",s.layout==="results"?"true":"false"),r.overviewCtrl.classList.toggle("on",s.layout==="overview"),r.overviewCtrl.setAttribute("aria-pressed",s.layout==="overview"?"true":"false"),X()}let I=null,ee=!1;function V(h){r.panelCtl.classList.toggle("in",h),r.panelCtl.setAttribute("aria-hidden",h?"false":"true"),r.panelCtl.querySelectorAll("button").forEach(C=>{C.tabIndex=h?0:-1}),r.resBar.classList.toggle("merged",h&&s.layout==="overview"),r.shelveBtn.hidden=s.layout!=="overview"||ee&&h}function X(){if(I?.disconnect(),I=null,ee=!1,s.layout!=="overview"||typeof IntersectionObserver!="function"){V(!0);return}ee=!0,V(!1),I=new IntersectionObserver(h=>V(h.some(C=>C.isIntersecting)),{root:r.resultsScroll,rootMargin:"0px 0px -120px 0px",threshold:0}),I.observe(n.results)}d.add(()=>{I?.disconnect(),I=null});function he(){r.listSeg.classList.toggle("on",s.view==="list"),r.gridSeg.classList.toggle("on",s.view==="grid")}function ue(){i.classList.toggle("side-collapsed",s.sidebarCollapsed),r.sideCollapse.setAttribute("aria-label",s.sidebarCollapsed?"Expand filters":"Collapse filters"),r.sideCollapse.setAttribute("aria-expanded",s.sidebarCollapsed?"false":"true")}function ce(){s.sidebarCollapsed=!s.sidebarCollapsed,Un(s.sidebarCollapsed),ue()}let ke=new Map;function me(h,C,Q){if(ke.get(h)===C)return;ke.set(h,C);let J=P.start(`render:${h}`);Q(),J()}let we=()=>JSON.stringify(s.selected),Y=()=>JSON.stringify([s.time,s.bbox]);function f(){me("sidebar",JSON.stringify([s.facetsVersion,we(),Y(),s.flavour,[...s.sidebarOpen],s.sidebarAddOpen,s.metadataVersion]),()=>es(_))}function x(){me("chips",JSON.stringify([we(),Y()]),()=>Yn(_))}function H(){s.layout==="overview"&&me("overview",JSON.stringify([s.facetsVersion,Y(),s.flavour,s.overviewStale,s.metadataVersion,s.overviewSort,s.overviewSpan,s.overviewH,[...s.overviewCollapsed],s.overviewAddOpen,s.overviewOrder,s.overviewStacked]),()=>or(_))}function j(){me("results",JSON.stringify([s.rowsVersion,s.view,s.search,s.searchError??"",s.totalCount]),()=>Xi(_))}function oe(){U?.render()}function ie(){f(),x(),H(),Xa(_),oe()}function Ae(){ae(),i.classList.remove("flavour-loading"),f(),x(),H(),j(),Qt(_),Re(_),oe(),$(),w()}async function xe(h){let C=h?s.rows.length:0,Q=p.nextRequestId();s.lastRequestId=Q;let J=Ge,de=p.channelSignal("search");s.search="loading",h||(s.start=0),j(),$(),h||W("Searching\u2026");try{let Ie=P.start("search:fetch+parse"),ye=await p.extendedSearch(s.flavour,s.uniqKey,bt(s),{start:C,signal:de});if(Ie(),s.lastRequestId!==Q||Ge!==J)return;P.time("search:normalize",()=>{let je=ye.search_results.map(Je=>$r(Je,s.uniqKey));if(h)for(let Je of je)s.rows.push(Je);else s.rows=je,s.rowsEpoch++;qt=J,s.rowsVersion++,s.totalCount=typeof ye.total_count=="number"?ye.total_count:s.rows.length,Q>=T&&(T=Q,s.facets=So(ye),s.facetsVersion++,s.primaryFacets=ye.primary_facets??[],s.facetMapping=ye.facet_mapping??{},vo(s))}),s.start=s.rows.length,s.search=s.rows.length===0?"empty":"loaded",s.searchError=void 0,s.overviewStale=!1,Ae(),W(s.rows.length===0?"No files match the current filters.":"")}catch(Ie){if(Ie instanceof De&&Ie.aborted||s.lastRequestId!==Q||Ge!==J)return;let ye=Ie instanceof De?Ie.message:"Search failed.";if(h){s.search="loaded",j(),$(),W(`Could not load more results: ${ye}`,4e3);return}s.search="error",s.searchError=ye,i.classList.remove("flavour-loading"),j(),$(),w(),W(ye,4e3),ae(k)}}function ae(h=!1){if(b||!o.syncUrl)return;let C=Xt(s);if(C.size===0){if(!h)return;C=new Set([...ht,...Bt,...We].map(J=>J.toLowerCase()))}b=!0;let Q=!1;for(let J of Object.keys(s.selected))C.has(fe(J).toLowerCase())||(delete s.selected[J],v.delete(J),Q=!0);Q&&(Oe(),be())}function Oe(){if(!(!o.syncUrl||typeof window>"u"||!window.history?.replaceState))try{let h=new URLSearchParams(window.location.search);for(let J of v)h.delete(J);v.clear(),s.flavour&&s.flavour!=="freva"&&(h.set("flavour",s.flavour),v.add("flavour"));for(let[J,de]of Mo(s))h.append(J,de),v.add(J);let C=h.toString(),Q=window.location.pathname+(C?`?${C}`:"")+window.location.hash;window.history.replaceState(null,"",Q)}catch{}}let Be=!1,ct=!1,tt=()=>Object.keys(s.baseFilters).length===0||Lt.includes(s.flavour)||!!s.flavourMaps[s.flavour];function Ue(){s.search="error",s.searchError="Scoped browsing is unavailable - this flavour\u2019s field mapping could not be loaded.",i.classList.remove("flavour-loading"),j(),$()}function be(){if(!tt()){ct?Ue():(Be=!0,s.search="loading",i.classList.add("flavour-loading"),j());return}Ge++,Oe(),p.channelSignal("search"),p.channelSignal("recount"),B(()=>void xe(!1),Td)}function Ye(){s.search!=="loading"&&qt===Ge&&xe(!0)}function dt(){xe(!1)}function Et(){D(()=>{(async()=>{let h=p.nextRequestId();pt=h;let C=Ge,Q=p.channelSignal("recount");try{let J=await p.metadataSearch(s.flavour,s.uniqKey,bt(s),Q);if(pt!==h||s.lastRequestId>h||h<T||Ge!==C)return;T=h,s.facets=So(J),s.facetsVersion++,s.primaryFacets=J.primary_facets??s.primaryFacets,s.facetMapping=J.facet_mapping??s.facetMapping,s.overviewStale=!1,vo(s)}catch(J){if(J instanceof De&&J.aborted||pt!==h||s.lastRequestId>h||Ge!==C)return;s.overviewStale=!0}H(),f()})()},Ed)}let pt=0,Ge=0,qt=0,T=0;function N(h,C,Q){Qe(s,h)||(s.externalEdits++,qr(s,h,C,Q),ie(),be())}function G(h,C){N(h,C,!1)}function te(h,C){N(h,C,!0)}function se(h){Qe(s,h)||rt(s,h)!==0&&(s.externalEdits++,Kr(s,h),ie(),be())}function Me(h,C){Qe(s,h)||(C?Ze(s,h).length:Kt(s,h).length)===0||(s.externalEdits++,Wr(s,h,C),ie(),be())}function Fe(){s.externalEdits++,Jr(s),ie(),be()}function Le(h){s.externalEdits++,s.time=h,ie(),be()}let Ve=(h,C)=>!h&&!C||!!h&&!!C&&h.from===C.from&&h.to===C.to&&h.mode===C.mode,Vt=(h,C)=>!h&&!C||!!h&&!!C&&h.minLon===C.minLon&&h.maxLon===C.maxLon&&h.minLat===C.minLat&&h.maxLat===C.maxLat&&h.mode===C.mode;function Ee(h){s.externalEdits++,s.bbox=Do(h),ie(),be()}function Xe(h){if(s.externalEdits++,h===s.flavour)return;let C=s.flavour;s.flavour=h,s.selected=Xr(s,s.selected,C,h),s.overviewShape=[],ho(),n.lensValue.textContent=h,i.classList.add("flavour-loading"),ie(),be()}function ze(h){s.layout=h,Fn(h),F(),h==="overview"&&(H(),Xa(_),Et(),r.resultsScroll.scrollTo?.({top:0}))}function Dt(h){s.view=h,Nn(h),he(),j()}let ga=0;function so(h){let C=s.theme!==h;s.theme=h,Pn(h);let Q=++ga;n.app.setAttribute("data-notransition","true"),O(),d.raf(()=>{d.raf(()=>{Q===ga&&n.app.removeAttribute("data-notransition")})}),C&&o.theme.onModeChange?.(h)}function lo(h){s.detailsOpen=h??!s.detailsOpen,n.infoBtn.classList.toggle("on",s.detailsOpen),n.infoBtn.setAttribute("aria-pressed",s.detailsOpen?"true":"false"),Re(_)}function Rs(h){s.focusKey=h,s.detailSource="focus",ua(_,[]),s.detailsOpen&&Re(_)}function Fs(h){if(s.pickedKeys.has(h))s.pickedKeys.delete(h);else{if(s.pickedKeys.size>=25)return W(`You can select up to ${25} files - deselect one to choose another.`,3e3),_.toast("warn",`Selection is limited to ${25} files. Deselect one first.`),!1;s.pickedKeys.add(h)}return s.detailSource="picks",ua(_,[h]),Qt(_),s.detailsOpen&&Re(_),!0}function zs(){s.pickedKeys.clear(),s.detailSource="focus",ua(_),Qt(_),s.detailsOpen&&Re(_)}function Ns(h){s.terminalDraft=h;let C=ka(wt(h)),{accepted:Q,rejected:J}=_t(s,C.rest),de=!Ve(s.time,C.time),Ie=!Vt(s.bbox,C.bbox);return _r(s.selected,Q)&&!de&&!Ie||(s.selected=Q,de&&(s.time=C.time),Ie&&(s.bbox=Do(C.bbox)),x(),f(),H(),Xa(_),be()),J.length+C.errors.length}function Hs(h){return ka(wt(h)).errors}function Us(h){Ti(_,h)}function Gs(h){Di(_,h)}function Qs(){u.close()}let kr=new Map,Ne=!1;function co(h){kr.get(h)?.flush();let C=d.child();return kr.set(h,C),queueMicrotask(()=>{Ne||u.closeIfAnchorDetached()}),C}let _={state:s,api:p,dis:d,region:co,cfg:o,roots:n,popover:u,commitSearch:be,loadNextPage:Ye,retrySearch:dt,syncAll:ie,renderSidebar:f,renderChips:x,renderResults:j,renderOverview:H,renderCommand:oe,renderDetails:()=>Re(_),recountOverview:Et,exportCatalogue:(h,C,Q)=>Or(h,C,Q),toggleFacet:G,excludeFacet:te,clearAllFacets:Fe,clearFacet:se,clearFacetMode:Me,setTime:Le,setBbox:Ee,setFlavour:Xe,setLayout:ze,setView:Dt,setTheme:so,toggleDetails:lo,focusFile:Rs,togglePick:Fs,clearPicks:zs,applyTerminalDraft:Ns,terminalDraftErrors:Hs,openTimeEditor:Us,openBboxEditor:Gs,closeAllPopovers:Qs,setStatus:W,log:(h,C)=>Z?.log(h,C),toast:(h,C)=>Z?.toast(h,C),openInspect:h=>Mr.open(h),openHelp:()=>po(!0)};function po(h){r.helpPanel.classList.toggle("show",h??!r.helpPanel.classList.contains("show"))}d.listen(r.helpPanel.querySelector(".help-x"),"click",()=>po(!1)),d.listen(document,"keydown",h=>{let C=h;C.key==="Escape"&&!C.defaultPrevented&&!u.isOpen()&&po(!1)}),Z=Is(_),Ls(n.app,d);let Mr=vi(_);o.features.terminal&&(U=ys(_)),o.devNotes&&(q=Ss(_)),d.listen(r.themeBtn,"click",()=>so(s.theme==="night"?"day":"night")),d.listen(r.sideCollapse,"click",()=>ce()),d.listen(r.listSeg,"click",()=>Dt("list")),d.listen(r.gridSeg,"click",()=>Dt("grid")),d.listen(r.shelveBtn,"click",()=>{Ni(_),Tr()});function Tr(){let h=s.overviewStacked,C=r.shelveBtn.querySelector(".tbtn-lbl");C&&(C.textContent=h?"Unstack":"Stack"),r.shelveBtn.setAttribute("aria-pressed",h?"true":"false"),r.shelveBtn.setAttribute("aria-label",h?"Unstack - restore the block layout":"Stack all blocks into full-width rows"),r.shelveBtn.classList.toggle("on",h)}Tr(),d.listen(r.selectAllBtn,"click",()=>{let h=dr(_);if(h.willClear)s.pickedKeys.clear(),W("Selection cleared.");else{s.pickedKeys.clear();for(let C of h.target)s.pickedKeys.add(C);W(h.capped?`Selected the first ${25} of ${s.rows.length.toLocaleString("en-US")} listed files - ${h.omitted.toLocaleString("en-US")} were not selected.`:`Selected ${h.target.length} file${h.target.length===1?"":"s"}.`,h.capped?4e3:0)}s.detailSource="picks",ua(_),Qt(_),s.detailsOpen&&Re(_)}),d.listen(r.resultsCtrl,"click",()=>ze("results")),d.listen(r.overviewCtrl,"click",()=>ze("overview")),d.listen(n.infoBtn,"click",()=>lo()),d.listen(r.cmdBtn,"click",()=>U?.toggle()),d.listen(r.inspectBtn,"click",()=>{Mr.openEmpty()}),d.listen(n.clearAllBtn,"click",()=>Fe());let Er=n.infoClose;if(Er&&d.listen(Er,"click",()=>lo(!1)),r.notesBtn&&q){let h=r.notesBtn,C=q;d.listen(h,"click",()=>{C.toggle(),h.classList.toggle("on",C.isShown())})}o.features.search&&Ds(_,r.searchInput),Os(_,r.searchInput,r.searchAside),d.listen(r.lensBtn,"click",()=>{if(u.isOpen()){u.close();return}let h=co("popover"),C=[];for(let Q of s.flavours){if(Q==="user"&&!o.authEnabled)continue;let J=Q===s.flavour,de=c("button",{class:`pop-item${J?" check on":""}`,type:"button"},[c("span",{class:"desc",text:Q}),J?c("span",{class:"tick"},[R(E.check,{size:13})]):null]);h.listen(de,"click",()=>{u.close(),Xe(Q)}),C.push(de)}r.lensBtn.setAttribute("aria-expanded","true"),u.open(r.lensBtn,C,{placement:"below",className:"lens-pop",autoFocus:!0,onClose:()=>r.lensBtn.setAttribute("aria-expanded","false")})}),d.listen(n.exportBtn,"click",()=>{if(pe()){W(`This query returns more than ${1e5.toLocaleString("en-US")} files - narrow it before exporting a catalogue.`);return}if(u.isOpen()){u.close();return}let h=co("popover");n.exportBtn.setAttribute("aria-expanded","true"),u.open(n.exportBtn,_a(h,{heading:Ui(s.totalCount),onPick:C=>{u.close(),Or(C)}}),{placement:"below",className:"export-pop",autoFocus:!0,onClose:()=>n.exportBtn.setAttribute("aria-expanded","false")})});let uo=0;function Dr(h){uo=Math.max(0,uo+h),n.exportBtn.classList.toggle("busy",uo>0)}async function Or(h,C,Q){let J={intake:{label:"Intake catalogue",filename:"freva-intake.json"},stac:{label:"STAC catalogue",filename:"freva-stac.zip"},uris:{label:"URI manifest",filename:"freva-uris.txt"}}[h],{label:de,filename:Ie}=J,ye=Q??s.uniqKey,je=C??bt(s),Je=h==="uris"?p.dataSearchUrl(s.flavour,ye,je):p.catalogueUrl(h,s.flavour,ye,je);if(le(),Je.length>6e3){let ge=`This selection makes too long a request for ${de} - select fewer files and try again.`;W(ge,4e3),_.toast("error",ge);return}let Pr=ge=>{let Ke=document.createElement("a");Ke.href=ge,Ke.download=Ie,Ke.rel="noopener",document.body.appendChild(Ke),Ke.click(),Ke.remove()};if(!o.getAuthToken()){let ge=new AbortController,Ke=d.add(()=>ge.abort());try{let at=await fetch(Je,{method:"HEAD",signal:ge.signal});if(Ne)return;if(!at.ok&&at.status!==405){let Rr=at.status===413?"This query is too large for the server to export - narrow it and try again.":at.status===414?`The request URL is too long for ${de} - select fewer files and try again.`:`${de} couldn't be prepared (server responded ${at.status}).`;W(Rr),_.toast("error",Rr);return}}catch{if(Ne||ge.signal.aborted)return}finally{Ke()}if(Ne)return;Pr(Je),W(`${de} download started.`),_.toast("success",`${de} download started.`);return}let mo=new AbortController,Ws=d.add(()=>mo.abort());Dr(1),W(`Preparing ${de}\u2026`);try{let Ke=await(h==="uris"?await p.manifestResponse(s.flavour,ye,je,mo.signal):await p.catalogueResponse(h,s.flavour,ye,je,mo.signal)).blob(),at=URL.createObjectURL(Ke);Pr(at),d.setTimeout(()=>URL.revokeObjectURL(at),0),Ne||_.toast("success",`${de} downloaded.`)}catch(ge){if(ge instanceof De&&ge.aborted||ge instanceof DOMException&&ge.name==="AbortError")return;if(ge instanceof De&&ge.status===414){_.toast("error","The request URL is too long - select fewer files and try again.");return}if(ge instanceof De&&(ge.status===413||ge.detail===go)){w(),_.toast("warn","The result stream is too big to export - narrow the query and try again.");return}_.toast("error",ge instanceof De?ge.message:"Catalogue export failed.")}finally{Ws(),Dr(-1)}}O(),F(),he(),ue(),f(),x(),j(),Qt(_),Re(_),oe(),$(),w(),e.appendChild(r.outer);let fo=Object.keys(s.baseFilters).length>0&&!Lt.includes(s.flavour),Ot=!1,Ir=()=>{Ot||Ne||(Ot=!0,xe(!1))},Lr=()=>{Ot||Ne||(Ot=!0,Ue())};fo?(s.search="loading",j()):Ir(),(async()=>{try{let h=await Tn(o,d,n.app);if(Ne)return;s.metadata=h,s.metadataVersion++,f(),H()}catch{}})();let Br=null;function ho(){let h=Br;if(!h)return;if(Array.isArray(h)){s.attributeKeys=h;return}let C=h[s.flavour];s.attributeKeys=Array.isArray(C)?C:[...new Set(Object.values(h).flat())]}return(async()=>{try{let h=await p.overview();if(Ne)return;let C=Array.isArray(h.flavours)?h.flavours:[],Q=[...Lt];for(let J of C)Q.includes(J)||Q.push(J);s.flavours=Q,M&&s.flavours.includes(M)&&M!==s.flavour&&(s.flavour=M,M=null,ho(),n.lensValue.textContent=s.flavour,ie(),Oe(),be()),Br=h.attributes??null,ho(),ae(),oe()}catch{k=!0,ae(!0)}})(),(async()=>{try{let h=await p.listFlavours();if(Ne)return;let C=Array.isArray(h.flavours)?h.flavours:[],Q=JSON.stringify(s.flavourMaps[s.flavour]?.forward??null),J=bt(s);s.flavourMaps={...s.flavourMaps,...xo(C)};let de=JSON.stringify(s.flavourMaps[s.flavour]?.forward??null),Ie=bt(s);if(ct=!0,fo&&!Ot){s.flavourMaps[s.flavour]?(oe(),Ir()):Lr();return}if(Be){Be=!1,f(),x(),H(),oe(),be();return}Q!==de&&(f(),x(),H(),oe(),J!==Ie&&be())}catch{ct=!0,fo&&!Ot?Lr():Be&&(Be=!1,Ue())}})(),{destroy(){Ne=!0,u.close(),d.flush(),r.outer.remove()},getState(){return structuredClone(s)},setTheme(h){Ne||so(h)}}}var jf=Cr;export{ao as SEARCH_INTENT_TEXT_KEY,Bs as SEARCH_INTENT_VERSION,ma as SEARCH_INTENT_VERSION_KEY,no as applySearchIntentV1,jf as default,Cr as mountDataBrowser,Cd as mountDataBrowserFromIntent,ro as parseSearchIntentV1,Ps as serializeSearchIntentV1,oo as textToFacetValue};
