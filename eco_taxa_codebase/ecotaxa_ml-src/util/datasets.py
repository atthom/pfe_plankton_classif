"""
Constants defining the data sets, grouings (taxonomic levels) and CV splits
"""

__all__ = ["DATASETS", "GROUPINGS", "SPLITS"]

DATASETS = ["flowcam", "uvp5ccelter", "zoocam", "zooscan"]
GROUPINGS = ["group1", "group2"]
SPLITS = ["0", "1", "2"]

ZOOPROCESS_FIELDS = {
    #Excluded for NaN: perimareaexc, feretareaexc, cdexc, convarea_area, symetrieh_area, symetriev_area, nb1_area, nb2_area, nb3_area, skeleton_area
    "flowcam": "area,meanimagegrey,mean,stddev,min,perim.,width,height,major,minor,circ.,feret,intden,median,skew,kurt,%area,area_exc,fractal,skelarea,slope,histcum1,histcum2,histcum3,nb1,nb2,nb3,symetrieh,symetriev,symetriehc,symetrievc,convperim,convarea,fcons,thickr,esd,elongation,range,meanpos,centroids,cv,sr,perimferet,perimmajor,circex,kurt_mean,skew_mean,convperim_perim,nb1_range,nb2_range,nb3_range,median_mean,median_mean_range",
    # Excuded for NaN: perimareaexc, feretareaexc, cdexc, areai
    "uvp5ccelter": "area,mean,stddev,mode,min,max,perim.,width,height,major,minor,circ.,feret,intden,median,skew,kurt,%area,area_exc,fractal,skelarea,slope,histcum1,histcum2,histcum3,nb1,nb2,nb3,symetrieh,symetriev,symetriehc,symetrievc,convperim,convarea,fcons,thickr,esd,elongation,range,meanpos,centroids,cv,sr,perimferet,perimmajor,circex,kurt_mean,skew_mean,convperim_perim,convarea_area,symetrieh_area,symetriev_area,nb1_area,nb2_area,nb3_area,nb1_range,nb2_range,nb3_range,median_mean,median_mean_range,skeleton_area",
    "zoocam": "area,%area,area_exc,symetrieh,symetriev,elongation,area_based_diameter,meangreyimage,meangreyobjet,modegreyobjet,sigmagrey,mingrey,maxgrey,sumgrey,breadth,length,perim,minferetdiam,maxferetdiam,meanferetdiam,feretelongation,compactness,intercept0,intercept45,intercept90,intercept135,convexhullarea,convexhullfillratio,convexperimeter,n_number_of_runs,n_chained_pixels,n_convex_hull_points,n_number_of_holes,transparence,roughness,rectangularity,skewness,kurtosis,fractal_box,hist25,hist50,hist75,valhist25,valhist50,valhist75,nobj25,nobj50,nobj75,thick_r,cdist,bord",
    "zooscan": "area,mean,stddev,mode,min,max,perim.,width,height,major,minor,circ.,feret,intden,median,skew,kurt,%area,area_exc,fractal,skelarea,slope,histcum1,histcum2,histcum3,nb1,nb2,nb3,symetrieh,symetriev,symetriehc,symetrievc,convperim,convarea,fcons,thickr,esd,elongation,range,meanpos,centroids,cv,sr,perimareaexc,feretareaexc,perimferet,perimmajor,circex,cdexc"
}