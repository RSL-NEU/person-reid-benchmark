# Plot histograms

set terminal postscript eps size 6.5,6.5 enhanced color font 'Helvetica,35'


# Options for evalType: "single", "multi"
evalType="single"

set yrange [*:]
set xlabel "Attribute" font "Helvetica,45" 
set ylabel "Mean rank-1 Matching rate (%)" offset 1.5 font "Helvetica,45"

if(evalType eq "single"){
saveFile="./attr_algo_l2_single.eps"
set title "Mean rank-1 performance vs. attributes: single-shot" font "Helvetica,45"
set xtics scale 0 ("VV" 0, "IV" 1, "BC" 2, "OCC" 3, "RES" 4, "DE" 5) 
}
if(evalType eq "multi"){
saveFile="./attr_algo_l2_multi.eps"
set title "Mean rank-1 performance vs. attributes: multi-shot" font "Helvetica,45"
set xtics scale 0 ("VV" 0, "IV" 1, "BC" 2, "OCC" 3, "RES" 4) 
}

set output saveFile

#set xtics scale 1 ("SAIVT-38" 0, "SAIVT-58" 1.2, "iLIDSVID" 2.4, "PRID" 3.6, "WARD-12" 4.8, "WARD-13" 6, "RAiD-12" 7.2, "RAiD-13" 8.4, "RAiD-14" 9.6, "CAVIAR" 10.8, "V47" 12, "Market1501" 13.2, "VIPeR" 14.4, "Airport" 15.6, "CUHK03" 16.8, "GRID" 18, "HDA" 19.2, "3DPeS" 20.4) rotate by 45 right

set offset -0.3,-0.3,0,0

set key top right box opaque
set key box lt -1
set key width 1
set key font 'Helvetica,38'
set style data histograms
set style fill solid border rgb "black"
set style histogram cluster gap 1.5
set grid ytics lc rgb "#bbbbbb" lw 6 lt 0
set grid xtics lc rgb "#bbbbbb" lw 6 lt 0
show grid

if(evalType eq "single"){
plot "Attributes_features_l2_single.dat" using 1 title "LOMO",\
 '' using 2 title "LDFV",\
 '' using 3 title "ELF",\
 '' using 4 title "histLBP",\
 '' using 5 title "DenColor",\
 '' using 6 title "gBiCov"
}
if(evalType eq "multi"){
plot "Attributes_features_l2_multi.dat" using 1 title "LOMO",\
 '' using 2 title "LDFV",\
 '' using 3 title "ELF",\
 '' using 4 title "histLBP",\
 '' using 5 title "DenColor",\
 '' using 6 title "gBiCov"
}
