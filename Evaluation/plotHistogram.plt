# Plot histograms

set terminal postscript eps size 6.5,6.5 enhanced color font 'Helvetica,35'


# Options for evalType: "single", "multi"
evalType="multi"

set yrange [*:]
set xlabel "Dataset" font "Helvetica,45" 
set ylabel "Rank-1 Matching rate (%)" offset 1.5 font "Helvetica,45"

if(evalType eq "single"){
saveFile="./pca_numPatch_impact_single.eps"
set title "PCA and number of patches: single-shot" font "Helvetica,45"
set xtics scale 0 ("VIPeR" 0, "Market1501" 1, "Airport" 2, "CUHK03" 3, "GRID" 4, "HDA" 5, "3DPeS" 6) rotate by 45 right
}
if(evalType eq "multi"){
saveFile="./pca_numPatch_impact_multi.eps"
set title "PCA and number of patches: multi-shot" font "Helvetica,45"
set xtics scale 0 ("SAIVT-38" 0, "SAIVT-58" 1, "iLIDSVID" 2, "PRID" 3, "WARD-12" 4, "WARD-13" 5, "RAiD-12" 6, "RAiD-13" 7, "RAiD-14" 8, "CAVIAR" 9, "V47" 10) rotate by 45 right
}

set output saveFile

#set xtics scale 1 ("SAIVT-38" 0, "SAIVT-58" 1.2, "iLIDSVID" 2.4, "PRID" 3.6, "WARD-12" 4.8, "WARD-13" 6, "RAiD-12" 7.2, "RAiD-13" 8.4, "RAiD-14" 9.6, "CAVIAR" 10.8, "V47" 12, "Market1501" 13.2, "VIPeR" 14.4, "Airport" 15.6, "CUHK03" 16.8, "GRID" 18, "HDA" 19.2, "3DPeS" 20.4) rotate by 45 right

set offset -0.3,-0.3,0,0

set key top left box opaque
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
plot "PCA_numPatch_rank1_results.dat" every ::11::17 using 1 title "patch:6, no pca",\
 '' every ::11::17 using 2 title "patch:6, pca",\
 '' every ::11::17 using 3 title "patch:9, pca",\
 '' every ::11::17 using 4 title "patch:15, pca",\
 '' every ::11::17 using 5 title "patch:24, pca"
}
if(evalType eq "multi"){
plot "PCA_numPatch_rank1_results.dat" every ::0::10 using 1 title "patch:6, no pca",\
 '' every ::0::10 using 2 title "patch:6, pca",\
 '' every ::0::10 using 3 title "patch:9, pca",\
 '' every ::0::10 using 4 title "patch:15, pca",\
 '' every ::0::10 using 5 title "patch:24, pca"
}
