set term png
set datafile separator ','
set key autotitle columnhead
set output "graphs_tpedu.png"
set key outside
plot for [col=2:8] "out.csv" using 1:col with lines
