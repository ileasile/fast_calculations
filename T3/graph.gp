set term png
set datafile separator ','
set key autotitle columnhead
set output "graphs.png"
set key outside
plot for [col=2:11] "out_c.csv" using 1:col with lines
