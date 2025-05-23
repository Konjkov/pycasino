#!/usr/bin/gnuplot -persist

set grid

#set xrange
set yrange [0:]

set xlabel "casino CPU time"
set ylabel "Pycasino CPU time"

set mxtics 5
set mytics 2

f1(x) = a1*x+b1
fit f1(x) "time.dat" using 2:3 via a1, b1

f2(x) = a2*x+b2
fit f2(x) "time.dat" using 4:5 via a2, b2

f3(x) = a3*x+b3
fit f3(x) "time.dat" using 6:7 via a3, b3

plot 'time.dat' using 2:3 with points title "Slater CPU time", f1(x),\
     'time.dat' using 4:5 with points title "Jastrow CPU time", f2(x),\
     'time.dat' using 6:7 with points title "Backflow CPU time", f3(x)
