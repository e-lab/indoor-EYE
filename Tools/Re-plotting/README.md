# How to cut stuff and re-plot it

Getting first chunk of data (minus n corrupted lines)
```bash
head -n-1 SW11A-1/accuracy.log > SW11A/accuracy.log
```

Appending other chunks of data (without 1 line header)
```bash
tail -n+2 SW11A-2/accuracy.log >> SW11A/accuracy.log
```

Show the results
```bash
cat SW11A/accuracy.log
```

Plotting and creating eps (**warning**: previously saved charts will be overwritten)
```bash
gnuplot -e "fileName='./HW2C/accuracy.log'; figure='accuracy'" plotCharts.plt"
gnuplot -e "fileName='./HW2C/cross-entropy.log'; figure='cross-entropy'" plotCharts.plt"
```
