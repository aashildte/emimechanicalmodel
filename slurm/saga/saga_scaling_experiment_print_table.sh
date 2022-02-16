#!/bin/bash
# EMI Mechanical model
#
# Copyright (C) 2021 Simula Research Laboratory
# Authors: James D. Trotter <james@simula.no>
#
# This script reads the output from a series of scaling experiments
# for the EMI Holzapfel-Ogden model, and prints a table of
# performance-related information.
#
# Example usage:
#
#  $ ./saga_scaling_experiment_print_table.sh
#

printf "%-100s\t%12s\t%12s\t%12s\t%9s\t%10s\n" "File" "Max time" "Max RSS [GB]" "Total RSS [GB]" "Max DOFs" "Newton its"

for f in ${@}; do
    fout=${f}/$(echo "${f}" | cut -d- -f2-)-stdout.txt
    ferr=${f}/$(echo "${f}" | cut -d- -f2-)-stderr.txt
    max_time=$(awk '
BEGIN { max_time=0 }
/Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): [[:digit:]]+:[[:digit:]]{2}:[[:digit:]]{2}/ {
  split($9,process_time_hhmmss,":")
  process_time=process_time_hhmmss[1]*3600+process_time_hhmmss[2]*60+process_time_hhmmss[3]
  if (max_time < process_time) max_time = process_time
}
/Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): [[:digit:]]+:[[:digit:]]{2}.[[:digit:]]{2}/ {
  split($9,process_time_mmss,":")
  process_time=process_time_mmss[1]*60+process_time_mmss[2]
  if (max_time < process_time) max_time = process_time
}
END {
  h=int(max_time/3600)
  m=int((max_time-3600*h)/60)
  s=max_time-3600*h-60*m
  printf "%d:%02d:%02d\n",h,m,s
}' ${ferr})

    max_rss=$(awk '
BEGIN { max_rss=0 }
/Maximum resident set size/ {
  process_rss=$7
  if (max_rss < process_rss) max_rss = process_rss
}
END {
  printf "%.2f\n",(max_rss/1000/1000)
}' ${ferr})

    total_rss=$(awk '
BEGIN { total_rss=0 }
/Maximum resident set size/ {
  total_rss = total_rss + $7
}
END {
  printf "%.2f\n",(total_rss/1000/1000)
}' ${ferr})

    max_dofs=$(awk '
BEGIN { max_dofs=0 }
/Degrees of freedom/ {
  process_dofs=$5
  if (max_dofs < process_dofs) max_dofs = process_dofs
}
END {
  printf "%8d\n",process_dofs
}' ${fout})

    newton_its=$(awk '
BEGIN { newton_its=0 }
/Newton iteration/ {
  newton_its=newton_its+1
}
END {
  printf "%6d\n",newton_its
}' ${fout})

    name=$(basename ${f})
    printf "%100s\t%12s\t%12s\t%12s\t%9s\t%10s\n" "${name:0:100}" "${max_time}" "${max_rss}" "${total_rss}" "${max_dofs}" "${newton_its}"
done
