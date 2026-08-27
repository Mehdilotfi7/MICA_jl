#!/bin/bash
set -e

cd benchmarking/tcpd/loss_sweep

# Get dataset count
n_datasets=$(julia --project=../../../codes/Mica.jl -e '
using JSON
ann = JSON.parsefile("benchmarking/tcpd/dataset/annotations.json")
dir = "benchmarking/tcpd/dataset/datasets"
let count = 0
    for d in readdir(dir)
        if isdir(joinpath(dir, d)) && isfile(joinpath(dir, d, "$(d).json")) && haskey(ann, d)
            cp_sets = Vector{Vector{Int}}()
            for (_, cp_list) in ann[d]
                if cp_list !== nothing && length(cp_list) > 0
                    cp_clean = filter(x -> x !== nothing, cp_list)
                    if length(cp_clean) > 0
                        push!(cp_sets, Int.(cp_clean))
                    end
                end
            end
            if length(cp_sets) > 0
                count += 1
            end
        end
    end
    println(count)
end
')

echo "Total datasets: $n_datasets"
echo "Launching 20 Julia processes for loss-function sweep..."

for i in $(seq 1 20); do
    start=$(( (i-1) * n_datasets / 20 + 1 ))
    end=$(( i * n_datasets / 20 ))
    if [ $start -le $end ]; then
        echo "  Process $i: datasets $start-$end"
        env JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 nohup julia --project=../../../codes/Mica.jl benchmark_tcpd_loss_sweep.jl $start $end "loss_p$(printf "%02d" $i)" > "benchmark_tcpd_loss_sweep_p$(printf "%02d" $i).log" 2>&1 &
        pids[$i]=$!
    fi
done

echo "Waiting for all processes..."
for i in $(seq 1 20); do
    if [ -n "${pids[$i]}" ]; then
        wait ${pids[$i]}
        echo "  Process $i (PID ${pids[$i]}) finished with exit code $?"
    fi
done

echo "All processes complete. Combining results..."

julia --project=../../../codes/Mica.jl -e '
using JSON
all_results = []
for i in 1:20
    suffix = "loss_p" * lpad(string(i), 2, "0")
    file = "benchmark_tcpd_loss_sweep_$(suffix).json"
    if isfile(file)
        data = JSON.parsefile(file)
        append!(all_results, data)
    end
end

open("benchmark_tcpd_loss_sweep_combined.json", "w") do f
    JSON.print(f, all_results, 2)
end

println("Combined results:")
println("  Total runs: $(length(all_results))")
'

echo "Done! Results saved to benchmark_tcpd_loss_sweep_combined.json"
