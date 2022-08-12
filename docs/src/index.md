# ProRF.jl Documentation
## Overview
`ProRF` provides a full process for applying the random forest model of protein sequences using `DecisionTree`.

## Install
```julia
using Pkg
Pkg.add(url="https://github.com/Chemical118/ProRF.jl")
```

## Examples
!!! tip "Performance Tip"
    `ProRF` support parallel computing, please turn on the julia with multiple threads. This can speed up execution time fairly.

    ```bash
    $ julia --threads auto
    ```
    For more information, read Multi-Threading [documentaion](https://docs.julialang.org/en/v1/manual/multi-threading/).

!!! note
    `ProRF` recommends interactive mode like `IJulia`. If you want to run in non-interactive mode, execute below code to see graphs. However, `ProRF` doesn't guarantee that you can see graphs.
    ```julia
    using ProRF
    julia_isinteractive(false)
    ```

### Data preprocessing
`ProRF` has a useful function for preprocessing data.
```julia
using ProRF, Printf

Find, Lind = data_preprocess_index("Data/algpdata.fasta", val_mode=true)
@printf "%d %d\n" Find Lind

data_preprocess_fill(Find, Lind,
                     "Data/algpdata.fasta",
                     "Data/Mega/ealtreedata.nwk",
                     "Data/jealgpdata.fasta",
                     val_mode=true);

view_sequence("Data/jealgpdata.fasta", save=true)
```

### Find best random forest arguments
`ProRF` helps you find arguments for the random forest.
```julia
using ProRF, Printf

RI = RFI("Data/jealgpdata.fasta", "Data/data.xlsx", 2:1:10, 100:10:500)
X, Y, L = get_data(RI, 2, 'D')

MeZ, SdZ = iter_get_reg_value(RI, X, Y, 10, val_mode=true)

view_reg3d(RI, MeZ, title="NRMSE value", azim=90, scale=3)
view_reg3d(RI, SdZ, title="NRMSE SD value", elev=120, scale=3)

N_Feature, N_Tree = get_reg_value_loc(RI, MZ)
@printf "Best Arguments : %d %d\n" N_Feature N_Tree
```

### Execute random forest
`ProRF` executes random forest flexibly and easily.
```julia
using ProRF, Printf

# Molecular mass of amino acid
myDict = Dict('A' => 89, 'R' => 174, 'N' => 132, 'D' => 133, 'C' => 121, 'Q' => 146,
'E' => 147, 'G' => 75, 'H' => 155, 'I' => 131, 'L' => 131, 'K' => 146, 'M' => 149, 
'F' => 165, 'P' => 115, 'S' => 105, 'T' => 119, 'W' => 204, 'Y' => 181, 'V' => 117)

R = RF("Data/jealgpdata.fasta", "Data/data.xlsx")
X, Y, L = get_data(R, 2, 'D', convert=myDict)

M = rf_model(X, Y, N_Feature, N_Tree)
@printf "Total NRMSE : %.6f\n" test_nrmse(M, X, Y)

MeF, SdF = iter_get_reg_importance(R, X, Y, L, N_Feature, N_Tree, 100, val_mode=true)
view_importance(R, L, MeF, SdF)

for (fe, loc) in sort(collect(zip(MeF, L)), by = x -> x[1])[1:10]
    @printf "Location %s : %.4f\n" loc fe
end
```